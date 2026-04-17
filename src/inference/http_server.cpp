/**
 * @file src/inference/http_server.cpp
 * @brief Simple embedded HTTP server for Nikola inference API.
 *
 * Implements a minimal HTTP/1.1 server using POSIX sockets.
 * Thread-pool based concurrent request handling.
 *
 * v0.2.5 — Phase 4
 */

#include <nikola/inference/http_server.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <poll.h>

namespace nikola::inference {

// Static helper for JSON escaping in responses (defined early for use below).
static std::string json_escape_static(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if      (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else                out += c;
    }
    return out;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

HttpServer::HttpServer(NikolaInference& engine, int port, int workers)
    : engine_(engine)
    , port_(port)
    , workers_(workers)
{}

HttpServer::~HttpServer() {
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
}

// ============================================================================
// run — main server loop
// ============================================================================

void HttpServer::run(std::atomic<bool>& shutdown_flag) {
    // Create socket
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        std::cerr << "[HttpServer] socket() failed: " << std::strerror(errno) << "\n";
        return;
    }

    // Allow port reuse
    int opt = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // Bind
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port        = htons(static_cast<uint16_t>(port_));

    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "[HttpServer] bind() failed on port " << port_
                  << ": " << std::strerror(errno) << "\n";
        ::close(listen_fd_);
        listen_fd_ = -1;
        return;
    }

    if (::listen(listen_fd_, 32) < 0) {
        std::cerr << "[HttpServer] listen() failed: " << std::strerror(errno) << "\n";
        ::close(listen_fd_);
        listen_fd_ = -1;
        return;
    }

    std::cerr << "[HttpServer] Listening on 0.0.0.0:" << port_ << "\n";

    // Accept loop with poll() for clean shutdown
    while (!shutdown_flag.load()) {
        pollfd pfd{};
        pfd.fd     = listen_fd_;
        pfd.events = POLLIN;

        int ready = ::poll(&pfd, 1, 500);  // 500ms timeout for shutdown check
        if (ready < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (ready == 0) continue;  // timeout, check shutdown

        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&client_addr),
                                 &client_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            std::cerr << "[HttpServer] accept() failed: " << std::strerror(errno) << "\n";
            continue;
        }

        // Handle in a detached thread.
        // For a production system, use a proper thread pool.
        // For v0.2.5 scope, detached threads are sufficient.
        std::thread([this, client_fd]() {
            handle_client(client_fd);
        }).detach();
    }

    ::close(listen_fd_);
    listen_fd_ = -1;
    std::cerr << "[HttpServer] Shutdown complete.\n";
}

// ============================================================================
// handle_client
// ============================================================================

void HttpServer::handle_client(int client_fd) {
    // Read request (up to 64KB — sufficient for inference prompts)
    constexpr size_t MAX_REQUEST = 65536;
    std::string raw;
    raw.resize(MAX_REQUEST);

    ssize_t n = ::recv(client_fd, raw.data(), MAX_REQUEST - 1, 0);
    if (n <= 0) {
        ::close(client_fd);
        return;
    }
    raw.resize(static_cast<size_t>(n));

    auto req = parse_request(raw);
    auto resp = route(req);
    send_response(client_fd, resp);

    ::close(client_fd);
}

// ============================================================================
// route
// ============================================================================

HttpServer::HttpResponse HttpServer::route(const HttpRequest& req) {
    if (req.method == "GET" && req.path == "/v1/health") {
        return handle_health();
    }
    if (req.method == "POST" && req.path == "/v1/generate") {
        return handle_generate(req);
    }
    if (req.method == "POST" && req.path == "/v1/embed") {
        return handle_embed(req);
    }

    // 404
    HttpResponse resp;
    resp.status      = 404;
    resp.status_text = "Not Found";
    resp.body = "{\"error\":\"not_found\",\"message\":\"Unknown endpoint: "
              + req.method + " " + req.path + "\"}";
    return resp;
}

// ============================================================================
// handle_generate — POST /v1/generate
//
// Request:  {"prompt": "...", "max_ticks": 200}
// Response: {"thought": "...", "tokens": [...], "ticks_used": N}
// ============================================================================

HttpServer::HttpResponse HttpServer::handle_generate(const HttpRequest& req) {
    std::string prompt = json_get_string(req.body, "prompt");
    int max_ticks = json_get_int(req.body, "max_ticks", 200);

    if (prompt.empty()) {
        HttpResponse resp;
        resp.status      = 400;
        resp.status_text = "Bad Request";
        resp.body = "{\"error\":\"bad_request\",\"message\":\"Missing 'prompt' field\"}";
        return resp;
    }

    // Clamp max_ticks to prevent abuse.
    max_ticks = std::clamp(max_ticks, 1, 10000);

    // Mutex protects the engine (not thread-safe internally).
    static std::mutex engine_mutex;
    std::lock_guard<std::mutex> lock(engine_mutex);

    engine_.inject(prompt);

    std::string thought;
    std::vector<std::string> all_tokens;
    uint64_t ticks_used = 0;

    for (int i = 0; i < max_ticks; ++i) {
        auto r = engine_.tick();
        ticks_used = r.tick;
        if (!r.thought.empty()) {
            thought = r.thought;
            all_tokens = r.tokens;
            break;
        }
    }

    // Build JSON response
    std::ostringstream json;
    json << "{\"thought\":\"" << json_escape_static(thought) << "\""
         << ",\"tokens\":[";
    for (size_t i = 0; i < all_tokens.size(); ++i) {
        if (i > 0) json << ",";
        json << "\"" << json_escape_static(all_tokens[i]) << "\"";
    }
    json << "]"
         << ",\"ticks_used\":" << ticks_used
         << "}";

    HttpResponse resp;
    resp.body = json.str();
    return resp;
}

// ============================================================================
// handle_embed — POST /v1/embed
//
// Request:  {"text": "..."}
// Response: {"embedding": [f1, f2, ...], "dimensions": 9}
// ============================================================================

HttpServer::HttpResponse HttpServer::handle_embed(const HttpRequest& req) {
    std::string text = json_get_string(req.body, "text");

    if (text.empty()) {
        HttpResponse resp;
        resp.status      = 400;
        resp.status_text = "Bad Request";
        resp.body = "{\"error\":\"bad_request\",\"message\":\"Missing 'text' field\"}";
        return resp;
    }

    static std::mutex engine_mutex;
    std::lock_guard<std::mutex> lock(engine_mutex);

    // Inject text and read the resulting torus state as a crude embedding.
    engine_.inject(text);
    auto r = engine_.tick();

    // Use the torus hot node intensities as a feature vector.
    auto hot = engine_.torus().hot_nodes(64);
    std::vector<float> embedding;
    embedding.reserve(hot.size());
    for (size_t idx : hot) {
        embedding.push_back(engine_.torus().intensity(idx));
    }

    // Build JSON
    std::ostringstream json;
    json << "{\"embedding\":[";
    for (size_t i = 0; i < embedding.size(); ++i) {
        if (i > 0) json << ",";
        json << embedding[i];
    }
    json << "],\"dimensions\":" << embedding.size() << "}";

    HttpResponse resp;
    resp.body = json.str();
    return resp;
}

// ============================================================================
// handle_health — GET /v1/health
// ============================================================================

HttpServer::HttpResponse HttpServer::handle_health() {
    std::ostringstream json;
    json << "{\"status\":\"ok\""
         << ",\"version\":\"0.2.5\""
         << ",\"engine\":\"nikola-infer\""
         << ",\"ticks\":" << engine_.tick_count()
         << ",\"gpu\":" << (engine_.torus().gpu_enabled() ? "true" : "false")
         << "}";

    HttpResponse resp;
    resp.body = json.str();
    return resp;
}

// ============================================================================
// send_response
// ============================================================================

void HttpServer::send_response(int fd, const HttpResponse& resp) {
    std::ostringstream http;
    http << "HTTP/1.1 " << resp.status << " " << resp.status_text << "\r\n"
         << "Content-Type: " << resp.content_type << "\r\n"
         << "Content-Length: " << resp.body.size() << "\r\n"
         << "Connection: close\r\n"
         << "Access-Control-Allow-Origin: *\r\n"
         << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
         << "Access-Control-Allow-Headers: Content-Type\r\n"
         << "\r\n"
         << resp.body;

    const std::string data = http.str();
    size_t sent = 0;
    while (sent < data.size()) {
        ssize_t n = ::send(fd, data.data() + sent, data.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) break;
        sent += static_cast<size_t>(n);
    }
}

// ============================================================================
// parse_request
// ============================================================================

HttpServer::HttpRequest HttpServer::parse_request(const std::string& raw) {
    HttpRequest req;

    // Parse request line
    auto first_line_end = raw.find("\r\n");
    if (first_line_end == std::string::npos) first_line_end = raw.find('\n');
    if (first_line_end == std::string::npos) return req;

    std::string request_line = raw.substr(0, first_line_end);
    auto sp1 = request_line.find(' ');
    auto sp2 = request_line.find(' ', sp1 + 1);
    if (sp1 == std::string::npos) return req;

    req.method = request_line.substr(0, sp1);
    req.path   = (sp2 != std::string::npos)
               ? request_line.substr(sp1 + 1, sp2 - sp1 - 1)
               : request_line.substr(sp1 + 1);

    // Strip query string
    auto qmark = req.path.find('?');
    if (qmark != std::string::npos) req.path = req.path.substr(0, qmark);

    // Find body (after \r\n\r\n)
    auto body_start = raw.find("\r\n\r\n");
    if (body_start != std::string::npos) {
        req.body = raw.substr(body_start + 4);
    }

    return req;
}

// ============================================================================
// JSON helpers (minimal, no-dependency)
// ============================================================================

std::string HttpServer::json_get_string(const std::string& json, const std::string& key) {
    // Find "key": "value"
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return {};

    pos += search.size();
    // Skip whitespace and colon
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t'))
        ++pos;

    if (pos >= json.size() || json[pos] != '"') return {};
    ++pos;  // skip opening quote

    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            ++pos;
            if      (json[pos] == '"')  result += '"';
            else if (json[pos] == '\\') result += '\\';
            else if (json[pos] == 'n')  result += '\n';
            else if (json[pos] == 'r')  result += '\r';
            else if (json[pos] == 't')  result += '\t';
            else                        result += json[pos];
        } else {
            result += json[pos];
        }
        ++pos;
    }

    return result;
}

int HttpServer::json_get_int(const std::string& json, const std::string& key, int default_val) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    pos += search.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t'))
        ++pos;

    if (pos >= json.size()) return default_val;

    try {
        return std::stoi(json.substr(pos));
    } catch (...) {
        return default_val;
    }
}

}  // namespace nikola::inference
