/**
 * @file include/nikola/inference/http_server.hpp
 * @brief Simple embedded HTTP server for Nikola inference API.
 *
 * Provides a minimal HTTP/1.1 server with no external dependencies.
 * Uses POSIX sockets and a thread pool for concurrent request handling.
 *
 * Endpoints:
 *   POST /v1/generate  — prompt → response (JSON)
 *   POST /v1/embed     — text → embedding vector (JSON)
 *   GET  /v1/health    — liveness check
 *
 * v0.2.5 — Phase 4
 */
#pragma once

#include <nikola/inference/nikola_inference.hpp>

#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace nikola::inference {

/**
 * @class HttpServer
 * @brief Serves the Nikola inference API over HTTP.
 *
 * Usage:
 * @code
 *   NikolaInference engine(cfg);
 *   HttpServer server(engine, 8080);
 *
 *   std::atomic<bool> shutdown{false};
 *   server.run(shutdown);  // blocks until shutdown is set
 * @endcode
 */
class HttpServer {
public:
    /**
     * @brief Construct the server.
     * @param engine  Inference engine (must outlive the server).
     * @param port    TCP port to listen on.
     * @param workers Number of worker threads (default: 4).
     */
    HttpServer(NikolaInference& engine, int port, int workers = 4);

    ~HttpServer();

    // Non-copyable.
    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    /**
     * @brief Start listening and block until shutdown_flag is set.
     * @param shutdown_flag  Atomic bool — set to true to stop.
     */
    void run(std::atomic<bool>& shutdown_flag);

private:
    NikolaInference& engine_;
    int port_;
    int workers_;
    int listen_fd_ = -1;

    /// Handle one client connection.
    void handle_client(int client_fd);

    /// Route a parsed HTTP request.
    struct HttpRequest {
        std::string method;
        std::string path;
        std::string body;
    };

    struct HttpResponse {
        int         status = 200;
        std::string status_text = "OK";
        std::string content_type = "application/json";
        std::string body;
    };

    HttpResponse route(const HttpRequest& req);

    // Endpoint handlers.
    HttpResponse handle_generate(const HttpRequest& req);
    HttpResponse handle_embed(const HttpRequest& req);
    HttpResponse handle_health();

    /// Send an HTTP response on a socket.
    static void send_response(int fd, const HttpResponse& resp);

    /// Parse raw HTTP request bytes.
    static HttpRequest parse_request(const std::string& raw);

    /// Simple JSON string extraction (no dependency).
    static std::string json_get_string(const std::string& json, const std::string& key);
    static int json_get_int(const std::string& json, const std::string& key, int default_val);
};

}  // namespace nikola::inference
