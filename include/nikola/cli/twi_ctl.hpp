#pragma once
/**
 * @file include/nikola/cli/twi_ctl.hpp
 * @brief v0.3.6 QoL slice 2: twi-ctl core command parser + RCIS request mapper.
 */

#include <nikola/infrastructure/rcis_protocol.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <zmq.hpp>

namespace nikola::cli::twi_ctl {

enum class Command : uint8_t {
    HELP,
    INIT,
    QUERY,
    INGEST,
    STATUS,
    PING,
    UNKNOWN,
};

struct ParsedCommand {
    Command     command   = Command::UNKNOWN;
    std::string payload;      ///< query/ingest text (or loaded file content)
    std::string file_path;    ///< ingest --file path (optional)
    std::string ingest_type = "text";

    float       threshold = 0.7f;
    int         steps     = 100;
    bool        json      = false;
    bool        dry_run   = false;
    std::string endpoint  = "tcp://127.0.0.1:5556";
    int         timeout_ms = 3000;

    bool        valid     = false;
    std::string error;
};

[[nodiscard]] inline const char* command_name(Command c) noexcept {
    switch (c) {
        case Command::HELP:   return "help";
        case Command::INIT:   return "init";
        case Command::QUERY:  return "query";
        case Command::INGEST: return "ingest";
        case Command::STATUS: return "status";
        case Command::PING:   return "ping";
        default:              return "unknown";
    }
}

[[nodiscard]] inline Command parse_command_name(std::string_view s) noexcept {
    if (s == "help")   return Command::HELP;
    if (s == "init")   return Command::INIT;
    if (s == "query")  return Command::QUERY;
    if (s == "ingest") return Command::INGEST;
    if (s == "status") return Command::STATUS;
    if (s == "ping")   return Command::PING;
    return Command::UNKNOWN;
}

[[nodiscard]] inline ParsedCommand parse_args(int argc, char** argv) {
    ParsedCommand out;

    if (argc < 2) {
        out.command = Command::HELP;
        out.valid = true;
        return out;
    }

    out.command = parse_command_name(argv[1]);
    if (out.command == Command::UNKNOWN) {
        out.error = "unknown command";
        return out;
    }

    // Parse options and positional payload (for query/ingest)
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto need_next = [&](const char* opt) -> std::optional<std::string> {
            if (i + 1 >= argc) {
                out.error = std::string(opt) + " requires an argument";
                return std::nullopt;
            }
            return std::string(argv[++i]);
        };

        if (a == "--json") {
            out.json = true;
        } else if (a == "--dry-run") {
            out.dry_run = true;
        } else if (a == "--endpoint" || a == "-e") {
            auto v = need_next("--endpoint");
            if (!v) return out;
            out.endpoint = *v;
        } else if (a == "--timeout-ms") {
            auto v = need_next("--timeout-ms");
            if (!v) return out;
            out.timeout_ms = std::stoi(*v);
        } else if (a == "--threshold" || a == "-t") {
            auto v = need_next("--threshold");
            if (!v) return out;
            out.threshold = std::stof(*v);
        } else if (a == "--steps" || a == "-s") {
            auto v = need_next("--steps");
            if (!v) return out;
            out.steps = std::stoi(*v);
        } else if (a == "--file" || a == "-f") {
            auto v = need_next("--file");
            if (!v) return out;
            out.file_path = *v;
        } else if (a == "--type") {
            auto v = need_next("--type");
            if (!v) return out;
            out.ingest_type = *v;
        } else if (!a.empty() && a[0] == '-') {
            out.error = "unknown option: " + a;
            return out;
        } else {
            if (!out.payload.empty()) out.payload += ' ';
            out.payload += a;
        }
    }

    if (out.threshold < 0.0f || out.threshold > 1.0f) {
        out.error = "threshold must be in [0,1]";
        return out;
    }
    if (out.steps <= 0) {
        out.error = "steps must be > 0";
        return out;
    }
    if (out.timeout_ms <= 0) {
        out.error = "timeout-ms must be > 0";
        return out;
    }

    // Command-specific validation
    if (out.command == Command::QUERY && out.payload.empty()) {
        out.error = "query requires text payload";
        return out;
    }
    if (out.command == Command::INGEST && out.payload.empty() && out.file_path.empty()) {
        out.error = "ingest requires text payload or --file";
        return out;
    }

    out.valid = true;
    return out;
}

[[nodiscard]] inline std::string make_request_id(Command c) {
    return std::string(command_name(c)) + "-" +
           std::to_string(infrastructure::rcis::now_ns());
}

[[nodiscard]] inline std::optional<::nikola::spine::RCISRequest>
build_rcis_request(const ParsedCommand& cmd) {
    if (!cmd.valid) return std::nullopt;

    ::nikola::spine::RCISRequest req;
    req.set_request_id(make_request_id(cmd.command));
    req.set_timestamp_ns(infrastructure::rcis::now_ns());

    switch (cmd.command) {
        case Command::QUERY:
            req.set_type(::nikola::spine::RCISRequest::INJECT_STIMULUS);
            req.set_stimulus_text(cmd.payload);
            return req;
        case Command::INGEST:
            req.set_type(::nikola::spine::RCISRequest::INJECT_STIMULUS);
            req.set_stimulus_text(cmd.payload);
            return req;
        case Command::STATUS:
            req.set_type(::nikola::spine::RCISRequest::FETCH_STATE);
            return req;
        case Command::PING:
            req.set_type(::nikola::spine::RCISRequest::PING);
            return req;
        default:
            break;
    }
    return std::nullopt;
}

[[nodiscard]] inline bool rcis_roundtrip(const ::nikola::spine::RCISRequest& request,
                                         ::nikola::spine::RCISResponse& response,
                                         std::string_view endpoint,
                                         int timeout_ms,
                                         std::string* error = nullptr) {
    try {
        zmq::context_t ctx(1);
        zmq::socket_t sock(ctx, zmq::socket_type::req);
        sock.set(zmq::sockopt::rcvtimeo, timeout_ms);
        sock.set(zmq::sockopt::sndtimeo, timeout_ms);
        sock.connect(std::string(endpoint));

        infrastructure::rcis::send_proto(sock, request);
        if (!infrastructure::rcis::recv_proto(sock, response)) {
            if (error) *error = "recv timeout or invalid response";
            return false;
        }
        return true;
    } catch (const std::exception& ex) {
        if (error) *error = ex.what();
        return false;
    }
}

} // namespace nikola::cli::twi_ctl
