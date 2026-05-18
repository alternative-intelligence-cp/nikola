/**
 * @file src/twi_ctl.cpp
 * @brief v0.3.6 QoL slice 2: minimal twi-ctl controller CLI.
 */

#include <nikola/cli/twi_ctl.hpp>
#include <nikola/security/ironhouse.hpp>

#include <fstream>
#include <iostream>
#include <sstream>

namespace {

using nikola::cli::twi_ctl::Command;
using nikola::cli::twi_ctl::ParsedCommand;

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if      (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

void print_help() {
    std::cout
        << "twi-ctl - Toroidal Waveform Intelligence Controller\n\n"
        << "USAGE:\n"
        << "  twi-ctl <command> [options]\n\n"
        << "COMMANDS:\n"
        << "  init                    Generate CurveZMQ client keypair\n"
        << "  query  <text>           Build RCIS INJECT_STIMULUS request\n"
        << "  ingest <text>|--file    Build RCIS INJECT_STIMULUS request\n"
        << "  status                  Build RCIS FETCH_STATE request\n"
        << "  ping                    Build RCIS PING request\n"
        << "  help                    Show this message\n\n"
        << "COMMON OPTIONS:\n"
        << "  --json                  Print machine-readable JSON\n"
        << "  --dry-run               Print built RCIS request and exit (no network)\n"
        << "  --endpoint,-e <uri>     RCIS control endpoint [tcp://127.0.0.1:5556]\n"
        << "  --timeout-ms <int>      Send/recv timeout in ms [3000]\n"
        << "  --threshold,-t <float>  Query threshold metadata [0.7]\n"
        << "  --steps,-s <int>        Query propagation steps [100]\n"
        << "  --file,-f <path>        Ingest payload from file\n"
        << "  --type <kind>           Ingest type hint [text]\n";
}

bool read_file_to_string(const std::string& path, std::string& out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return false;
    std::ostringstream ss;
    ss << ifs.rdbuf();
    out = ss.str();
    return true;
}

int print_request(const ParsedCommand& parsed, const ::nikola::spine::RCISRequest& req) {
    if (parsed.json) {
        std::cout << "{"
                  << "\"request_id\":\"" << json_escape(req.request_id()) << "\","
                  << "\"type\":" << static_cast<int>(req.type()) << ","
                  << "\"stimulus_text\":\"" << json_escape(req.stimulus_text()) << "\","
                  << "\"threshold\":" << parsed.threshold << ","
                  << "\"steps\":" << parsed.steps << ","
                  << "\"endpoint\":\"" << json_escape(parsed.endpoint) << "\","
                  << "\"timeout_ms\":" << parsed.timeout_ms << ","
                  << "\"dry_run\":" << (parsed.dry_run ? "true" : "false")
                  << "}\n";
    } else {
        std::cout << "[twi-ctl] Built RCIS request\n"
                  << "  request_id: " << req.request_id() << "\n"
                  << "  type:       " << req.type() << "\n";
        if (!req.stimulus_text().empty()) {
            std::cout << "  stimulus:   " << req.stimulus_text() << "\n";
        }
        if (parsed.command == Command::QUERY) {
            std::cout << "  threshold:  " << parsed.threshold << "\n"
                      << "  steps:      " << parsed.steps << "\n";
        }
        std::cout << "  endpoint:   " << parsed.endpoint << "\n"
                  << "  timeout-ms: " << parsed.timeout_ms << "\n"
                  << "  mode:       " << (parsed.dry_run ? "dry-run" : "live") << "\n";
    }
    return 0;
}

int print_response(const ParsedCommand& parsed, const ::nikola::spine::RCISResponse& resp) {
    if (parsed.json) {
        std::cout << "{"
                  << "\"request_id\":\"" << json_escape(resp.request_id()) << "\","
                  << "\"status\":" << static_cast<int>(resp.status()) << ","
                  << "\"message\":\"" << json_escape(resp.message()) << "\","
                  << "\"state_json\":\"" << json_escape(resp.state_json()) << "\""
                  << "}\n";
    } else {
        std::cout << "[twi-ctl] RCIS response\n"
                  << "  request_id: " << resp.request_id() << "\n"
                  << "  status:     " << resp.status() << "\n"
                  << "  message:    " << resp.message() << "\n";
        if (!resp.state_json().empty()) {
            std::cout << "  state_json: " << resp.state_json() << "\n";
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    auto parsed = nikola::cli::twi_ctl::parse_args(argc, argv);
    if (!parsed.valid) {
        if (!parsed.error.empty()) {
            std::cerr << "[twi-ctl] error: " << parsed.error << "\n";
        }
        print_help();
        return parsed.error.empty() ? 0 : 1;
    }

    if (parsed.command == Command::HELP) {
        print_help();
        return 0;
    }

    if (parsed.command == Command::INIT) {
        const auto kp = nikola::security::generate_ironhouse_keypair();
        if (parsed.json) {
            std::cout << "{\"public_key\":\"" << kp.pub() << "\",\"secret_key\":\"" << kp.sec() << "\"}\n";
        } else {
            std::cout << "[twi-ctl] Generated CurveZMQ keypair\n"
                      << "  public_key: " << kp.pub() << "\n"
                      << "  secret_key: " << kp.sec() << "\n";
        }
        return 0;
    }

    // Ingest file path -> payload text
    if (parsed.command == Command::INGEST && !parsed.file_path.empty()) {
        if (!read_file_to_string(parsed.file_path, parsed.payload)) {
            std::cerr << "[twi-ctl] error: failed to read file: " << parsed.file_path << "\n";
            return 1;
        }
    }

    const auto req = nikola::cli::twi_ctl::build_rcis_request(parsed);
    if (!req.has_value()) {
        std::cerr << "[twi-ctl] error: command does not map to an RCIS request\n";
        return 1;
    }

    if (parsed.dry_run) {
        return print_request(parsed, *req);
    }

    ::nikola::spine::RCISResponse resp;
    std::string err;
    if (!nikola::cli::twi_ctl::rcis_roundtrip(*req, resp,
                                              parsed.endpoint,
                                              parsed.timeout_ms,
                                              &err)) {
        std::cerr << "[twi-ctl] error: transport failure: " << err << "\n";
        return 2;
    }

    return print_response(parsed, resp);
}
