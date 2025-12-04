# EXTERNAL TOOL AGENTS

## 12.1 Tavily Search Client

**Purpose:** Broad web search for factual information, current events.

**API:** RESTful HTTP API requiring API key.

### Implementation

```cpp
class TavilyClient {
    std::string api_key;
    std::string base_url = "https://api.tavily.com";

public:
    TavilyClient(const std::string& key) : api_key(key) {}

    std::string search(const std::string& query, int max_results = 5) {
        // Construct request
        nlohmann::json request_body = {
            {"api_key", api_key},
            {"query", query},
            {"search_depth", "advanced"},
            {"max_results", max_results}
        };

        // HTTP POST
        auto response = http_post(base_url + "/search", request_body.dump());

        // Parse response
        auto json_response = nlohmann::json::parse(response);

        // Extract results
        std::string compiled_results;
        for (const auto& result : json_response["results"]) {
            compiled_results += result["title"].get<std::string>() + "\n";
            compiled_results += result["content"].get<std::string>() + "\n";
            compiled_results += result["url"].get<std::string>() + "\n\n";
        }

        return compiled_results;
    }
};
```

## 12.2 Firecrawl API Client

**Purpose:** Deep web scraping, convert DOM to clean Markdown.

### Implementation

```cpp
class FirecrawlClient {
    std::string api_key;
    std::string base_url = "https://api.firecrawl.dev";

public:
    FirecrawlClient(const std::string& key) : api_key(key) {}

    std::string scrape_url(const std::string& url) {
        nlohmann::json request_body = {
            {"url", url},
            {"formats", {"markdown"}},
            {"onlyMainContent", true}
        };

        // HTTP POST with auth header
        std::map<std::string, std::string> headers = {
            {"Authorization", "Bearer " + api_key},
            {"Content-Type", "application/json"}
        };

        auto response = http_post(base_url + "/v1/scrape",
                                  request_body.dump(),
                                  headers);

        auto json_response = nlohmann::json::parse(response);

        return json_response["data"]["markdown"].get<std::string>();
    }
};
```

## 12.3 Gemini CLI Tool

**Purpose:** Translation between waveforms and natural language, semantic understanding.

### Implementation

```cpp
class GeminiClient {
    std::string api_key;
    std::string base_url = "https://generativelanguage.googleapis.com/v1beta";
    std::string model = "gemini-1.5-pro";

public:
    GeminiClient(const std::string& key) : api_key(key) {}

    std::string generate(const std::string& prompt) {
        nlohmann::json request_body = {
            {"contents", {{
                {"parts", {{
                    {"text", prompt}
                }}}
            }}},
            {"generationConfig", {
                {"temperature", 0.7},
                {"maxOutputTokens", 2048}
            }}
        };

        std::string url = base_url + "/models/" + model + ":generateContent?key=" + api_key;

        auto response = http_post(url, request_body.dump());

        auto json_response = nlohmann::json::parse(response);

        return json_response["candidates"][0]["content"]["parts"][0]["text"].get<std::string>();
    }

    std::string translate_wave_to_text(const std::vector<Nit>& nonary_vector) {
        // Convert nonary to string representation
        std::string wave_str = "Nonary vector: [";
        for (const auto& nit : nonary_vector) {
            wave_str += std::to_string(static_cast<int>(nit)) + ", ";
        }
        wave_str += "]";

        std::string prompt = "Translate this nonary encoded waveform to natural language: " + wave_str;

        return generate(prompt);
    }
};
```

## 12.4 Custom HTTP Client

**Purpose:** Generic HTTP/HTTPS requests with full control (Postman-like).

All HTTP operations are asynchronous using std::future to prevent blocking the main cognitive loop during network I/O.

### Implementation

```cpp
#include <future>
#include <thread>
#include <curl/curl.h>

class CustomHTTPClient {
    CURL* curl;

public:
    CustomHTTPClient() {
        // Note: curl_global_init() must be called once at process startup in main()
        // It is NOT thread-safe and must not be called from multiple threads
        curl = curl_easy_init();
    }

    ~CustomHTTPClient() {
        curl_easy_cleanup(curl);
        // Note: curl_global_cleanup() should be called once at process shutdown in main()
    }

    // Async GET with std::future (non-blocking)
    std::future<std::string> get_async(const std::string& url,
                                         const std::map<std::string, std::string>& headers = {}) {
        return std::async(std::launch::async, [this, url, headers]() {
            return this->get_sync(url, headers);
        });
    }

    // Async POST with std::future (non-blocking)
    std::future<std::string> post_async(const std::string& url,
                                          const std::string& data,
                                          const std::map<std::string, std::string>& headers = {}) {
        return std::async(std::launch::async, [this, url, data, headers]() {
            return this->post_sync(url, data, headers);
        });
    }

    // Synchronous GET (for backward compatibility)
    std::string get_sync(const std::string& url,
                         const std::map<std::string, std::string>& headers = {}) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

        // Set headers
        struct curl_slist* header_list = nullptr;
        for (const auto& [key, value] : headers) {
            std::string header = key + ": " + value;
            header_list = curl_slist_append(header_list, header.c_str());
        }
        if (header_list) {
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
        }

        // Response buffer
        std::string response;
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        // Perform
        CURLcode res = curl_easy_perform(curl);

        if (header_list) {
            curl_slist_free_all(header_list);
        }

        if (res != CURLE_OK) {
            throw std::runtime_error("curl_easy_perform() failed: " +
                                      std::string(curl_easy_strerror(res)));
        }

        return response;
    }

    // Synchronous POST (for backward compatibility)
    std::string post_sync(const std::string& url,
                          const std::string& data,
                          const std::map<std::string, std::string>& headers = {}) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());

        // Set headers
        struct curl_slist* header_list = nullptr;
        for (const auto& [key, value] : headers) {
            std::string header = key + ": " + value;
            header_list = curl_slist_append(header_list, header.c_str());
        }
        if (header_list) {
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
        }

        // Response buffer
        std::string response;
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        // Perform
        CURLcode res = curl_easy_perform(curl);

        if (header_list) {
            curl_slist_free_all(header_list);
        }

        if (res != CURLE_OK) {
            throw std::runtime_error("curl_easy_perform() failed: " +
                                      std::string(curl_easy_strerror(res)));
        }

        return response;
    }

private:
    static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
        ((std::string*)userp)->append((char*)contents, size * nmemb);
        return size * nmemb;
    }
};

// Global helper functions - async by default (non-blocking)
std::future<std::string> http_get(const std::string& url,
                                    const std::map<std::string, std::string>& headers = {}) {
    static thread_local CustomHTTPClient client;
    return client.get_async(url, headers);
}

std::future<std::string> http_post(const std::string& url,
                                     const std::string& data,
                                     const std::map<std::string, std::string>& headers = {}) {
    static thread_local CustomHTTPClient client;
    return client.post_async(url, data, headers);
}

// Synchronous versions (for backward compatibility, use sparingly)
std::string http_get_sync(const std::string& url,
                           const std::map<std::string, std::string>& headers = {}) {
    CustomHTTPClient client;
    return client.get_sync(url, headers);
}

std::string http_post_sync(const std::string& url,
                            const std::string& data,
                            const std::map<std::string, std::string>& headers = {}) {
    CustomHTTPClient client;
    return client.post_sync(url, data, headers);
}
```

**Usage Pattern in Orchestrator:**

```cpp
// Non-blocking HTTP call - cognitive loop continues during network I/O
auto future_response = http_post(tavily_url, request_body.dump());

// Continue physics propagation while waiting for network
for (int i = 0; i < 10; ++i) {
    torus.propagate(0.001);  // Physics doesn't stall
}

// Check if response ready (non-blocking poll)
if (future_response.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
    auto response = future_response.get();
    // Process response
} else {
    // Network still in progress, continue with other work
}
```

## 12.4.1 Introspective HTTP Debugger

**[ADDENDUM]**

The specification requires a client "similar to postman". This is implemented not just as a network utility, but as a **Cognitive Tool** exposed to the Orchestrator.

### Tool Architecture: NikolaPostman

Unlike a standard curl wrapper, this tool exposes an **Inspection Interface**:

1. **Drafting Mode:** The AI creates a RequestObject
2. **Simulation:** The AI can "dry run" the request - the system runs local heuristics to predict if the request will fail (e.g., checking for missing Auth headers, malformed JSON bodies) before hitting the network
3. **Introspection:** The AI receives a structured breakdown of the TCP handshake, TLS negotiation, and raw headers, allowing it to debug connection issues "consciously" rather than just receiving a Connection Failed error

### Data Structure (Protocol Buffer)

```protobuf
message HTTPInspectionReport {
   string stage = 1;          // e.g., "DNS_LOOKUP", "TLS_HANDSHAKE"
   double latency_ms = 2;
   map<string, string> request_headers = 3;
   string raw_wire_data = 4;  // Hex dump of what was actually sent
   repeated string heuristic_warnings = 5; // e.g., "Content-Type missing"
}
```

## 12.5 Implementation Details

### HTTP Request Parser

```cpp
// Production-grade HTTP parsing using cpp-httplib
// This library provides RFC 7230 compliant parsing with support for:
//   - Chunked transfer encoding
//   - Multipart bodies
//   - Multi-line headers (folding)
//   - HTTP/1.1 pipelining
//
// Security note: Manual string parsing using std::getline is not permitted
// due to vulnerabilities (HTTP Request Smuggling, malformed header crashes).
//
// cpp-httplib is header-only with no build dependencies.
// Add to CMakeLists.txt:
//   find_package(httplib CONFIG REQUIRED)
//   target_link_libraries(nikola PRIVATE httplib::httplib)

#include <httplib.h>

struct HTTPRequest {
    std::string method;
    std::string url;
    std::map<std::string, std::string> headers;
    std::string body;
};

// Parse HTTP request using cpp-httplib for RFC 7230 compliance
HTTPRequest parse_http_request(const std::string& raw_request) {
    HTTPRequest req;

    // Create a temporary parser instance
    httplib::detail::BufferStream buffer_stream;
    buffer_stream.write(raw_request.c_str(), raw_request.size());

    // Use httplib's internal parser for production-grade parsing
    httplib::Request parsed_req;
    httplib::detail::read_headers(buffer_stream, parsed_req.headers);

    // Extract method and path from request line
    std::istringstream first_line(raw_request.substr(0, raw_request.find('\n')));
    std::string http_version;
    first_line >> req.method >> req.url >> http_version;

    // Copy headers
    for (const auto& header : parsed_req.headers) {
        req.headers[header.first] = header.second;
    }

    // Extract body (handles chunked encoding, content-length, etc.)
    size_t header_end = raw_request.find("\r\n\r\n");
    if (header_end != std::string::npos) {
        req.body = raw_request.substr(header_end + 4);

        // Handle Transfer-Encoding: chunked
        auto te_iter = req.headers.find("Transfer-Encoding");
        if (te_iter != req.headers.end() && te_iter->second == "chunked") {
            req.body = httplib::detail::decode_chunked_encoding(req.body);
        }
    }

    return req;
}

// Alternative Option 2: llhttp (faster, C-based parser used by Node.js)
// Requires linking: -lllhttp
// See: https://github.com/nodejs/llhttp
//
// #include <llhttp.h>
//
// struct HTTPParserContext {
//     HTTPRequest* req;
//     std::string current_header_field;
// };
//
// int on_url(llhttp_t* parser, const char* at, size_t length) {
//     auto* ctx = static_cast<HTTPParserContext*>(parser->data);
//     ctx->req->url.assign(at, length);
//     return 0;
// }
//
// int on_header_field(llhttp_t* parser, const char* at, size_t length) {
//     auto* ctx = static_cast<HTTPParserContext*>(parser->data);
//     ctx->current_header_field.assign(at, length);
//     return 0;
// }
//
// int on_header_value(llhttp_t* parser, const char* at, size_t length) {
//     auto* ctx = static_cast<HTTPParserContext*>(parser->data);
//     ctx->req->headers[ctx->current_header_field].assign(at, length);
//     return 0;
// }
//
// int on_body(llhttp_t* parser, const char* at, size_t length) {
//     auto* ctx = static_cast<HTTPParserContext*>(parser->data);
//     ctx->req->body.append(at, length);
//     return 0;
// }
//
// HTTPRequest parse_http_request_llhttp(const std::string& raw_request) {
//     HTTPRequest req;
//     HTTPParserContext ctx{&req, ""};
//
//     llhttp_t parser;
//     llhttp_settings_t settings;
//
//     llhttp_settings_init(&settings);
//     settings.on_url = on_url;
//     settings.on_header_field = on_header_field;
//     settings.on_header_value = on_header_value;
//     settings.on_body = on_body;
//
//     llhttp_init(&parser, HTTP_REQUEST, &settings);
//     parser.data = &ctx;
//
//     llhttp_execute(&parser, raw_request.c_str(), raw_request.size());
//
//     // Extract method from parser
//     req.method = llhttp_method_name(static_cast<llhttp_method_t>(parser.method));
//
//     return req;
// }
```

### Tool Manager

```cpp
class ExternalToolManager {
    TavilyClient tavily;
    FirecrawlClient firecrawl;
    GeminiClient gemini;
    CustomHTTPClient http;

public:
    ExternalToolManager(const std::string& tavily_key,
                         const std::string& firecrawl_key,
                         const std::string& gemini_key)
        : tavily(tavily_key), firecrawl(firecrawl_key), gemini(gemini_key) {}

    std::string fetch(ExternalTool tool, const std::string& query) {
        switch (tool) {
            case ExternalTool::TAVILY:
                return tavily.search(query);

            case ExternalTool::FIRECRAWL:
                // Extract URL from query
                auto url = extract_url(query);
                return firecrawl.scrape_url(url);

            case ExternalTool::GEMINI:
                return gemini.generate(query);

            case ExternalTool::HTTP_CLIENT: {
                // Parse query as HTTP request (format: "METHOD URL\nHeader: Value\n\nBody")
                HTTPRequest req = parse_http_request(query);
                if (req.method == "GET") {
                    return http.get(req.url, req.headers);
                } else if (req.method == "POST") {
                    return http.post(req.url, req.body, req.headers);
                } else if (req.method == "PUT") {
                    return http.put(req.url, req.body, req.headers);
                }
                throw std::runtime_error("Unsupported HTTP method: " + req.method);
            }

            default:
                throw std::runtime_error("Unknown tool");
        }
    }
};
```

## 12.6 Main Entry Point - API Key Loading

**Purpose:** Load external tool API keys from environment variables and instantiate ExternalToolManager.

**Implementation:**

```cpp
// File: src/main.cpp

#include "nikola/infrastructure/external_tools.hpp"
#include "nikola/infrastructure/orchestrator.hpp"
#include <iostream>
#include <cstdlib>

std::string get_required_env(const char* var_name) {
    const char* value = std::getenv(var_name);
    if (!value || std::string(value).empty()) {
        std::cerr << "[FATAL] Required environment variable " << var_name
                  << " is not set" << std::endl;
        std::exit(1);
    }
    return std::string(value);
}

std::string get_optional_env(const char* var_name, const std::string& default_value = "") {
    const char* value = std::getenv(var_name);
    return value ? std::string(value) : default_value;
}

int main(int argc, char* argv[]) {
    std::cout << "[NIKOLA] Initializing Nikola Model v0.0.4..." << std::endl;

    // Load API keys from environment variables
    std::string tavily_key = get_required_env("TAVILY_API_KEY");
    std::string firecrawl_key = get_required_env("FIRECRAWL_API_KEY");
    std::string gemini_key = get_required_env("GEMINI_API_KEY");

    std::cout << "[CONFIG] External tool API keys loaded successfully" << std::endl;

    // Initialize External Tool Manager
    ExternalToolManager tool_manager(tavily_key, firecrawl_key, gemini_key);

    // Initialize Orchestrator with tool manager
    Orchestrator orchestrator(tool_manager);

    std::cout << "[NIKOLA] System initialized. Ready for queries." << std::endl;

    // Main event loop
    orchestrator.run();

    return 0;
}
```

**Environment Variable Validation:**

```cpp
// File: src/config/env_validator.hpp
#pragma once

#include <string>
#include <vector>
#include <map>

class EnvironmentValidator {
public:
    struct ValidationResult {
        bool success;
        std::vector<std::string> missing_vars;
        std::vector<std::string> warnings;
    };

    static ValidationResult validate_required_vars() {
        ValidationResult result;
        result.success = true;

        const std::vector<std::string> required_vars = {
            "TAVILY_API_KEY",
            "FIRECRAWL_API_KEY",
            "GEMINI_API_KEY"
        };

        for (const auto& var : required_vars) {
            const char* value = std::getenv(var.c_str());
            if (!value || std::string(value).empty()) {
                result.missing_vars.push_back(var);
                result.success = false;
            }
        }

        return result;
    }

    static void print_validation_errors(const ValidationResult& result) {
        if (!result.success) {
            std::cerr << "[ERROR] Missing required environment variables:" << std::endl;
            for (const auto& var : result.missing_vars) {
                std::cerr << "  - " << var << std::endl;
            }
            std::cerr << "\nPlease set these variables before starting Nikola:" << std::endl;
            std::cerr << "  export TAVILY_API_KEY=your_key_here" << std::endl;
            std::cerr << "  export FIRECRAWL_API_KEY=your_key_here" << std::endl;
            std::cerr << "  export GEMINI_API_KEY=your_key_here" << std::endl;
        }
    }
};
```

**Docker Integration:**

The environment variables are passed through Docker Compose (see Section 25.1):

```yaml
# docker-compose.yml
services:
  nikola-spine:
    environment:
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - FIRECRAWL_API_KEY=${FIRECRAWL_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
```

**Startup Validation:**

```cpp
// Enhanced main.cpp with validation

int main(int argc, char* argv[]) {
    std::cout << "[NIKOLA] Initializing Nikola Model v0.0.4..." << std::endl;

    // Validate environment
    auto validation = EnvironmentValidator::validate_required_vars();
    if (!validation.success) {
        EnvironmentValidator::print_validation_errors(validation);
        return 1;
    }

    // Load API keys (now guaranteed to exist)
    std::string tavily_key = std::getenv("TAVILY_API_KEY");
    std::string firecrawl_key = std::getenv("FIRECRAWL_API_KEY");
    std::string gemini_key = std::getenv("GEMINI_API_KEY");

    // Initialize system
    ExternalToolManager tool_manager(tavily_key, firecrawl_key, gemini_key);
    Orchestrator orchestrator(tool_manager);

    std::cout << "[NIKOLA] System initialized. Ready." << std::endl;
    orchestrator.run();

    return 0;
}
```

---

**Cross-References:**
- See Section 11 for Orchestrator integration and tool selection logic
- See Section 9.4 for external tool integration in memory pipeline
- See Appendix C for Protocol Buffer schemas
