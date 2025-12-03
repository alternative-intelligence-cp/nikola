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

### Implementation

```cpp
class CustomHTTPClient {
    CURL* curl;

public:
    CustomHTTPClient() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        curl = curl_easy_init();
    }

    ~CustomHTTPClient() {
        curl_easy_cleanup(curl);
        curl_global_cleanup();
    }

    std::string get(const std::string& url,
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

    std::string post(const std::string& url,
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

// Helper functions
std::string http_get(const std::string& url,
                      const std::map<std::string, std::string>& headers = {}) {
    CustomHTTPClient client;
    return client.get(url, headers);
}

std::string http_post(const std::string& url,
                       const std::string& data,
                       const std::map<std::string, std::string>& headers = {}) {
    CustomHTTPClient client;
    return client.post(url, data, headers);
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

            case ExternalTool::HTTP_CLIENT:
                // Parse query as HTTP request
                auto [method, url, headers] = parse_http_request(query);
                if (method == "GET") {
                    return http.get(url, headers);
                } else if (method == "POST") {
                    return http.post(url, /* extract body */, headers);
                }

            default:
                throw std::runtime_error("Unknown tool");
        }
    }
};
```

---

**Cross-References:**
- See Section 11 for Orchestrator integration and tool selection logic
- See Section 9.4 for external tool integration in memory pipeline
- See Appendix C for Protocol Buffer schemas
