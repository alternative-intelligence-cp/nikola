# AUTONOMOUS INGESTION PIPELINE

## 16.1 Directory Watching with inotify

**Watched Directory:** `/var/lib/nikola/ingest/`

**Events:** `IN_CLOSE_WRITE`, `IN_MOVED_TO`

### Implementation

```cpp
#include <sys/inotify.h>
#include <unistd.h>

class IngestionSentinel {
    int inotify_fd = -1;
    int watch_descriptor = -1;
    std::string watch_path = "/var/lib/nikola/ingest/";

    ThreadSafeQueue<std::filesystem::path> ingest_queue;
    std::thread watch_thread;
    std::thread digester_thread;
    std::atomic<bool> running{true};

public:
    IngestionSentinel() {
        // Initialize inotify
        inotify_fd = inotify_init1(IN_NONBLOCK);
        if (inotify_fd < 0) {
            throw std::runtime_error("Failed to initialize inotify");
        }

        // Add watch
        watch_descriptor = inotify_add_watch(inotify_fd,
                                              watch_path.c_str(),
                                              IN_CLOSE_WRITE | IN_MOVED_TO);

        // Start threads
        watch_thread = std::thread(&IngestionSentinel::watch_loop, this);
        digester_thread = std::thread(&IngestionSentinel::digester_loop, this);
    }

    ~IngestionSentinel() {
        running = false;

        if (watch_thread.joinable()) watch_thread.join();
        if (digester_thread.joinable()) digester_thread.join();

        if (watch_descriptor >= 0) {
            inotify_rm_watch(inotify_fd, watch_descriptor);
        }
        if (inotify_fd >= 0) {
            close(inotify_fd);
        }
    }

private:
    void watch_loop() {
        constexpr size_t BUF_LEN = 4096;
        char buffer[BUF_LEN];

        while (running) {
            ssize_t length = read(inotify_fd, buffer, BUF_LEN);

            if (length < 0) {
                if (errno == EAGAIN) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
                break;
            }

            // Parse events
            for (char* ptr = buffer; ptr < buffer + length; ) {
                struct inotify_event* event = (struct inotify_event*)ptr;

                if (event->len > 0 && !(event->mask & IN_ISDIR)) {
                    std::filesystem::path file_path = watch_path;
                    file_path /= event->name;

                    std::cout << "[INGEST] Detected: " << file_path << std::endl;

                    ingest_queue.push(file_path);
                }

                ptr += sizeof(struct inotify_event) + event->len;
            }
        }
    }

    void digester_loop() {
        while (running) {
            auto file_path_opt = ingest_queue.pop_with_timeout(std::chrono::seconds(1));

            if (file_path_opt) {
                process_file(*file_path_opt);
            }
        }
    }

    void process_file(const std::filesystem::path& file_path);
};
```

## 16.2 MIME Detection with libmagic

**Purpose:** Identify file type by content, not extension

### Implementation

```cpp
#include <magic.h>

std::string detect_mime_type(const std::filesystem::path& file_path) {
    magic_t magic_cookie = magic_open(MAGIC_MIME_TYPE);
    if (!magic_cookie) {
        throw std::runtime_error("Failed to initialize libmagic");
    }

    magic_load(magic_cookie, nullptr);

    const char* mime = magic_file(magic_cookie, file_path.c_str());
    std::string result(mime ? mime : "application/octet-stream");

    magic_close(magic_cookie);

    return result;
}
```

## 16.3 File Processing Pipeline

### Pipeline

```
File Detected
    ↓
MIME Detection
    ↓
Routing by Type
    ├─→ text/* → Direct read
    ├─→ application/pdf → PDF extraction (poppler)
    ├─→ application/zip → Decompress & recursive
    └─→ Other → Skip or Gemini analysis
    ↓
Text Extraction
    ↓
Chunking (if large)
    ↓
Embedding (Nonary Embedder)
    ↓
Storage in Torus
    ↓
Archive Original File
```

### Implementation

```cpp
void IngestionSentinel::process_file(const std::filesystem::path& file_path) {
    try {
        // 1. Detect MIME type
        std::string mime = detect_mime_type(file_path);
        std::cout << "[INGEST] MIME: " << mime << std::endl;

        // 2. Route by type
        std::string content;

        if (mime.starts_with("text/")) {
            // Direct read
            std::ifstream file(file_path);
            content = std::string(std::istreambuf_iterator<char>(file),
                                   std::istreambuf_iterator<char>());
        } else if (mime == "application/pdf") {
            // Extract using poppler (via executor)
            content = extract_pdf_text(file_path);
        } else if (mime == "application/zip" || mime == "application/x-tar") {
            // Decompress and recursively ingest
            auto extracted_dir = decompress_archive(file_path);
            ingest_directory_recursive(extracted_dir);
            return;
        } else {
            std::cout << "[INGEST] Skipping unsupported type: " << mime << std::endl;
            return;
        }

        // 3. Embed
        NonaryEmbedder embedder;
        auto waveform = embedder.embed(content);

        // 4. Store
        // (Would connect to orchestrator/torus)
        std::cout << "[INGEST] Embedded and stored: " << file_path.filename() << std::endl;

        // 5. Archive
        std::filesystem::path archive_dir = "/var/lib/nikola/archive/";
        archive_dir /= current_date_string();
        std::filesystem::create_directories(archive_dir);
        std::filesystem::rename(file_path, archive_dir / file_path.filename());

    } catch (const std::exception& e) {
        std::cerr << "[INGEST] Error processing " << file_path << ": "
                  << e.what() << std::endl;
    }
}
```

## 16.4 Implementation

### Thread-Safe Queue

```cpp
template<typename T>
class ThreadSafeQueue {
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cv;

public:
    void push(const T& item) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(item);
        cv.notify_one();
    }

    std::optional<T> pop_with_timeout(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex);

        if (cv.wait_for(lock, timeout, [this] { return !queue.empty(); })) {
            T item = queue.front();
            queue.pop();
            return item;
        }

        return std::nullopt;
    }
};
```

---

**Cross-References:**
- See Section 9 for Nonary Embedder
- See Section 13 for Executor/KVM for PDF extraction
- See Section 9.3 for Storage in Torus
- See Section 14 for Boredom-triggered ingestion
