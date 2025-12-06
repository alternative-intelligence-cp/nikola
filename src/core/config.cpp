#include "nikola/core/config.hpp"

#include <fstream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <cctype>

namespace nikola::core {

// Static member initialization
Config* Config::instance_ = nullptr;
std::mutex Config::instance_mutex_;

Config::Config() {
    load_defaults();
    load_from_file("/etc/nikola/nikola.conf");
    load_from_environment();
}

Config& Config::get() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (!instance_) {
        instance_ = new Config();
    }
    return *instance_;
}

std::string Config::get_value(const std::string& key) const {
    std::lock_guard<std::mutex> lock(config_mutex_);

    auto it = config_map_.find(key);
    if (it != config_map_.end()) {
        return it->second;
    }
    return "";
}

// ============================================================================
// PATH ACCESSORS
// ============================================================================

std::string Config::key_directory() const {
    return get_value("key_directory");
}

std::string Config::gold_checkpoint_dir() const {
    return get_value("gold_checkpoint_dir");
}

std::string Config::runtime_directory() const {
    return get_value("runtime_directory");
}

std::string Config::work_directory() const {
    return get_value("work_directory");
}

std::string Config::ingest_directory() const {
    return get_value("ingest_directory");
}

std::string Config::archive_directory() const {
    return get_value("archive_directory");
}

std::string Config::lsm_data_directory() const {
    return get_value("lsm_data_directory");
}

std::string Config::dream_directory() const {
    return get_value("dream_directory");
}

std::string Config::identity_directory() const {
    return get_value("identity_directory");
}

// ============================================================================
// TESTING & OVERRIDE SUPPORT
// ============================================================================

void Config::set(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    config_map_[key] = value;
}

void Config::reset_to_defaults() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    config_map_.clear();
    load_defaults();
}

bool Config::has_key(const std::string& key) const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_map_.find(key) != config_map_.end();
}

std::map<std::string, std::string> Config::get_all() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_map_;
}

// ============================================================================
// PRIVATE METHODS
// ============================================================================

void Config::load_defaults() {
    // Security: Use /run (tmpfs) for sockets, not /tmp (Finding 4.1)
    config_map_["key_directory"] = "/etc/nikola/keys";
    config_map_["gold_checkpoint_dir"] = "/var/lib/nikola/gold";
    config_map_["runtime_directory"] = "/run/nikola";
    config_map_["work_directory"] = "/var/lib/nikola/work";
    config_map_["ingest_directory"] = "/var/lib/nikola/ingest";
    config_map_["archive_directory"] = "/var/lib/nikola/archive";
    config_map_["lsm_data_directory"] = "/var/lib/nikola/lsm";
    config_map_["dream_directory"] = "/var/lib/nikola/dreams";
    config_map_["identity_directory"] = "/var/lib/nikola/identity";
}

void Config::load_from_file(const std::string& config_file_path) {
    std::ifstream file(config_file_path);
    if (!file.is_open()) {
        // Config file is optional
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Parse KEY=VALUE format
        size_t equals_pos = line.find('=');
        if (equals_pos == std::string::npos) {
            continue;  // Skip malformed lines
        }

        std::string key = line.substr(0, equals_pos);
        std::string value = line.substr(equals_pos + 1);

        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        if (!key.empty()) {
            config_map_[key] = value;
        }
    }
}

void Config::load_from_environment() {
    // Map of environment variable names to config keys
    const std::map<std::string, std::string> env_map = {
        {"NIKOLA_KEY_DIRECTORY", "key_directory"},
        {"NIKOLA_GOLD_CHECKPOINT_DIR", "gold_checkpoint_dir"},
        {"NIKOLA_RUNTIME_DIRECTORY", "runtime_directory"},
        {"NIKOLA_WORK_DIRECTORY", "work_directory"},
        {"NIKOLA_INGEST_DIRECTORY", "ingest_directory"},
        {"NIKOLA_ARCHIVE_DIRECTORY", "archive_directory"},
        {"NIKOLA_LSM_DATA_DIRECTORY", "lsm_data_directory"},
        {"NIKOLA_DREAM_DIRECTORY", "dream_directory"},
        {"NIKOLA_IDENTITY_DIRECTORY", "identity_directory"}
    };

    for (const auto& [env_name, config_key] : env_map) {
        const char* env_value = std::getenv(env_name.c_str());
        if (env_value != nullptr && env_value[0] != '\0') {
            config_map_[config_key] = env_value;
        }
    }
}

std::string Config::env_to_key(const std::string& env_name) const {
    // Convert NIKOLA_KEY_DIRECTORY -> key_directory
    if (env_name.substr(0, 7) != "NIKOLA_") {
        return "";
    }

    std::string key = env_name.substr(7);  // Remove "NIKOLA_" prefix

    // Convert to lowercase
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    return key;
}

} // namespace nikola::core
