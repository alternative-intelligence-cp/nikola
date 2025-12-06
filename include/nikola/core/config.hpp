#pragma once

#include <string>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>

namespace nikola::core {

/**
 * @brief Centralized configuration management for Nikola
 *
 * Replaces hardcoded file paths throughout the codebase with configurable
 * values that can be overridden via:
 * 1. Programmatic setter (highest priority - for testing/dependency injection)
 * 2. Environment variables (NIKOLA_*)
 * 3. Configuration file (/etc/nikola/nikola.conf)
 * 4. Compiled defaults (lowest priority)
 *
 * Thread-safe singleton pattern.
 *
 * @see NIKOLA_AUDIT_IMPLEMENTATION_PLAN.md - Finding 2.1
 */
class Config {
private:
    // Singleton instance
    static Config* instance_;
    static std::mutex instance_mutex_;

    // Configuration values (mutable for testing/override)
    std::map<std::string, std::string> config_map_;
    mutable std::mutex config_mutex_;

    // Private constructor for singleton
    Config();

public:
    /**
     * @brief Get singleton instance (thread-safe)
     * @return Reference to Config singleton
     */
    static Config& get();

    // Prevent copies and moves
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;
    Config(Config&&) = delete;
    Config& operator=(Config&&) = delete;

    /**
     * @brief Get configuration value by key
     * @param key Configuration key
     * @return Configuration value, or empty string if not found
     */
    std::string get_value(const std::string& key) const;

    // ========================================================================
    // PATH ACCESSORS
    // ========================================================================

    /**
     * @brief ZMQ Curve25519 key storage directory
     * @default /etc/nikola/keys
     * @env NIKOLA_KEY_DIRECTORY
     */
    std::string key_directory() const;

    /**
     * @brief Gold LLM checkpoint storage directory
     * @default /var/lib/nikola/gold
     * @env NIKOLA_GOLD_CHECKPOINT_DIR
     */
    std::string gold_checkpoint_dir() const;

    /**
     * @brief Runtime directory for IPC sockets (tmpfs)
     * @default /run/nikola
     * @env NIKOLA_RUNTIME_DIRECTORY
     * @note Changed from /tmp to /run for security (Finding 4.1)
     */
    std::string runtime_directory() const;

    /**
     * @brief Working directory for VM overlays and temporary processing
     * @default /var/lib/nikola/work
     * @env NIKOLA_WORK_DIRECTORY
     */
    std::string work_directory() const;

    /**
     * @brief Ingestion directory for incoming data
     * @default /var/lib/nikola/ingest
     * @env NIKOLA_INGEST_DIRECTORY
     */
    std::string ingest_directory() const;

    /**
     * @brief Archive directory for processed data
     * @default /var/lib/nikola/archive
     * @env NIKOLA_ARCHIVE_DIRECTORY
     */
    std::string archive_directory() const;

    /**
     * @brief LSM tree data directory for DMC persistence
     * @default /var/lib/nikola/lsm
     * @env NIKOLA_LSM_DATA_DIRECTORY
     */
    std::string lsm_data_directory() const;

    /**
     * @brief Dream state checkpoint directory
     * @default /var/lib/nikola/dreams
     * @env NIKOLA_DREAM_DIRECTORY
     */
    std::string dream_directory() const;

    /**
     * @brief Identity/personality checkpoint directory
     * @default /var/lib/nikola/identity
     * @env NIKOLA_IDENTITY_DIRECTORY
     */
    std::string identity_directory() const;

    // ========================================================================
    // TESTING & OVERRIDE SUPPORT
    // ========================================================================

    /**
     * @brief Set configuration value programmatically (highest priority)
     * @param key Configuration key
     * @param value New value
     * @note Thread-safe. Used for testing and dependency injection.
     */
    void set(const std::string& key, const std::string& value);

    /**
     * @brief Reset all configuration to compiled defaults
     * @note Thread-safe. Used for testing.
     */
    void reset_to_defaults();

    /**
     * @brief Check if a configuration key exists
     * @param key Configuration key
     * @return true if key exists in configuration
     */
    bool has_key(const std::string& key) const;

    /**
     * @brief Get all configuration keys and values
     * @return Map of all configuration
     * @note Thread-safe. Used for debugging/logging.
     */
    std::map<std::string, std::string> get_all() const;

private:
    /**
     * @brief Load compiled default values
     */
    void load_defaults();

    /**
     * @brief Load configuration from file (optional)
     * @param config_file_path Path to configuration file
     *
     * Format: KEY=VALUE (one per line, # for comments)
     */
    void load_from_file(const std::string& config_file_path);

    /**
     * @brief Load configuration from environment variables
     *
     * Reads all NIKOLA_* environment variables and maps them to config keys
     * (e.g., NIKOLA_KEY_DIRECTORY -> key_directory)
     */
    void load_from_environment();

    /**
     * @brief Convert environment variable name to config key
     * @param env_name Environment variable name (e.g., NIKOLA_KEY_DIRECTORY)
     * @return Config key (e.g., key_directory)
     */
    std::string env_to_key(const std::string& env_name) const;
};

} // namespace nikola::core
