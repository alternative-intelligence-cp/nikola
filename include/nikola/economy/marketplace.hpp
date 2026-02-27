#pragma once
/**
 * @file marketplace.hpp
 * @brief Phase 130 — NeuralMarketplace: autonomous service commerce
 *
 * A peer-to-peer service registry where Nikola instances can list capabilities
 * (compute, analysis, summarization…), discover peers' offerings, execute mock
 * transactions, and track economic history — all without real blockchain deps.
 * Fiat blockchain integration (Polygon CDK) is deferred; this layer provides
 * the data model and logic surface for local and simulated multi-agent use.
 */

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <cstdint>
#include <optional>
#include <algorithm>
#include <cctype>

namespace nikola::economy {

/// Maximum number of listings held in memory (FIFO eviction)
inline constexpr size_t MARKET_MAX_LISTINGS   = 256;
/// Maximum transaction history entries
inline constexpr size_t MARKET_MAX_HISTORY    = 512;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Status of a marketplace transaction
enum class TxStatus : uint8_t {
    PENDING,
    EXECUTED,
    FAILED,
    REFUNDED
};

/// A service offering in the neural marketplace
struct ServiceListing {
    std::string service_id;          ///< Unique identifier (provider:name)
    std::string provider_address;    ///< Provider ID / address
    std::string description;         ///< Human-readable description
    uint64_t    price_wei   = 0;     ///< Price in Wei-equivalent units
    double      quality_score = 0.5; ///< Quality [0,1] — updated from feedback
    int         execution_count = 0; ///< Times successfully executed
};

/// A completed (or pending) marketplace transaction
struct Transaction {
    std::string tx_hash;          ///< Unique transaction hash
    std::string service_id;       ///< Service purchased
    std::string buyer_address;    ///< Purchaser ID
    uint64_t    payment_wei = 0;  ///< Amount paid
    uint64_t    tick        = 0;  ///< Tick at purchase time
    TxStatus    status      = TxStatus::PENDING;
    std::string result;           ///< Service output (set after execute)
};

// ---------------------------------------------------------------------------
// NeuralMarketplace
// ---------------------------------------------------------------------------

class NeuralMarketplace {
public:
    // -----------------------------------------------------------------------
    // Listing management
    // -----------------------------------------------------------------------

    /// Add or update a service listing (dedup by service_id)
    void list_service(const ServiceListing& service);

    /// Remove a listing by service_id; no-op if not found
    void delist_service(const std::string& service_id);

    /// Find a listing by service_id; returns nullopt if not found
    [[nodiscard]] std::optional<ServiceListing>
    find_service(const std::string& service_id) const;

    /// All current listings
    [[nodiscard]] const std::vector<ServiceListing>& all_listings() const;

    /// Listings provided by a specific address
    [[nodiscard]] std::vector<ServiceListing>
    services_by_provider(const std::string& provider_address) const;

    /// Keyword search (case-insensitive, matches description or service_id)
    [[nodiscard]] std::vector<ServiceListing>
    browse_services(const std::string& query) const;

    /// Number of active listings
    [[nodiscard]] size_t listing_count() const;

    // -----------------------------------------------------------------------
    // Transactions
    // -----------------------------------------------------------------------

    /**
     * @brief Record intent to purchase a service
     * @param service_id  Service to buy
     * @param buyer       Buyer identifier
     * @param payment_wei Amount offered
     * @param tick        Current simulation tick
     * @return Transaction hash (empty on failure — service not found)
     */
    std::string purchase_service(const std::string& service_id,
                                  const std::string& buyer,
                                  uint64_t payment_wei,
                                  uint64_t tick = 0);

    /**
     * @brief Execute a pending transaction (simulate service delivery)
     * @param tx_hash    Hash returned by purchase_service
     * @param input_data Input passed to the service
     * @return Service result string (empty if tx not found)
     */
    std::string execute_service(const std::string& tx_hash,
                                const std::string& input_data);

    /// Find a transaction by hash; returns nullptr if not found
    [[nodiscard]] const Transaction*
    find_transaction(const std::string& tx_hash) const;

    /// All transaction history (most-recent last, FIFO-capped)
    [[nodiscard]] const std::vector<Transaction>& transaction_history() const;

    /// Number of transactions recorded
    [[nodiscard]] size_t transaction_count() const;

    // -----------------------------------------------------------------------
    // Quality feedback
    // -----------------------------------------------------------------------

    /**
     * @brief Update quality score of a listing after execution
     * @param service_id  Service to rate
     * @param score       Score [0,1] from this interaction (EMA blended)
     */
    void rate_service(const std::string& service_id, double score);

    // -----------------------------------------------------------------------
    // Stats + management
    // -----------------------------------------------------------------------

    struct Stats {
        size_t   listing_count     = 0;
        size_t   transaction_count = 0;
        uint64_t total_volume_wei  = 0;
        size_t   executed_count    = 0;
        size_t   failed_count      = 0;
        double   mean_quality      = 0.0;
    };

    [[nodiscard]] Stats stats() const;

    /// Remove all listings and history
    void clear();

    // -----------------------------------------------------------------------
    // Callback
    // -----------------------------------------------------------------------

    using OnTransaction = std::function<void(const Transaction&)>;
    void on_transaction(OnTransaction cb) { on_tx_cb_ = std::move(cb); }

    // -----------------------------------------------------------------------
    // Static helpers
    // -----------------------------------------------------------------------

    /// Generate a deterministic mock tx hash from service_id + tick
    static std::string make_tx_hash(const std::string& service_id, uint64_t tick);

    /// Case-insensitive substring containment check
    static bool keyword_match(const std::string& haystack,
                               const std::string& needle);

private:
    std::vector<ServiceListing> listings_;
    std::vector<Transaction>    history_;
    OnTransaction               on_tx_cb_;

    void     evict_listings_if_needed();
    void     evict_history_if_needed();
    std::string generate_result(const ServiceListing& svc,
                                 const std::string& input) const;
};

} // namespace nikola::economy
