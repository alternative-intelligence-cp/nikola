/**
 * @file marketplace.cpp
 * @brief Phase 130 — NeuralMarketplace implementation
 */

#include "nikola/economy/marketplace.hpp"

#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <numeric>

namespace nikola::economy {

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

std::string NeuralMarketplace::make_tx_hash(const std::string& service_id,
                                              uint64_t tick) {
    // Deterministic: FNV-1a over (service_id + tick) → 16-char hex
    constexpr uint64_t FNV_PRIME  = 0x100000001b3ULL;
    constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
    uint64_t h = FNV_OFFSET;
    for (char c : service_id) {
        h ^= static_cast<uint8_t>(c);
        h *= FNV_PRIME;
    }
    // Mix tick
    for (int i = 0; i < 8; ++i) {
        h ^= static_cast<uint8_t>((tick >> (i * 8)) & 0xff);
        h *= FNV_PRIME;
    }
    std::ostringstream oss;
    oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << h;
    return oss.str();
}

bool NeuralMarketplace::keyword_match(const std::string& haystack,
                                       const std::string& needle) {
    if (needle.empty()) return true;
    auto lc = [](unsigned char c) { return std::tolower(c); };
    std::string h_lo = haystack;
    std::string n_lo = needle;
    std::transform(h_lo.begin(), h_lo.end(), h_lo.begin(), lc);
    std::transform(n_lo.begin(), n_lo.end(), n_lo.begin(), lc);
    return h_lo.find(n_lo) != std::string::npos;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void NeuralMarketplace::evict_listings_if_needed() {
    while (listings_.size() > MARKET_MAX_LISTINGS) {
        listings_.erase(listings_.begin());
    }
}

void NeuralMarketplace::evict_history_if_needed() {
    while (history_.size() > MARKET_MAX_HISTORY) {
        history_.erase(history_.begin());
    }
}

std::string NeuralMarketplace::generate_result(const ServiceListing& svc,
                                                 const std::string& input) const {
    return "result::" + svc.service_id + "::" + input.substr(0, 16);
}

// ---------------------------------------------------------------------------
// Listing management
// ---------------------------------------------------------------------------

void NeuralMarketplace::list_service(const ServiceListing& service) {
    // Dedup by service_id: if exists, update in place
    for (auto& existing : listings_) {
        if (existing.service_id == service.service_id) {
            existing = service;
            return;
        }
    }
    listings_.push_back(service);
    evict_listings_if_needed();
}

void NeuralMarketplace::delist_service(const std::string& service_id) {
    listings_.erase(
        std::remove_if(listings_.begin(), listings_.end(),
                       [&](const ServiceListing& s) {
                           return s.service_id == service_id;
                       }),
        listings_.end());
}

std::optional<ServiceListing>
NeuralMarketplace::find_service(const std::string& service_id) const {
    for (const auto& s : listings_) {
        if (s.service_id == service_id) return s;
    }
    return std::nullopt;
}

const std::vector<ServiceListing>& NeuralMarketplace::all_listings() const {
    return listings_;
}

std::vector<ServiceListing>
NeuralMarketplace::services_by_provider(const std::string& provider_address) const {
    std::vector<ServiceListing> out;
    for (const auto& s : listings_) {
        if (s.provider_address == provider_address) out.push_back(s);
    }
    return out;
}

std::vector<ServiceListing>
NeuralMarketplace::browse_services(const std::string& query) const {
    std::vector<ServiceListing> out;
    for (const auto& s : listings_) {
        if (keyword_match(s.description, query) ||
            keyword_match(s.service_id,  query))
        {
            out.push_back(s);
        }
    }
    // Sort by quality descending
    std::sort(out.begin(), out.end(),
              [](const ServiceListing& a, const ServiceListing& b) {
                  return a.quality_score > b.quality_score;
              });
    return out;
}

size_t NeuralMarketplace::listing_count() const {
    return listings_.size();
}

// ---------------------------------------------------------------------------
// Transactions
// ---------------------------------------------------------------------------

std::string NeuralMarketplace::purchase_service(const std::string& service_id,
                                                  const std::string& buyer,
                                                  uint64_t payment_wei,
                                                  uint64_t tick) {
    auto it = std::find_if(listings_.begin(), listings_.end(),
                            [&](const ServiceListing& s) {
                                return s.service_id == service_id;
                            });
    if (it == listings_.end()) return {};  // service not found

    Transaction tx;
    tx.service_id     = service_id;
    tx.buyer_address  = buyer;
    tx.payment_wei    = payment_wei;
    tx.tick           = tick;
    tx.status         = TxStatus::PENDING;
    tx.tx_hash        = make_tx_hash(service_id, tick);

    history_.push_back(tx);
    evict_history_if_needed();

    if (on_tx_cb_) on_tx_cb_(history_.back());
    return tx.tx_hash;
}

std::string NeuralMarketplace::execute_service(const std::string& tx_hash,
                                                const std::string& input_data) {
    for (auto& tx : history_) {
        if (tx.tx_hash != tx_hash) continue;

        // Find the listing
        auto it = std::find_if(listings_.begin(), listings_.end(),
                               [&](const ServiceListing& s) {
                                   return s.service_id == tx.service_id;
                               });
        if (it == listings_.end()) {
            tx.status = TxStatus::FAILED;
            return {};
        }

        tx.result  = generate_result(*it, input_data);
        tx.status  = TxStatus::EXECUTED;
        it->execution_count++;

        if (on_tx_cb_) on_tx_cb_(tx);
        return tx.result;
    }
    return {};  // tx_hash not found
}

const Transaction*
NeuralMarketplace::find_transaction(const std::string& tx_hash) const {
    for (const auto& tx : history_) {
        if (tx.tx_hash == tx_hash) return &tx;
    }
    return nullptr;
}

const std::vector<Transaction>& NeuralMarketplace::transaction_history() const {
    return history_;
}

size_t NeuralMarketplace::transaction_count() const {
    return history_.size();
}

// ---------------------------------------------------------------------------
// Quality feedback
// ---------------------------------------------------------------------------

void NeuralMarketplace::rate_service(const std::string& service_id, double score) {
    score = std::clamp(score, 0.0, 1.0);
    for (auto& s : listings_) {
        if (s.service_id == service_id) {
            // EMA with alpha=0.30
            s.quality_score = 0.70 * s.quality_score + 0.30 * score;
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

NeuralMarketplace::Stats NeuralMarketplace::stats() const {
    Stats s;
    s.listing_count     = listings_.size();
    s.transaction_count = history_.size();
    for (const auto& tx : history_) {
        s.total_volume_wei += tx.payment_wei;
        if (tx.status == TxStatus::EXECUTED) ++s.executed_count;
        if (tx.status == TxStatus::FAILED)   ++s.failed_count;
    }
    if (!listings_.empty()) {
        double q = 0.0;
        for (const auto& l : listings_) q += l.quality_score;
        s.mean_quality = q / static_cast<double>(listings_.size());
    }
    return s;
}

void NeuralMarketplace::clear() {
    listings_.clear();
    history_.clear();
}

} // namespace nikola::economy
