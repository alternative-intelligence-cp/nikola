#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace nikola::economy {

/**
 * @brief Service offering in the neural marketplace
 */
struct ServiceListing {
    std::string service_id;
    std::string provider_address;
    std::string description;
    uint64_t price_wei;          // Price in Wei (10^-18 ETH)
    double quality_score;        // 0.0-1.0
    int execution_count;
};

/**
 * @brief Neural Marketplace for autonomous commerce
 *
 * Allows instances to:
 * - List services they can provide (computation, analysis, etc.)
 * - Browse services from other instances
 * - Purchase services using state channels (off-chain)
 * - Settle payments on Polygon CDK (on-chain)
 *
 * Example services:
 * - "Summarize PDF" (Firecrawl specialist)
 * - "Solve physics equation" (Physics engine specialist)
 * - "Image analysis" (Visual processing specialist)
 *
 * @status STUB - Implementation deferred to Phase 4
 */
class NeuralMarketplace {
public:
    /**
     * @brief List a service this instance can provide
     * @param service Service details
     */
    void list_service(const ServiceListing& service);

    /**
     * @brief Remove service from marketplace
     * @param service_id Service to remove
     */
    void delist_service(const std::string& service_id);

    /**
     * @brief Browse available services
     * @param query Search query
     * @return Matching services
     */
    std::vector<ServiceListing> browse_services(const std::string& query);

    /**
     * @brief Purchase service from another instance
     * @param service_id Service to purchase
     * @param payment_wei Amount in Wei
     * @return Transaction hash
     */
    std::string purchase_service(const std::string& service_id, uint64_t payment_wei);

    /**
     * @brief Execute purchased service
     * @param transaction_hash Purchase transaction
     * @param input_data Service input
     * @return Service output
     */
    std::string execute_service(const std::string& transaction_hash,
                                const std::string& input_data);

private:
    std::vector<ServiceListing> my_services_;
};

} // namespace nikola::economy
