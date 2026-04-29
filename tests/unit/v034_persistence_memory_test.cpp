#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <nikola/memory/resonance_inverted_index.hpp>
#include <nikola/persistence/log_euclidean_neurogenesis.hpp>

using Catch::Approx;
using namespace nikola::memory;
using namespace nikola::persistence;

namespace {

ResonanceSignature sig(std::initializer_list<float> xs) {
    ResonanceSignature out{};
    std::size_t i = 0;
    for (float v : xs) {
        if (i >= out.size()) break;
        out[i++] = v;
    }
    return out;
}

NeuroMetricTensor diag_metric(double d0,
                              double d1,
                              double d2,
                              double d3,
                              double d4,
                              double d5,
                              double d6,
                              double d7,
                              double d8) {
    NeuroMetricTensor g = NeuroMetricTensor::Zero();
    g(0,0)=d0; g(1,1)=d1; g(2,2)=d2; g(3,3)=d3; g(4,4)=d4;
    g(5,5)=d5; g(6,6)=d6; g(7,7)=d7; g(8,8)=d8;
    return g;
}

} // namespace

TEST_CASE("v0.3.4 §1 RII starts empty", "[v034][rii]") {
    ResonanceInvertedIndex rii;
    CHECK(rii.empty());
    CHECK(rii.size() == 0);
}

TEST_CASE("v0.3.4 §2 RII upsert inserts records", "[v034][rii]") {
    ResonanceInvertedIndex rii;
    rii.upsert(7, sig({1,0,0,0,0,0,0,0,0}), 0.4f, 10);
    rii.upsert(8, sig({0,1,0,0,0,0,0,0,0}), 0.5f, 11);

    CHECK(rii.size() == 2);
    CHECK(rii.contains(7));
    CHECK(rii.contains(8));
}

TEST_CASE("v0.3.4 §3 RII upsert on same location updates not duplicates", "[v034][rii]") {
    ResonanceInvertedIndex rii;
    rii.upsert(42, sig({1,0,0,0,0,0,0,0,0}), 0.1f, 1);
    rii.upsert(42, sig({0,1,0,0,0,0,0,0,0}), 0.9f, 2);

    CHECK(rii.size() == 1);
    ResonanceRecord rec{};
    REQUIRE(rii.try_get(42, rec));
    CHECK(rec.resonance == Approx(0.9f));
    CHECK(rec.tick == 2);
}

TEST_CASE("v0.3.4 §4 RII query returns best aligned location", "[v034][rii]") {
    ResonanceInvertedIndex rii;
    rii.upsert(1, sig({1,0,0,0,0,0,0,0,0}), 0.5f, 1);
    rii.upsert(2, sig({0,1,0,0,0,0,0,0,0}), 0.5f, 1);

    auto hits = rii.query(sig({1,0,0,0,0,0,0,0,0}), 2);
    REQUIRE(hits.size() >= 1);
    CHECK(hits[0].location == 1);
    CHECK(hits[0].cosine == Approx(1.0f).margin(1e-6f));
}

TEST_CASE("v0.3.4 §5 RII min_cosine filters weak matches", "[v034][rii]") {
    ResonanceInvertedIndex rii;
    rii.upsert(1, sig({1,0,0,0,0,0,0,0,0}), 0.4f, 1);
    rii.upsert(2, sig({0,1,0,0,0,0,0,0,0}), 0.4f, 1);

    auto hits = rii.query(sig({1,0,0,0,0,0,0,0,0}), 10, 0.8f);
    REQUIRE(hits.size() == 1);
    CHECK(hits[0].location == 1);
}

TEST_CASE("v0.3.4 §6 RII includes Hamming-neighbor buckets", "[v034][rii]") {
    ResonanceInvertedIndex rii;
    rii.upsert(100, sig({1,1,1,1,1,1,1,1,1}), 0.3f, 3);

    // Flip sign of one component -> bucket differs by one bit.
    auto hits = rii.query(sig({-1,1,1,1,1,1,1,1,1}), 5, -1.0f);
    REQUIRE_FALSE(hits.empty());
    CHECK(hits[0].location == 100);
}

TEST_CASE("v0.3.4 §7 RII top_k limits output", "[v034][rii]") {
    ResonanceInvertedIndex rii;
    for (uint64_t i = 0; i < 10; ++i) {
        rii.upsert(i, sig({1,0,0,0,0,0,0,0,0}), 0.2f + 0.01f * static_cast<float>(i), i);
    }

    auto hits = rii.query(sig({1,0,0,0,0,0,0,0,0}), 3);
    CHECK(hits.size() == 3);
}

TEST_CASE("v0.3.4 §8 RII normalize handles zero vector", "[v034][rii]") {
    ResonanceSignature z{};
    auto n = ResonanceInvertedIndex::normalize(z);
    CHECK(n[0] == Approx(1.0f));
    for (std::size_t i = 1; i < n.size(); ++i) CHECK(n[i] == Approx(0.0f));
}

TEST_CASE("v0.3.4 §9 RII clear removes all records", "[v034][rii]") {
    ResonanceInvertedIndex rii;
    rii.upsert(1, sig({1,0,0,0,0,0,0,0,0}), 0.1f, 1);
    rii.clear();
    CHECK(rii.empty());
    CHECK(rii.size() == 0);
}

TEST_CASE("v0.3.4 §10 RII rejects invalid inputs", "[v034][rii]") {
    ResonanceInvertedIndex rii;
    CHECK_THROWS_AS(rii.upsert(1, sig({1,0,0,0,0,0,0,0,0}), -0.1f, 1), std::invalid_argument);
    CHECK_THROWS_AS(rii.query(sig({1,0,0,0,0,0,0,0,0}), 5, 1.1f), std::invalid_argument);
}

TEST_CASE("v0.3.4 §11 M6 SPD check works", "[v034][m6]") {
    const NeuroMetricTensor g = diag_metric(1,2,3,4,5,6,7,8,9);
    CHECK(is_spd_metric(g));
}

TEST_CASE("v0.3.4 §12 project_metric_to_spd repairs non-SPD", "[v034][m6]") {
    NeuroMetricTensor g = NeuroMetricTensor::Identity();
    g(0,0) = -0.2;

    const auto fixed = project_metric_to_spd(g);
    CHECK(is_spd_metric(fixed));
}

TEST_CASE("v0.3.4 §13 matrix_log_spd of identity is zero", "[v034][m6]") {
    const NeuroMetricTensor I = NeuroMetricTensor::Identity();
    const auto L = matrix_log_spd(I);
    CHECK(L.norm() == Approx(0.0).margin(1e-10));
}

TEST_CASE("v0.3.4 §14 matrix_exp_sym inverts matrix_log_spd", "[v034][m6]") {
    const auto g = diag_metric(1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9);
    const auto l = matrix_log_spd(g);
    const auto e = matrix_exp_sym(l);

    CHECK((e - g).norm() == Approx(0.0).margin(1e-8));
}

TEST_CASE("v0.3.4 §15 interpolation alpha=0 and alpha=1 hit endpoints", "[v034][m6]") {
    const auto a = diag_metric(1,2,3,4,5,6,7,8,9);
    const auto b = diag_metric(2,3,4,5,6,7,8,9,10);

    const auto g0 = log_euclidean_interpolate(a, b, 0.0);
    const auto g1 = log_euclidean_interpolate(a, b, 1.0);

    CHECK((g0 - a).norm() == Approx(0.0).margin(1e-8));
    CHECK((g1 - b).norm() == Approx(0.0).margin(1e-8));
}

TEST_CASE("v0.3.4 §16 midpoint interpolation stays SPD", "[v034][m6]") {
    const auto a = diag_metric(1,2,3,4,5,6,7,8,9);
    const auto b = diag_metric(3,4,5,6,7,8,9,10,11);

    const auto g = log_euclidean_interpolate(a, b, 0.5);
    CHECK(is_spd_metric(g));
}

TEST_CASE("v0.3.4 §17 midpoint geodesic is approximately centered", "[v034][m6]") {
    const auto a = diag_metric(1,2,3,4,5,6,7,8,9);
    const auto b = diag_metric(4,5,6,7,8,9,10,11,12);

    const auto mid = log_euclidean_interpolate(a, b, 0.5);
    const double da = log_euclidean_distance(a, mid);
    const double db = log_euclidean_distance(mid, b);

    CHECK(da == Approx(db).epsilon(1e-8));
}

TEST_CASE("v0.3.4 §18 interpolation rejects invalid alpha", "[v034][m6]") {
    const auto a = NeuroMetricTensor::Identity();
    const auto b = 2.0 * NeuroMetricTensor::Identity();

    CHECK_THROWS_AS(log_euclidean_interpolate(a, b, -0.1), std::invalid_argument);
    CHECK_THROWS_AS(log_euclidean_interpolate(a, b, 1.1), std::invalid_argument);
}
