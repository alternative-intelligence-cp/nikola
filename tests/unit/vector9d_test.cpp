/**
 * @file vector9d_test.cpp
 * @brief Unit tests for Foundation::Vector9D
 * 
 * Tests 9D vector math, toroidal distance metrics, and periodic boundaries.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <nikola/foundation/vector9d.hpp>
#include <cmath>

using namespace nikola::foundation;
using Catch::Matchers::WithinAbs;

TEST_CASE("Vector9D construction", "[foundation][vector9d]") {
    SECTION("Default constructor (zero vector)") {
        Vector9D v;
        for (int i = 0; i < 9; ++i) {
            REQUIRE(v[i] == 0.0);
        }
    }
    
    SECTION("Initializer list constructor") {
        Vector9D v({1, 2, 3, 4, 5, 6, 7, 8, 9});
        REQUIRE(v[0] == 1.0);
        REQUIRE(v[4] == 5.0);
        REQUIRE(v[8] == 9.0);
    }
    
    SECTION("Array constructor") {
        std::array<double, 9> data = {9, 8, 7, 6, 5, 4, 3, 2, 1};
        Vector9D v(data);
        REQUIRE(v[0] == 9.0);
        REQUIRE(v[8] == 1.0);
    }
}

TEST_CASE("Vector9D arithmetic operations", "[foundation][vector9d]") {
    Vector9D a({1, 2, 3, 4, 5, 6, 7, 8, 9});
    Vector9D b({9, 8, 7, 6, 5, 4, 3, 2, 1});
    
    SECTION("Vector addition") {
        Vector9D c = a + b;
        for (int i = 0; i < 9; ++i) {
            REQUIRE(c[i] == 10.0);
        }
    }
    
    SECTION("Vector subtraction") {
        Vector9D c = a - b;
        REQUIRE(c[0] == -8.0);
        REQUIRE(c[4] == 0.0);
        REQUIRE(c[8] == 8.0);
    }
    
    SECTION("Scalar multiplication") {
        Vector9D c = a * 2.0;
        REQUIRE(c[0] == 2.0);
        REQUIRE(c[8] == 18.0);
    }
    
    SECTION("Scalar division") {
        Vector9D c = a / 2.0;
        REQUIRE_THAT(c[0], WithinAbs(0.5, 1e-10));
        REQUIRE_THAT(c[8], WithinAbs(4.5, 1e-10));
    }
    
    SECTION("In-place addition") {
        Vector9D c = a;
        c += b;
        for (int i = 0; i < 9; ++i) {
            REQUIRE(c[i] == 10.0);
        }
    }
    
    SECTION("In-place subtraction") {
        Vector9D c = a;
        c -= b;
        REQUIRE(c[0] == -8.0);
        REQUIRE(c[8] == 8.0);
    }
}

TEST_CASE("Vector9D norm and normalization", "[foundation][vector9d]") {
    SECTION("Norm of unit vector") {
        Vector9D v({1, 0, 0, 0, 0, 0, 0, 0, 0});
        REQUIRE_THAT(v.norm(), WithinAbs(1.0, 1e-10));
    }
    
    SECTION("Norm of (1,1,1,1,1,1,1,1,1)") {
        Vector9D v({1, 1, 1, 1, 1, 1, 1, 1, 1});
        REQUIRE_THAT(v.norm(), WithinAbs(3.0, 1e-10));  // sqrt(9) = 3
    }
    
    SECTION("Normalization") {
        Vector9D v({3, 4, 0, 0, 0, 0, 0, 0, 0});
        Vector9D n = v.normalized();
        
        REQUIRE_THAT(n.norm(), WithinAbs(1.0, 1e-10));
        REQUIRE_THAT(n[0], WithinAbs(0.6, 1e-10));  // 3/5
        REQUIRE_THAT(n[1], WithinAbs(0.8, 1e-10));  // 4/5
    }
    
    SECTION("Normalization of 9D vector") {
        Vector9D v({1, 2, 3, 4, 5, 6, 7, 8, 9});
        Vector9D n = v.normalized();
        
        double expected_norm = std::sqrt(1+4+9+16+25+36+49+64+81);  // 285
        REQUIRE_THAT(n.norm(), WithinAbs(1.0, 1e-10));
        REQUIRE_THAT(n[0], WithinAbs(1.0 / std::sqrt(285), 1e-10));
    }
}

TEST_CASE("Vector9D dot product", "[foundation][vector9d]") {
    SECTION("Orthogonal vectors") {
        Vector9D v1({1, 0, 0, 0, 0, 0, 0, 0, 0});
        Vector9D v2({0, 1, 0, 0, 0, 0, 0, 0, 0});
        
        REQUIRE(v1.dot(v2) == 0.0);
    }
    
    SECTION("Parallel vectors") {
        Vector9D v1({1, 2, 3, 4, 5, 6, 7, 8, 9});
        Vector9D v2({2, 4, 6, 8, 10, 12, 14, 16, 18});
        
        double expected = 2*(1+4+9+16+25+36+49+64+81);  // 2*285 = 570
        REQUIRE(v1.dot(v2) == expected);
    }
    
    SECTION("General dot product") {
        Vector9D v1({1, 0, -1, 2, 0, 3, 0, 0, 1});
        Vector9D v2({2, 1, 0, -1, 3, 1, 0, 0, 2});
        
        // 1*2 + 0*1 + (-1)*0 + 2*(-1) + 0*3 + 3*1 + 0*0 + 0*0 + 1*2
        // = 2 + 0 + 0 - 2 + 0 + 3 + 0 + 0 + 2 = 5
        REQUIRE(v1.dot(v2) == 5.0);
    }
}

TEST_CASE("Vector9D toroidal distance", "[foundation][vector9d]") {
    SECTION("Same point (zero distance)") {
        Vector9D v({0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5});
        REQUIRE_THAT(toroidal_distance(v, v), WithinAbs(0.0, 1e-10));
    }
    
    SECTION("Direct distance (no wrapping)") {
        Vector9D a({0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2});
        Vector9D b({0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3});
        
        // Distance = sqrt(9 * 0.1^2) = sqrt(0.09) = 0.3
        REQUIRE_THAT(toroidal_distance(a, b), WithinAbs(0.3, 1e-10));
    }
    
    SECTION("Wrapped distance (shorter through boundary)") {
        Vector9D a({0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        Vector9D b({0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        
        // Direct distance: 0.8
        // Wrapped distance: 0.2 (0.1 to 0.0, then 1.0 to 0.9)
        // Should choose wrapped (0.2)
        REQUIRE_THAT(toroidal_distance(a, b), WithinAbs(0.2, 1e-10));
    }
    
    SECTION("Toroidal distance in all dimensions") {
        Vector9D a({0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5});
        Vector9D b({0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5});
        
        // Wrapped distances:
        // dim 0: 0.2, dim 1: 0.2, dim 2: 0.4, dim 3: 0.4
        // dim 4: 0.4, dim 5: 0.4, dim 6: 0.2, dim 7: 0.2, dim 8: 0.0
        // Distance = sqrt(0.04 + 0.04 + 0.16 + 0.16 + 0.16 + 0.16 + 0.04 + 0.04 + 0) = sqrt(0.8)
        double expected = std::sqrt(0.8);
        REQUIRE_THAT(toroidal_distance(a, b), WithinAbs(expected, 1e-10));
    }
    
    SECTION("Maximum distance (opposite corners)") {
        Vector9D a({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        Vector9D b({0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5});
        
        // Maximum toroidal distance in each dimension is 0.5
        // Distance = sqrt(9 * 0.25) = sqrt(2.25) = 1.5
        REQUIRE_THAT(toroidal_distance(a, b), WithinAbs(1.5, 1e-10));
    }
}

TEST_CASE("Vector9D element access", "[foundation][vector9d]") {
    Vector9D v({1, 2, 3, 4, 5, 6, 7, 8, 9});
    
    SECTION("Read access") {
        REQUIRE(v[0] == 1.0);
        REQUIRE(v[4] == 5.0);
        REQUIRE(v[8] == 9.0);
    }
    
    SECTION("Write access") {
        v[0] = 10.0;
        v[8] = 90.0;
        
        REQUIRE(v[0] == 10.0);
        REQUIRE(v[8] == 90.0);
    }
}

TEST_CASE("Vector9D constexpr operations", "[foundation][vector9d]") {
    // Test that constexpr functions compile in constexpr contexts
    constexpr Vector9D v;
    static_assert(v[0] == 0.0, "Constexpr default constructor");
    
    constexpr Vector9D v2({1, 2, 3, 4, 5, 6, 7, 8, 9});
    static_assert(v2[0] == 1.0, "Constexpr initializer list");
    static_assert(v2[8] == 9.0, "Constexpr element access");
}

TEST_CASE("Vector9D edge cases", "[foundation][vector9d]") {
    SECTION("Very small values") {
        Vector9D v({1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10});
        REQUIRE_THAT(v.norm(), WithinAbs(3e-10, 1e-15));
    }
    
    SECTION("Very large values") {
        Vector9D v({1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10});
        REQUIRE_THAT(v.norm(), WithinAbs(3e10, 1e5));
    }
    
    SECTION("Mixed positive/negative") {
        Vector9D v({1, -1, 1, -1, 1, -1, 1, -1, 1});
        REQUIRE_THAT(v.norm(), WithinAbs(3.0, 1e-10));
        
        Vector9D sum = v + Vector9D({1, 1, 1, 1, 1, 1, 1, 1, 1});
        REQUIRE(sum[0] == 2.0);
        REQUIRE(sum[1] == 0.0);
    }
}
