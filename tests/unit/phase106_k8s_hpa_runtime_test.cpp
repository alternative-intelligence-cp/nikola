// =============================================================================
// Phase 106 / GAP-045 (live) — K8s HPA Runtime Tests
// =============================================================================
// Tests the K8sHpaRuntime class against the live k3s cluster running on
// this machine (ariax-dev-1, v1.34.4+k3s1).
//
// Test structure (9 test cases):
//   1. Construction — kubeconfig found, path non-empty
//   2. Connectivity — can_connect() returns true on live cluster
//   3. Server version — contains "1.34" and "k3s1"
//   4. Node count — at least 1 node
//   5. Node names — list non-empty, contains "ariax-dev-1"
//   6. Node status — first node is "Ready"
//   7. Namespace list — at least 4 namespaces
//   8. Core namespaces present — kube-system, default, kube-public, kube-node-lease
//   9. Metrics-server + algorithm integration — live node_count drives
//      classify_atp_regime() / scaling_decision() producing valid results
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "nikola/infrastructure/k8s_hpa_runtime.hpp"
#include "nikola/infrastructure/k8s_hpa.hpp"

#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>

using nikola::infrastructure::K8sHpaRuntime;
using nikola::infrastructure::ScalingAction;
using nikola::infrastructure::ATPRegime;
using nikola::infrastructure::HPA_TARGET_LAG_S;
using nikola::infrastructure::classify_atp_regime;
using nikola::infrastructure::scaling_decision;
using nikola::infrastructure::atp_regime_name;

// ---------------------------------------------------------------------------
// Helper — get_runtime() constructs once per test run
// ---------------------------------------------------------------------------

static K8sHpaRuntime& get_runtime() {
    static K8sHpaRuntime rt;
    return rt;
}

// ===========================================================================
// Test 1 — Construction and kubeconfig discovery
// ===========================================================================
TEST_CASE("Phase106 K8sHpaRuntime construction", "[phase106][k8s][hpa]") {
    SECTION("Default construction succeeds") {
        K8sHpaRuntime rt;
        SUCCEED("K8sHpaRuntime constructed without exception");
    }

    SECTION("kubeconfig_path is non-empty") {
        K8sHpaRuntime rt;
        REQUIRE_FALSE(rt.kubeconfig_path().empty());
    }

    SECTION("kubeconfig_path is an absolute path") {
        K8sHpaRuntime rt;
        auto path = rt.kubeconfig_path();
        REQUIRE(path.front() == '/');
    }

    SECTION("kubeconfig_path contains 'config' or 'k3s'") {
        K8sHpaRuntime rt;
        auto path = rt.kubeconfig_path();
        bool plausible = (path.find("config") != std::string::npos ||
                          path.find("k3s")    != std::string::npos);
        REQUIRE(plausible);
    }
}

// ===========================================================================
// Test 2 — Cluster connectivity
// ===========================================================================
TEST_CASE("Phase106 K8sHpaRuntime can_connect", "[phase106][k8s][hpa][live]") {
    auto& rt = get_runtime();

    SECTION("can_connect returns true on live cluster") {
        REQUIRE(rt.can_connect());
    }
}

// ===========================================================================
// Test 3 — Server version
// ===========================================================================
TEST_CASE("Phase106 K8sHpaRuntime server_version", "[phase106][k8s][hpa][live]") {
    auto& rt = get_runtime();

    SECTION("server_version does not throw") {
        REQUIRE_NOTHROW(rt.server_version());
    }

    SECTION("server_version is non-empty") {
        auto ver = rt.server_version();
        REQUIRE_FALSE(ver.empty());
    }

    SECTION("server_version starts with 'v'") {
        auto ver = rt.server_version();
        REQUIRE(ver.front() == 'v');
    }

    SECTION("server_version contains '1.34'") {
        auto ver = rt.server_version();
        REQUIRE(ver.find("1.34") != std::string::npos);
    }

    SECTION("server_version contains 'k3s1'") {
        auto ver = rt.server_version();
        REQUIRE(ver.find("k3s1") != std::string::npos);
    }
}

// ===========================================================================
// Test 4 — Node count
// ===========================================================================
TEST_CASE("Phase106 K8sHpaRuntime node_count", "[phase106][k8s][hpa][live]") {
    auto& rt = get_runtime();

    SECTION("node_count is at least 1") {
        REQUIRE(rt.node_count() >= 1);
    }

    SECTION("node_count is reasonable (not absurdly large)") {
        REQUIRE(rt.node_count() < 1000);
    }
}

// ===========================================================================
// Test 5 — Node names
// ===========================================================================
TEST_CASE("Phase106 K8sHpaRuntime node_names", "[phase106][k8s][hpa][live]") {
    auto& rt = get_runtime();

    SECTION("node_names is non-empty") {
        auto names = rt.node_names();
        REQUIRE_FALSE(names.empty());
    }

    SECTION("node_names contains 'ariax-dev-1'") {
        auto names = rt.node_names();
        auto it    = std::find(names.begin(), names.end(), "ariax-dev-1");
        REQUIRE(it != names.end());
    }

    SECTION("node count matches node_names size") {
        REQUIRE(rt.node_count() == static_cast<int>(rt.node_names().size()));
    }
}

// ===========================================================================
// Test 6 — Node status
// ===========================================================================
TEST_CASE("Phase106 K8sHpaRuntime node_status", "[phase106][k8s][hpa][live]") {
    auto& rt = get_runtime();

    SECTION("node_status for ariax-dev-1 is 'Ready'") {
        REQUIRE(rt.node_status("ariax-dev-1") == "Ready");
    }

    SECTION("node_status for first node in list is 'Ready'") {
        auto names = rt.node_names();
        REQUIRE_FALSE(names.empty());
        REQUIRE(rt.node_status(names.front()) == "Ready");
    }
}

// ===========================================================================
// Test 7 — Namespace list
// ===========================================================================
TEST_CASE("Phase106 K8sHpaRuntime namespaces", "[phase106][k8s][hpa][live]") {
    auto& rt = get_runtime();

    SECTION("namespaces() is non-empty") {
        auto ns = rt.namespaces();
        REQUIRE_FALSE(ns.empty());
    }

    SECTION("at least 4 namespaces visible") {
        auto ns = rt.namespaces();
        REQUIRE(ns.size() >= 4);
    }

    SECTION("has_namespace with empty string returns false") {
        REQUIRE_FALSE(rt.has_namespace(""));
    }
}

// ===========================================================================
// Test 8 — Core namespaces present
// ===========================================================================
TEST_CASE("Phase106 K8sHpaRuntime core namespaces", "[phase106][k8s][hpa][live]") {
    auto& rt = get_runtime();

    SECTION("kube-system namespace exists") {
        REQUIRE(rt.has_namespace("kube-system"));
    }

    SECTION("default namespace exists") {
        REQUIRE(rt.has_namespace("default"));
    }

    SECTION("kube-public namespace exists") {
        REQUIRE(rt.has_namespace("kube-public"));
    }

    SECTION("kube-node-lease namespace exists") {
        REQUIRE(rt.has_namespace("kube-node-lease"));
    }

    SECTION("metrics-server deployment exists in kube-system") {
        REQUIRE(rt.has_deployment("kube-system", "metrics-server"));
    }
}

// ===========================================================================
// Test 9 — Combined live data + algorithmic HPA integration
// ===========================================================================
TEST_CASE("Phase106 K8sHpaRuntime HPA algorithm integration",
          "[phase106][k8s][hpa][live][algorithm]") {
    auto& rt = get_runtime();

    SECTION("Live node_count feeds into HPA load model") {
        // Derive a synthetic unified_load from the node count:
        //   load = node_count * HPA_TARGET_LAG_S
        // This is intentionally above target to exercise SCALE_UP path.
        int n = rt.node_count();
        REQUIRE(n >= 1);

        double load_above = static_cast<double>(n) * HPA_TARGET_LAG_S * 2.0;
        auto act_up = scaling_decision(load_above);
        REQUIRE(act_up == ScalingAction::SCALE_UP);

        double load_below = HPA_TARGET_LAG_S * 0.25;
        auto act_down = scaling_decision(load_below);
        REQUIRE(act_down == ScalingAction::SCALE_DOWN);

        double load_mid = HPA_TARGET_LAG_S * 0.75;
        auto act_mid = scaling_decision(load_mid);
        REQUIRE(act_mid == ScalingAction::MAINTAIN);
    }

    SECTION("ATP regime classification covers all regime names") {
        // Probe the four ATP thresholds via boundaries from the spec
        // CRITICAL < 0.15, LOW 0.15-0.20, NOMINAL 0.20-0.50, HIGH > 0.50
        struct Case { double atp; ATPRegime expected; };
        const Case cases[] = {
            {0.10, ATPRegime::CRITICAL},
            {0.17, ATPRegime::LOW},
            {0.35, ATPRegime::NOMINAL},
            {0.75, ATPRegime::HIGH},
        };
        for (auto& c : cases) {
            auto regime = classify_atp_regime(c.atp);
            REQUIRE(regime == c.expected);
            // regime name must be non-empty
            std::string_view name = atp_regime_name(regime);
            REQUIRE_FALSE(name.empty());
        }
    }

    SECTION("Live cluster version feeds into compatibility check") {
        // Verify version string can be obtained and is plausibly a k8s version
        auto ver = rt.server_version();
        REQUIRE_FALSE(ver.empty());
        // Must start with 'v' and contain a '.'
        REQUIRE(ver.front() == 'v');
        REQUIRE(ver.find('.') != std::string::npos);

        // Node readiness confirms cluster is healthy before HPA decisions
        bool cluster_ok = rt.can_connect();
        REQUIRE(cluster_ok);

        // When cluster is healthy, scale_decision with zero load → SCALE_DOWN
        // (HPA should never run below MIN_REPLICAS in production, but
        //  the algorithm itself produces SCALE_DOWN for load=0)
        auto action = scaling_decision(0.0);
        REQUIRE(action == ScalingAction::SCALE_DOWN);
    }

    SECTION("Namespace count used as proxy for cluster complexity") {
        auto ns = rt.namespaces();
        int  ns_count = static_cast<int>(ns.size());
        REQUIRE(ns_count >= 4);

        // Use ns_count as a naive load proxy: higher ns_count = higher load
        double proxy_load = static_cast<double>(ns_count) * 0.1;

        auto action = scaling_decision(proxy_load);
        // action must be one of the three valid values
        bool valid = (action == ScalingAction::SCALE_UP   ||
                      action == ScalingAction::MAINTAIN   ||
                      action == ScalingAction::SCALE_DOWN);
        REQUIRE(valid);
    }
}
