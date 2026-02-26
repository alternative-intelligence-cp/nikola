// =============================================================================
// NIKOLA — Phase 106 / GAP-045 (live)
// K8s HPA Runtime — live kubectl interface for k3s Kubernetes cluster
// =============================================================================
// Spec   : GAP-045 (live)  HPA Runtime Integration
// Section: §12  K8s/HPA
// Author : Nikola Phase 106
// License: MIT
// =============================================================================
//
// Wraps popen()/pclose() calls to `kubectl` with an explicit KUBECONFIG,
// exposing cluster metadata needed to drive the algorithmic HPA logic in
// k8s_hpa.hpp.
//
// Usage:
//   nikola::infrastructure::K8sHpaRuntime rt;
//   if (rt.can_connect()) {
//       auto ver = rt.server_version();         // "v1.34.4+k3s1"
//       auto ns  = rt.namespaces();             // {"default", "kube-system", ...}
//       int n    = rt.node_count();             // 1
//   }
// =============================================================================

#pragma once

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

namespace nikola::infrastructure {

// ---------------------------------------------------------------------------
// K8sHpaRuntime
// ---------------------------------------------------------------------------

/// Live Kubernetes cluster interface driven by popen()+kubectl.
///
/// Constructor locates a usable kubeconfig (tries $HOME/.kube/config, then
/// /etc/rancher/k3s/k3s.yaml, then $KUBECONFIG env var) and stores the path
/// for subsequent queries.   All kubectl invocations set KUBECONFIG explicitly
/// so the object works even when $KUBECONFIG is not set in the environment.
class K8sHpaRuntime {
public:
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Locate kubeconfig and prepare for queries.
    /// Does NOT proactively call the cluster; call can_connect() to verify.
    explicit K8sHpaRuntime()
        : kubeconfig_(find_kubeconfig())
    {}

    // -----------------------------------------------------------------------
    // Cluster probe
    // -----------------------------------------------------------------------

    /// Returns true when the cluster API server responds (i.e. `kubectl get
    /// nodes` exits cleanly and at least one node appears in the output).
    [[nodiscard]] bool can_connect() const noexcept {
        try {
            auto out = run_kubectl("get nodes --no-headers 2>&1");
            return !out.empty() && out.find("Ready") != std::string::npos;
        } catch (...) {
            return false;
        }
    }

    // -----------------------------------------------------------------------
    // Cluster metadata
    // -----------------------------------------------------------------------

    /// Returns the server version string as reported by `kubectl version`.
    /// E.g. "v1.34.4+k3s1".   Throws std::runtime_error on failure.
    [[nodiscard]] std::string server_version() const {
        auto out = run_kubectl("version 2>&1");
        // Look for "Server Version: v..." line
        std::istringstream ss(out);
        std::string line;
        while (std::getline(ss, line)) {
            auto pos = line.find("Server Version: v");
            if (pos != std::string::npos) {
                // substr from the 'v' that starts the version token
                auto v = pos + std::string("Server Version: ").size();
                std::istringstream ls(line.substr(v));
                std::string tok;
                ls >> tok;
                return tok;
            }
        }
        throw std::runtime_error("K8sHpaRuntime: could not parse server version from: " + out);
    }

    /// Returns the number of nodes registered in the cluster.
    [[nodiscard]] int node_count() const {
        auto out = run_kubectl("get nodes --no-headers 2>&1");
        return static_cast<int>(parse_lines(out).size());
    }

    /// Returns a list of node names.
    [[nodiscard]] std::vector<std::string> node_names() const {
        auto out  = run_kubectl("get nodes --no-headers 2>&1");
        auto rows = parse_lines(out);
        std::vector<std::string> names;
        names.reserve(rows.size());
        for (auto& row : rows) {
            std::istringstream ss(row);
            std::string name;
            if (ss >> name) names.push_back(std::move(name));
        }
        return names;
    }

    /// Returns "Ready", "NotReady", or "Unknown" for the named node.
    [[nodiscard]] std::string node_status(const std::string& node) const {
        auto out  = run_kubectl("get node " + node + " --no-headers 2>&1");
        auto rows = parse_lines(out);
        if (rows.empty()) return "Unknown";
        // Columns: NAME  STATUS  ROLES  AGE  VERSION
        std::istringstream ss(rows.front());
        std::string name, status;
        if (ss >> name >> status) return status;
        return "Unknown";
    }

    /// Returns all namespace names visible in the cluster.
    [[nodiscard]] std::vector<std::string> namespaces() const {
        auto out  = run_kubectl("get namespaces --no-headers 2>&1");
        auto rows = parse_lines(out);
        std::vector<std::string> nss;
        nss.reserve(rows.size());
        for (auto& row : rows) {
            std::istringstream ss(row);
            std::string ns;
            if (ss >> ns) nss.push_back(std::move(ns));
        }
        return nss;
    }

    /// True when namespace `name` exists in the cluster.
    [[nodiscard]] bool has_namespace(const std::string& name) const {
        auto ns = namespaces();
        return std::find(ns.begin(), ns.end(), name) != ns.end();
    }

    /// Returns all deployment names in namespace `ns`.
    [[nodiscard]] std::vector<std::string> deployments(const std::string& ns) const {
        auto cmd  = "get deployments -n " + ns + " --no-headers 2>&1";
        auto out  = run_kubectl(cmd);
        auto rows = parse_lines(out);
        std::vector<std::string> names;
        names.reserve(rows.size());
        for (auto& row : rows) {
            std::istringstream ss(row);
            std::string name;
            if (ss >> name) names.push_back(std::move(name));
        }
        return names;
    }

    /// True when deployment `dep` exists in namespace `ns`.
    [[nodiscard]] bool has_deployment(const std::string& ns,
                                      const std::string& dep) const {
        auto deps = deployments(ns);
        return std::find(deps.begin(), deps.end(), dep) != deps.end();
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Path to the kubeconfig file used for all kubectl calls.
    [[nodiscard]] const std::string& kubeconfig_path() const noexcept {
        return kubeconfig_;
    }

private:
    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    std::string kubeconfig_;

    /// Run `kubectl <args>` with the stored kubeconfig, return full stdout+stderr.
    /// Throws std::runtime_error when popen() itself fails.
    [[nodiscard]] std::string run_kubectl(const std::string& args) const {
        std::string cmd = "KUBECONFIG=" + kubeconfig_ + " kubectl " + args;
        // NOLINTNEXTLINE(cert-env33-c)
        FILE* fp = popen(cmd.c_str(), "r");
        if (!fp) {
            throw std::runtime_error("K8sHpaRuntime: popen failed for: " + cmd);
        }
        std::string result;
        char buf[256];
        while (std::fgets(buf, sizeof(buf), fp)) {
            result += buf;
        }
        pclose(fp);
        return result;
    }

    /// Split multi-line string into non-empty trimmed lines.
    [[nodiscard]] static std::vector<std::string> parse_lines(const std::string& s) {
        std::vector<std::string> lines;
        std::istringstream ss(s);
        std::string line;
        while (std::getline(ss, line)) {
            // trim trailing \r
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) lines.push_back(line);
        }
        return lines;
    }

    /// Locate a readable kubeconfig file.
    /// Search order:
    ///   1. $HOME/.kube/config
    ///   2. /etc/rancher/k3s/k3s.yaml   (k3s default — may require root)
    ///   3. $KUBECONFIG env var
    [[nodiscard]] static std::string find_kubeconfig() {
        // Option 1: $HOME/.kube/config
        const char* home = std::getenv("HOME");
        if (home) {
            std::string p = std::string(home) + "/.kube/config";
            if (FILE* f = std::fopen(p.c_str(), "r")) {
                std::fclose(f);
                return p;
            }
        }
        // Option 2: k3s default location
        {
            const char* k3s = "/etc/rancher/k3s/k3s.yaml";
            if (FILE* f = std::fopen(k3s, "r")) {
                std::fclose(f);
                return std::string(k3s);
            }
        }
        // Option 3: $KUBECONFIG env var
        {
            const char* env = std::getenv("KUBECONFIG");
            if (env && env[0] != '\0') {
                return std::string(env);
            }
        }
        // Fallback: return a reasonable default even if not readable
        return std::string(home ? home : "/root") + "/.kube/config";
    }
};

} // namespace nikola::infrastructure
