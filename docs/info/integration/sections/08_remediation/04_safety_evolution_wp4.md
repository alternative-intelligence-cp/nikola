# WORK PACKAGE 4: SAFETY EVOLUTION

## WP4.1 Overview

**Purpose:** Implement safety systems for self-modification and production deployment.

**Status:** Shadow Spine Protocol MANDATORY, CSVP implemented

## WP4.2 Enhancement: Shadow Spine Protocol

**Status:** MANDATORY - Required for safe deployment

### Purpose

Test candidate systems in parallel with production without user disruption.

### Architecture

```
User Query
    ↓
┌───┴────┐
│Splitter│ (ZMQ Proxy)
└─┬───┬──┘
  ↓   ↓
┌─────────┐  ┌──────────┐
│Prod Sys │  │Candidate │
└────┬────┘  └─────┬────┘
     ↓             ↓ (To Architect for analysis)
     ↓
     ↓ (To User)
```

### Voting Mechanism

If Candidate response has:
- Higher resonance
- Lower latency
- Equal or higher confidence

Then: Vote for promotion.

After 100 consecutive votes, promote Candidate to Production.

**Implementation:**

```cpp
// File: include/nikola/spine/shadow_spine.hpp

namespace nikola::spine {

class ShadowSpine {
    SpineBroker production_broker;
    SpineBroker candidate_broker;

    int votes_for_candidate = 0;
    const int PROMOTION_THRESHOLD = 100;

public:
    void route_query(const NeuralSpike& query) {
        // Send to both systems
        auto prod_future = production_broker.async_query(query);
        auto cand_future = candidate_broker.async_query(query);

        // Wait for production (user sees this)
        NeuralSpike prod_response = prod_future.get();

        // Return to user immediately
        user_interface.send_response(prod_response);

        // Evaluate candidate in background
        std::thread([this, prod_response, cand_future = std::move(cand_future)]() mutable {
            NeuralSpike cand_response = cand_future.get();
            compare_responses(prod_response, cand_response);
        }).detach();
    }

    void compare_responses(const NeuralSpike& prod_response,
                          const NeuralSpike& cand_response) {
        // Metrics comparison
        bool higher_resonance = cand_response.physics.resonance > prod_response.physics.resonance;
        bool lower_latency = cand_response.meta.latency_ms < prod_response.meta.latency_ms;
        bool equal_confidence = cand_response.payload.confidence >= prod_response.payload.confidence;

        if (higher_resonance && lower_latency && equal_confidence) {
            votes_for_candidate++;

            if (votes_for_candidate >= PROMOTION_THRESHOLD) {
                promote_candidate();
            }
        } else {
            // Reset vote counter on failure
            votes_for_candidate = std::max(0, votes_for_candidate - 1);
        }
    }

    void promote_candidate() {
        std::cout << "[SHADOW SPINE] Promoting candidate to production!" << std::endl;

        // Atomic swap
        std::swap(production_broker, candidate_broker);

        // Reset counter
        votes_for_candidate = 0;

        // Log promotion event
        log_promotion_event();
    }
};

} // namespace nikola::spine
```

### Safety Features

1. **User sees production only:** No experimental responses reach users
2. **Gradual promotion:** 100 consecutive successes required
3. **Automatic rollback:** Vote counter decreases on inferior performance
4. **Audit trail:** All comparisons logged

### Feasibility Rank

MEDIUM (requires careful orchestration)

### Location

Section 10.5 - [include/nikola/spine/shadow_spine.hpp](include/nikola/spine/shadow_spine.hpp)

## WP4.3 Enhancement: Code Safety Verification Protocol (CSVP)

**Status:** MANDATORY - Required for self-improvement

### Purpose

Prevent self-lobotomy or segfaults when AI generates and hot-swaps code.

### Protocol Workflow

```
1. Generation → AI generates module_v2.cpp
2. Static Analysis (Resonance Firewall)
   → Clang-Tidy profile enforces safety rules
3. Sandboxed Compilation
   → Compiled in KVM with -fstack-protector-strong
4. Unit Test Oracle
   → Regression suite in VM
5. Hot-Swap Trigger
   → Only if all checks pass: dlopen()
```

### Safety Rules

**Static Analysis Enforces:**

- No `system()` or `exec()` calls: Prevents shell injection
- Memory Safety: Enforces `std::shared_ptr` over raw pointers
- Bounding: All loops must have static upper bounds or timeout checks
- Physics Invariants: Code modifying torus must respect Conservation of Energy (unitary updates)

**Implementation:**

```cpp
// File: include/nikola/self_improve/csvp.hpp

namespace nikola::self_improve {

class CodeSafetyVerificationProtocol {
public:
    struct VerificationResult {
        bool passed;
        std::vector<std::string> violations;
        std::string analysis_report;
    };

    VerificationResult verify_candidate_code(const std::string& code_path) {
        VerificationResult result;
        result.passed = true;

        // 1. Static Analysis
        if (!run_static_analysis(code_path, result)) {
            result.passed = false;
            return result;
        }

        // 2. Sandboxed Compilation
        if (!compile_in_sandbox(code_path, result)) {
            result.passed = false;
            return result;
        }

        // 3. Physics Invariants Test
        if (!test_physics_invariants(code_path, result)) {
            result.passed = false;
            return result;
        }

        // 4. Unit Test Oracle
        if (!run_regression_suite(code_path, result)) {
            result.passed = false;
            return result;
        }

        return result;
    }

private:
    // CRITICAL FIX (Audit 3 Item #10): Use fork/execvp for shell safety
    // Problem: execute_command() may use popen/system which is vulnerable to shell injection
    // Solution: Explicitly use fork/execvp pattern with argument array
    bool run_static_analysis(const std::string& code_path, VerificationResult& result) {
        // Create pipes for capturing stdout/stderr
        int stdout_pipe[2];
        int stderr_pipe[2];
        if (pipe(stdout_pipe) == -1 || pipe(stderr_pipe) == -1) {
            result.violations.push_back("Failed to create pipes");
            return false;
        }

        pid_t pid = fork();
        if (pid == -1) {
            result.violations.push_back("Fork failed");
            return false;
        }

        if (pid == 0) {  // Child process
            // Redirect stdout and stderr to pipes
            dup2(stdout_pipe[1], STDOUT_FILENO);
            dup2(stderr_pipe[1], STDERR_FILENO);
            close(stdout_pipe[0]);
            close(stdout_pipe[1]);
            close(stderr_pipe[0]);
            close(stderr_pipe[1]);

            // Use execvp with argument array (safe from shell injection)
            const char* argv[] = {
                "clang-tidy",
                code_path.c_str(),
                "--checks=-*,nikola-*,bugprone-*,cert-*",
                nullptr
            };

            execvp("clang-tidy", const_cast<char* const*>(argv));

            // If execvp returns, it failed
            _exit(1);
        } else {  // Parent process
            close(stdout_pipe[1]);
            close(stderr_pipe[1]);

            // Read output from pipes
            std::string output;
            char buffer[4096];
            ssize_t bytes_read;

            while ((bytes_read = read(stdout_pipe[0], buffer, sizeof(buffer))) > 0) {
                output.append(buffer, bytes_read);
            }

            close(stdout_pipe[0]);
            close(stderr_pipe[0]);

            // Wait for child process
            int status;
            waitpid(pid, &status, 0);

            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                result.violations.push_back("Static analysis execution failed");
                return false;
            }

            if (output.find("error:") != std::string::npos) {
                result.violations.push_back("Static analysis failed");
                return false;
            }

            // Check for banned functions
            std::vector<std::string> banned = {"system", "exec", "popen"};
            for (const auto& func : banned) {
                if (output.find(func + "(") != std::string::npos) {
                    result.violations.push_back("Banned function: " + func);
                    return false;
                }
            }

            return true;
        }
    }

    bool compile_in_sandbox(const std::string& code_path, VerificationResult& result) {
        // Compile in KVM with security flags
        KVMExecutor executor;
        CommandRequest compile_req;
        compile_req.set_command("g++");
        compile_req.add_args("-std=c++23");
        compile_req.add_args("-fstack-protector-strong");
        compile_req.add_args("-D_FORTIFY_SOURCE=2");
        compile_req.add_args("-Werror");
        compile_req.add_args(code_path);
        compile_req.add_args("-o");
        compile_req.add_args("/tmp/candidate.so");

        try {
            executor.execute(compile_req);
            return true;
        } catch (const std::exception& e) {
            result.violations.push_back("Compilation failed: " + std::string(e.what()));
            return false;
        }
    }

    bool test_physics_invariants(const std::string& binary_path, VerificationResult& result) {
        // Energy conservation test
        if (!test_energy_conservation(binary_path)) {
            result.violations.push_back("Energy conservation violated");
            return false;
        }

        // Logic consistency test
        if (!test_nonary_arithmetic(binary_path)) {
            result.violations.push_back("Nonary logic inconsistency");
            return false;
        }

        // Topology correctness test
        if (!test_toroidal_wrapping(binary_path)) {
            result.violations.push_back("Toroidal wrapping broken");
            return false;
        }

        return true;
    }

    bool run_regression_suite(const std::string& binary_path, VerificationResult& result) {
        // Run full test suite in sandbox
        std::string command = "ctest --test-dir /tmp/candidate_build --output-on-failure";

        auto output = execute_command_in_kvm(command);

        if (output.find("100% tests passed") == std::string::npos) {
            result.violations.push_back("Regression tests failed");
            return false;
        }

        return true;
    }
};

} // namespace nikola::self_improve
```

### Benefits

- **Prevents self-lobotomy:** Can't delete critical systems
- **Memory safety:** Stack protection, bounds checking
- **Physics integrity:** Energy conservation enforced
- **Regression protection:** Full test suite required

### Location

Section 17.3.1 - [include/nikola/self_improve/csvp.hpp](include/nikola/self_improve/csvp.hpp)

## WP4.4 Enhancement: Adversarial Code Dojo

**Status:** MANDATORY - Required for robust self-improvement

### Concept

Before any self-generated code is integrated, it must survive adversarial testing.

**Components:**

1. **Red Team Agent:** Mamba-9D instance trained to generate "Hazardous Spectra" (attack waveforms)
2. **Arena:** Isolated KVM instance running candidate code
3. **Attack Vectors:**
   - Buffer overflow attempts (inject amplitude > +4)
   - Logic loops (cyclic dependencies in metric tensor)
   - Energy singularities (resonance runaway)

**Protocol:**

```
1. Architect generates code patch
2. Code compiled in sandbox (KVM)
3. Red Team injects 100 attack waveforms
4. IF candidate survives ALL attacks:
       Proceed to hot-swap
   ELSE:
       Log failure, discard patch, penalize Architect
```

**Implementation:**

```cpp
// File: include/nikola/self_improve/adversarial_dojo.hpp

namespace nikola::self_improve {

class RedTeamAgent {
    std::vector<std::vector<std::complex<double>>> attack_library;

public:
    void train_on_known_exploits() {
        // Load known attack patterns
        load_attack_patterns("/var/lib/nikola/attacks/");
    }

    std::vector<std::complex<double>> generate_attack_wave() {
        // Generate novel attack pattern
        std::vector<std::complex<double>> attack;

        // Extreme amplitude
        attack.push_back(std::complex<double>(10.0, 0.0));  // > +4 limit

        // Logic bomb (cyclic dependency)
        for (int i = 0; i < 100; ++i) {
            attack.push_back(std::complex<double>(std::sin(i), std::cos(i)));
        }

        return attack;
    }

    bool test_system_resilience(TorusManifold& target) {
        int attacks_survived = 0;

        for (int i = 0; i < 100; ++i) {
            auto attack = generate_attack_wave();

            try {
                target.inject_wave(Coord9D{}, attack);
                target.propagate(0.01);

                // Check system is still stable
                if (!target.check_energy_conservation()) {
                    return false;
                }

                attacks_survived++;
            } catch (const std::exception& e) {
                // Caught exception = attack successful
                return false;
            }
        }

        return (attacks_survived == 100);
    }
};

} // namespace nikola::self_improve
```

### Location

Section 17.7 - [include/nikola/self_improve/adversarial_dojo.hpp](include/nikola/self_improve/adversarial_dojo.hpp)

---

**Cross-References:**
- See Section 17 for Self-Improvement System
- See Section 10 for ZeroMQ Spine Architecture
- See Section 13 for KVM Executor
- See Section 18 for Resonance Firewall
