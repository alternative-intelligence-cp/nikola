# Gemini Deep Research Task: Bootstrap Token Security Vulnerability

## Problem Statement

**Location**: Section 4.5 (Security Subsystem - ZeroMQ Ironhouse) + Section 9.1 (Bootstrap Sequence)

**Issue Discovered**: The 300-second bootstrap token window creates a **race condition in containerized/cloud environments** where stdout may not be accessible to administrators.

### Specific Details

1. **Current Bootstrap Token Protocol** (Section 4.5):
   ```
   1. Check Whitelist: On startup, ZAPHandler checks if whitelist file is empty
   2. Bootstrap Mode: If empty, enters BOOTSTRAP mode
   3. Token Generation: Generates 256-bit "Admin Token", prints to stdout
   4. Pairing Window: 300-second countdown begins
   5. Claiming: Admin runs `twi-ctl pair <token>`, sends public key + token hash
   6. Lockdown: Server verifies, adds to whitelist, invalidates token
   ```

2. **Vulnerability Scenarios**:
   
   **Scenario A: Docker with Log Aggregation**
   ```bash
   $ docker run -d nikola/twi:latest  # Detached mode
   # Token printed to container stdout
   # But admin can't see it - logs go to ELK stack
   # By the time admin retrieves logs, 300 seconds elapsed → Lockdown
   # System permanently locked, requires container rebuild
   ```
   
   **Scenario B: Kubernetes with CrashLoopBackOff**
   ```bash
   $ kubectl apply -f nikola-deployment.yaml
   # Pod starts, prints token, admin tries to connect
   # Connection fails (network not ready), pod crashes
   # Kubernetes restarts pod → NEW token generated
   # Admin's old token is now invalid
   # Infinite loop of restarts
   ```
   
   **Scenario C: Systemd Service with Journal**
   ```bash
   $ sudo systemctl start nikola.service
   # Token printed to journalctl
   # Admin must run: journalctl -u nikola -n 100 | grep "Bootstrap Token"
   # But systemd may buffer output, causing delay
   # Token expires before admin retrieves it
   ```

3. **Security vs Usability Tradeoff**:
   - **High Security**: Short timeout (30 seconds) → prevents brute force
   - **High Usability**: Long timeout (10 minutes) → easier for admins
   - **Problem**: Neither solves inaccessible stdout

## Research Objectives

### Primary Question
**How can we provide a secure bootstrap mechanism that works in headless, containerized, and cloud-native environments without compromising the zero-trust security model?**

### Sub-Questions to Investigate

1. **Alternative Token Delivery Methods**:
   
   **Option A: Environment Variable Override**
   ```bash
   $ docker run -e NIKOLA_BOOTSTRAP_TOKEN=<pre-shared-secret> nikola/twi
   # System accepts this token instead of generating random one
   # Admin knows token in advance, can pair immediately
   ```
   **Pro**: Works in all environments
   **Con**: Token in environment variable (visible in `docker inspect`, Kubernetes describe)
   
   **Option B: Volume-Mounted Token File**
   ```bash
   $ echo "admin-token-secret" > /secure/token.txt
   $ docker run -v /secure:/nikola/secrets:ro nikola/twi
   # System reads token from /nikola/secrets/token.txt
   ```
   **Pro**: Token in separate secure storage (K8s Secret, Docker Secret)
   **Con**: Requires volume mount, more complex setup
   
   **Option C: Kubernetes Secret Integration**
   ```yaml
   apiVersion: v1
   kind: Secret
   metadata:
     name: nikola-bootstrap
   type: Opaque
   data:
     token: <base64-encoded-token>
   ---
   env:
     - name: NIKOLA_BOOTSTRAP_TOKEN
       valueFrom:
         secretKeyRef:
           name: nikola-bootstrap
           key: token
   ```
   **Pro**: Native K8s integration, encrypted at rest
   **Con**: Kubernetes-specific, not portable
   
   **Option D: Init Container with Token Exchange**
   ```yaml
   initContainers:
   - name: token-generator
     image: nikola/init-token
     command: ["generate-token-and-wait-for-pairing"]
     volumeMounts:
     - name: shared-token
       mountPath: /shared
   ```
   **Pro**: Separates token generation from main container
   **Con**: Complex orchestration

2. **Token Persistence Strategies**:
   - **Ephemeral**: Token valid only for first boot (current behavior)
   - **Persistent**: Token stored in database, reusable across restarts
   - **Rotating**: Token changes every N hours, admin must re-pair
   - **Master Key**: One-time setup token, generates session tokens

3. **Fallback Authentication**:
   - If bootstrap token expires, how does admin recover?
   - Manual whitelist file edit? (Requires filesystem access)
   - Recovery mode via local socket? (Unix domain socket, no network)
   - Emergency reset command? (Dangerous - could be exploited)

4. **Audit Trail**:
   - How do we log bootstrap attempts for security monitoring?
   - Prometheus metrics for failed pairings?
   - Alert if >3 failed attempts in 60 seconds (brute force detection)?

## Required Deliverables

1. **Security Threat Model**:
   Analyze attack vectors for each proposed solution:
   ```
   Threat 1: Attacker intercepts environment variable
   - Impact: Complete system compromise
   - Mitigation: Use encrypted secrets, short-lived tokens
   
   Threat 2: Attacker accesses Docker logs
   - Impact: Token leakage
   - Mitigation: Clear logs after pairing, use ephemeral tokens
   
   Threat 3: Attacker brute-forces token
   - Impact: Unauthorized access
   - Mitigation: Rate limiting, high entropy (256-bit), short timeout
   ```

2. **Implementation Specification**:
   Complete code for:
   - Environment variable bootstrap
   - Volume-mounted token file
   - Kubernetes Secret integration
   - Recovery mode for locked-out admins

3. **Deployment Guides**:
   Step-by-step for each platform:
   - **Docker Compose**: How to generate and inject token
   - **Kubernetes**: How to create Secret and reference it
   - **Bare Metal / Systemd**: How to use stdout safely
   - **AWS ECS**: How to use AWS Secrets Manager
   - **Azure Container Instances**: How to use Key Vault

4. **Comparison Matrix**:
   ```
   Solution                  | Security | Usability | Platform Support | Complexity
   --------------------------|----------|-----------|------------------|------------
   Stdout (current)          | High     | Low       | All              | Low
   Environment Variable      | Medium   | High      | All              | Low
   Volume-Mounted File       | High     | Medium    | All              | Medium
   Kubernetes Secret         | Very High| High      | K8s only         | Medium
   Init Container            | High     | High      | K8s/Docker       | High
   Master Key + Sessions     | Very High| Very High | All              | Very High
   ```

## Proposed Solution: Tiered Bootstrap Strategy

### Tier 1: Headless Mode (Environment Variable)
```cpp
std::optional<std::string> get_bootstrap_token() {
    // Check for environment variable override (HIGHEST PRIORITY)
    const char* env_token = std::getenv("NIKOLA_BOOTSTRAP_TOKEN");
    if (env_token && validate_token_format(env_token)) {
        spdlog::warn("Using bootstrap token from NIKOLA_BOOTSTRAP_TOKEN");
        spdlog::warn("This should only be used in containerized environments");
        return std::string(env_token);
    }
    
    // Check for file-based token (MEDIUM PRIORITY)
    std::filesystem::path token_file = "/nikola/secrets/bootstrap.token";
    if (std::filesystem::exists(token_file)) {
        std::ifstream f(token_file);
        std::string token;
        std::getline(f, token);
        if (validate_token_format(token)) {
            spdlog::info("Using bootstrap token from {}", token_file);
            return token;
        }
    }
    
    // Generate random token (LOWEST PRIORITY - current behavior)
    std::string random_token = generate_secure_random_token();
    std::cout << "╔════════════════════════════════════════════╗\n";
    std::cout << "║  NIKOLA BOOTSTRAP TOKEN (300s window)     ║\n";
    std::cout << "╠════════════════════════════════════════════╣\n";
    std::cout << "║  " << random_token << "  ║\n";
    std::cout << "╚════════════════════════════════════════════╝\n";
    std::cout << std::flush;  // CRITICAL: Force flush for Docker logs
    
    return random_token;
}
```

### Tier 2: Recovery Mode (Unix Socket)
```cpp
// If bootstrap token expires and no keys whitelisted,
// expose Unix domain socket for local recovery
void enable_recovery_mode() {
    if (!whitelist_empty()) return;  // Only if locked out
    
    // Create Unix socket (only accessible to root/same user)
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr;
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, "/var/run/nikola-recovery.sock");
    
    bind(sock, (struct sockaddr*)&addr, sizeof(addr));
    listen(sock, 1);
    
    spdlog::warn("Bootstrap token expired. Recovery mode enabled.");
    spdlog::warn("Connect via: sudo twi-ctl recover /var/run/nikola-recovery.sock");
    
    // Accept ONE connection, allow ONE pairing, then close
    // (Prevents persistent recovery attack surface)
}
```

### Tier 3: Audit Logging
```cpp
void log_bootstrap_attempt(const std::string& client_pubkey, bool success) {
    nlohmann::json event = {
        {"timestamp", std::chrono::system_clock::now()},
        {"event", "bootstrap_pairing"},
        {"client_pubkey", client_pubkey},
        {"success", success},
        {"remote_ip", get_client_ip()}
    };
    
    audit_logger->info(event.dump());
    
    // Prometheus metric
    bootstrap_attempts_total.Increment();
    if (!success) {
        bootstrap_failures_total.Increment();
        
        // Rate limit: Too many failures → temporary ban
        if (get_failure_count_last_minute() > 5) {
            spdlog::error("Excessive bootstrap failures - entering lockdown");
            enable_emergency_lockdown();
        }
    }
}
```

## Research Questions

1. **Cloud-Native Security**:
   - How does Hashicorp Vault handle bootstrap credentials?
   - How does Consul implement secure agent joining?
   - How does etcd handle initial cluster authentication?

2. **Container Security Standards**:
   - What does the CIS Docker Benchmark recommend?
   - What does the CIS Kubernetes Benchmark recommend?
   - What is Google's Binary Authorization approach?

3. **Credential Rotation**:
   - Should the bootstrap token rotate after first use?
   - Should we support multiple valid tokens simultaneously?
   - How do we deprecate old tokens gracefully?

4. **Compliance Requirements**:
   - Does this meet SOC 2 requirements?
   - Does this meet ISO 27001 requirements?
   - Does this meet NIST 800-53 requirements?

## Success Criteria

- [ ] Works in Docker detached mode
- [ ] Works in Kubernetes with Secret
- [ ] Works in systemd with journald
- [ ] Recovery mode for locked-out admins
- [ ] Audit logging for all bootstrap attempts
- [ ] No secrets in plaintext logs
- [ ] Passes security penetration testing
- [ ] Documentation for all deployment scenarios

## Output Format

Please provide:
1. **Threat Model** (2-3 pages): Attack vectors and mitigations
2. **Implementation Guide** (3-5 pages): Code for all tiers
3. **Deployment Playbooks** (2-3 pages): Step-by-step for each platform
4. **Security Audit** (1-2 pages): Compliance checklist
5. **Recovery Procedures** (1 page): What to do if locked out

## Additional Context

This affects:
- Section 4.5: Security Subsystem (ZAPHandler)
- Section 8.1: Phase 0 (bootstrap sequence)
- Section 10.3: CLI Controller (twi-ctl pairing)
- Docker deployment (Section 11-G)

Real-world deployment requirements:
- Must work in air-gapped environments (no internet access)
- Must work in zero-trust networks (no SSH access to containers)
- Must be automatable (Infrastructure as Code with Terraform/Ansible)

---

**Priority**: P2 - HIGH (Blocks production deployment)
**Estimated Research Time**: 4-6 hours
**Dependencies**: TASK_BOOTSTRAP_TIMING (300-second window coordination)
