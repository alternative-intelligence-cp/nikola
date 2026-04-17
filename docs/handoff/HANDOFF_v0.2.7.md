# Nikola v0.2.x Series Handoff — April 17, 2026

## Series Summary: Intelligence in the Wild

The 0.2.x series (8 releases, v0.2.0–v0.2.7) connected Nikola to the external world,
giving it autonomy, goals, personality, and production infrastructure.

---

## Release Summary

| Release | Theme | Key Deliverables |
|---------|-------|------------------|
| v0.2.0 | External Data Retrieval | TavilyOracle, FirecrawlOracle, ResearchRouter, OraclePool, CoherenceOracle |
| v0.2.1 | GoalSystem & Motivation | GoalDAG, tier-based rewards, serialization, cycle prevention |
| v0.2.2 | Training Ingestion | AutoIngestor, IngestionFilter, SimHash dedup, byte budget, safety check |
| v0.2.3 | Identity & Personality | PersonalityDrift, PreferenceEngine, Autobiography, SkillTracker, ValueFormation |
| v0.2.4 | Docker & Deployment | Multi-stage builds (GPU/CPU), Docker Compose 4-layer, ZMQ spine networking |
| v0.2.5 | Inference Server | NikolaInference engine, HTTP /v1/generate + /v1/embed, lightweight runner |
| v0.2.6 | Security Polish | SecurityPipeline (CSVP→Blacklist→KVM→eBPF→Anomaly), production KVM sandbox |
| v0.2.7 | Audit & Documentation | 46 integration tests, 8 performance benchmarks, security audit, docs update |

---

## Test Suite

| Metric | Value |
|--------|-------|
| Total CTest targets | ~220 |
| Integration tests (v0.2.x) | 46 test cases, 142 assertions |
| Performance benchmarks | 8 suites |
| Pre-existing failures | 2 (Phase160CudaSelectiveScan_Benchmark, Phase139ObservabilityWiring) |

---

## Architecture Additions (0.2.x)

```
src/autonomy/
  ├── tavily_oracle.cpp          — Tavily search API integration
  ├── firecrawl_oracle.cpp       — Firecrawl web scraping
  ├── research_router.cpp        — Query classification & routing
  ├── oracle_pool.cpp            — Multi-oracle aggregation
  ├── coherence_oracle.cpp       — Cross-source coherence scoring
  ├── goal_system.cpp            — DAG-based goal management
  ├── auto_ingestor.cpp          — File monitoring & ingestion
  ├── ingestion_filter.cpp       — Budget, safety, dedup filtering
  ├── ingestion_orchestrator.cpp — Ingestion pipeline coordination
  ├── personality_drift.cpp      — Trait evolution from experience
  ├── preference_engine.cpp      — Learned preferences with decay
  ├── autobiography.cpp          — Narrative self-model
  ├── skill_tracker.cpp          — Competency tracking
  ├── value_formation.cpp        — Value system development
  ├── decision_loop.cpp          — Main cognitive tick loop
  └── self_improvement_engine.cpp — Code generation & deployment

src/security/
  ├── code_safety_verifier.cpp   — CSVP static analysis
  ├── code_pattern_blacklist.cpp — Dangerous pattern detection
  ├── kvm_sandbox.hpp            — KVM micro-VM isolation
  ├── ebpf_monitor.hpp           — eBPF runtime monitoring
  ├── anomaly_detector.cpp       — Statistical anomaly detection
  └── hybrid_verifier.cpp        — Ed25519 + SPHINCS+ signatures

src/inference/
  ├── nikola_inference.cpp       — Main inference engine
  └── http_server.cpp            — REST API server
```

---

## Security Audit Summary (v0.2.7)

- **18 findings**: 1 critical (mitigated), 3 high, 4 medium, 5 low, 5 info
- **No exploitable vulnerabilities** in current deployment
- Key recommendations for 0.3.x:
  1. Replace `popen()` with `fork()/execvp()` in SIE
  2. HTTP server: bind localhost, add auth, add thread pool
  3. Add IP-range validation to HttpClient for SSRF prevention
- Full report: `docs/SECURITY_AUDIT_v0.2.7.md`

---

## Performance Baseline (v0.2.7)

| Benchmark | Result |
|-----------|--------|
| GoalDAG 1000+ goals | < 5s |
| Ingestion chunking | Throughput measured |
| SimHash filtering | High throughput |
| PersonalityDrift + PreferenceEngine | 10K ops measured |
| LMDB persistence | Sub-second save/load |
| SecurityPipeline per-module | Latency measured |
| DecisionLoop 1000 ticks | ~70s (~14 ticks/s) |

---

## Known Issues

1. **NikolaInference requires ONNX model** — tests skip gracefully when model unavailable
2. **HTTP server binds 0.0.0.0** — needs localhost default for security
3. **Phase160CudaSelectiveScan_Benchmark** — pre-existing failure (CUDA-specific)
4. **Phase139ObservabilityWiring** — pre-existing intermittent failure

---

## What's Next: 0.3.x — Production Hardening

Planned focus areas:
- HTTP server hardening (auth, rate limiting, localhost binding)
- SIE `popen()` → `fork()/execvp()` migration
- SSRF prevention (IP validation)
- Content moderation upgrade (ML-based safety filter)
- Thread pool for inference server
- Scaling & multi-node support

---

## Key Paths

| Component | Path |
|-----------|------|
| Source | `REPOS/nikola/src/` |
| Tests | `REPOS/nikola/tests/` |
| Integration tests | `tests/integration/v02x_integration_test.cpp` |
| Performance tests | `tests/integration/v027_performance_audit_test.cpp` |
| Security audit | `docs/SECURITY_AUDIT_v0.2.7.md` |
| Engineering report | `docs/info/engineering/` |
| Docker config | `docs/DOCKER.md` |
| Build | `cd build && cmake .. && make -j$(nproc)` |
| Branch | `dev` (merge to `main` for releases) |
| GitHub | `alternative-intelligence-cp/nikola` |
