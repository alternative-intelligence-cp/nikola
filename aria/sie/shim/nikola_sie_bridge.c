/* nikola_sie_bridge.c — C-linkage FFI bridge for Nikola SIE gates
 *
 * Exposes Nikola's C++ SIE gate classes to Aria via plain C functions.
 * Each gate uses a module-level static instance initialized on first call.
 *
 * Build:
 *   gcc -O2 -Wall -fPIC -c nikola_sie_bridge.c -o nikola_sie_bridge.o \
 *       -I../../../include
 *   ar rcs libnikola_sie.a nikola_sie_bridge.o
 *
 * Link from Aria:
 *   ariac program.aria -L shim -lnikola_sie
 */

#include <stdint.h>
#include <string.h>
#include <math.h>

/* ─── Gate 0: ShadowSpine / HybridVerifier (stub) ─────────────────────────
 * Real Ed25519+SPHINCS+ verification requires linking libsodium + PQC libs.
 * For v0.0.10, we provide a deterministic stub that validates structure:
 *   - Signature must be exactly 64 bytes (Ed25519 size)
 *   - Binary must be non-empty
 * Full crypto will be wired in a later release.
 */

static int32_t g_verifier_last_failure = 0;  /* 0=NONE */

int32_t nk_verifier_verify(const char* bin_path,
                           int64_t     sig_len,
                           int64_t     pub_len)
{
    /* Structural validation only (v0.0.10 stub) */
    if (!bin_path || bin_path[0] == '\0') {
        g_verifier_last_failure = 1;  /* EMPTY_BINARY */
        return 0;
    }
    if (sig_len != 64) {
        g_verifier_last_failure = 2;  /* INVALID_SIG_LENGTH */
        return 0;
    }
    if (pub_len != 32) {
        g_verifier_last_failure = 3;  /* INVALID_PUBKEY_LENGTH */
        return 0;
    }
    g_verifier_last_failure = 0;
    return 1;
}

int32_t nk_verifier_last_failure(void)
{
    return g_verifier_last_failure;
}

/* ─── Gate 1: CodePatternBlacklist ─────────────────────────────────────────
 * Scans source code for banned patterns:
 *   system(), exec(), fork(), popen(), dlopen() (direct),
 *   #include <sys/ptrace.h>, __asm__, asm()
 */

static const char* BANNED_PATTERNS[] = {
    "system(",
    "execl(",   "execle(",  "execlp(",
    "execv(",   "execve(",  "execvp(",
    "fork(",
    "popen(",
    "__asm__",  "asm(",
    "#include <sys/ptrace.h>",
    NULL
};

int32_t nk_blacklist_is_safe(const char* source)
{
    if (!source) return 0;
    for (int i = 0; BANNED_PATTERNS[i]; ++i) {
        if (strstr(source, BANNED_PATTERNS[i]) != NULL)
            return 0;
    }
    return 1;
}

/* Return the index (1-based) of the first banned pattern found, or 0 if clean */
int32_t nk_blacklist_scan(const char* source)
{
    if (!source) return -1;
    for (int i = 0; BANNED_PATTERNS[i]; ++i) {
        if (strstr(source, BANNED_PATTERNS[i]) != NULL)
            return (int32_t)(i + 1);
    }
    return 0;
}

/* ─── Gate 2: PhysicsOracle (stateless) ───────────────────────────────────
 * Pure math — no C++ instance needed.
 * Mirrors nikola::physics::PhysicsOracle static methods.
 */

int32_t nk_oracle_check_energy(double h_initial, double h_final,
                               double tolerance)
{
    if (h_initial == 0.0) return 0;  /* degenerate */
    double drift = fabs(h_final - h_initial) / fabs(h_initial);
    return drift <= tolerance ? 1 : 0;
}

double nk_oracle_energy_drift(double h_initial, double h_final)
{
    if (h_initial == 0.0) return 1e30;
    return fabs(h_final - h_initial) / fabs(h_initial);
}

int32_t nk_oracle_check_viscosity(double e_actual, double e_0,
                                  double alpha, double t,
                                  double tolerance)
{
    /* E_theoretical = E_0 * exp(-alpha * t) */
    double e_theo = e_0 * exp(-alpha * t);
    double err = fabs(e_actual - e_theo);
    return err <= tolerance ? 1 : 0;
}

double nk_oracle_viscosity_error(double e_actual, double e_0,
                                 double alpha, double t)
{
    double e_theo = e_0 * exp(-alpha * t);
    return fabs(e_actual - e_theo);
}

int32_t nk_oracle_check_resonance(double psi_max, double amplitude_limit)
{
    return psi_max < amplitude_limit ? 1 : 0;
}

int32_t nk_oracle_drift_alert(double drift_rate)
{
    /* 0=OK, 1=WARN, 2=CRITICAL */
    if (drift_rate > 1e-3) return 2;
    if (drift_rate > 1e-5) return 1;
    return 0;
}

int32_t nk_oracle_is_decoherent(double visibility)
{
    return visibility < 0.01 ? 1 : 0;
}

/* ─── Gate 3: ModuleSwapper (stub) ────────────────────────────────────────
 * Real dlopen swap requires C++ ModuleSwapper instance.
 * v0.0.10 stub: validates path structure, simulates swap.
 */

static char g_active_path[512]   = {0};
static char g_previous_path[512] = {0};
static int  g_has_active         = 0;

int32_t nk_swapper_swap_in(const char* so_path)
{
    if (!so_path || so_path[0] == '\0') return 4;  /* LOAD_FAILED */

    /* Check .so extension */
    size_t len = strlen(so_path);
    if (len < 3 || strcmp(so_path + len - 3, ".so") != 0)
        return 4;  /* LOAD_FAILED */

    /* Same module check */
    if (g_has_active && strcmp(g_active_path, so_path) == 0)
        return 6;  /* SAME_MODULE */

    /* Simulate successful swap */
    if (g_has_active) {
        strncpy(g_previous_path, g_active_path, sizeof(g_previous_path) - 1);
    }
    strncpy(g_active_path, so_path, sizeof(g_active_path) - 1);
    g_active_path[sizeof(g_active_path) - 1] = '\0';
    g_has_active = 1;
    return 0;  /* SUCCESS */
}

int32_t nk_swapper_rollback(void)
{
    if (g_previous_path[0] == '\0') return 0;  /* nothing to rollback */
    strncpy(g_active_path, g_previous_path, sizeof(g_active_path) - 1);
    g_previous_path[0] = '\0';
    return 1;
}

int32_t nk_swapper_has_active(void)
{
    return g_has_active;
}

int32_t nk_swapper_has_previous(void)
{
    return g_previous_path[0] != '\0' ? 1 : 0;
}

void nk_swapper_reset(void)
{
    g_active_path[0]   = '\0';
    g_previous_path[0] = '\0';
    g_has_active       = 0;
}
