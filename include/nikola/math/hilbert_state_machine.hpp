#pragma once
// =============================================================================
// GAP-009 — 9D Hilbert Curve State Machine (Hamilton & Rau-Chaplin 2007)
//
// MATHEMATICAL BACKGROUND:
//   The original hypothesis of a flat 512-entry HILBERT_ROTATION_9D indexed by
//   (rotation XOR rotated) is algebraically insufficient for 9D.  XOR is an
//   Abelian (commutative) operation; the 9D rotation group SO(9) is
//   non-Abelian.  XOR can encode the 2^9 = 512 entry-point reflection masks
//   but CANNOT encode the 9 cyclic axis permutations.
//
//   Correct state space = n × 2^n = 9 × 512 = 4,608 compound states, where
//   each state is the pair (e, d):
//     e ∈ {0..511}  — n-bit reflection mask (entry point)
//     d ∈ {0..8}    — cyclic axis shift (principal direction)
//
//   Reference: Hamilton & Rau-Chaplin (2007), "Compact Hilbert Indices for
//   Multi-Dimensional Data", Dalhousie University CS-2006-07.
//
// ALGORITHM (Hamilton FSM, 3-step per hierarchical level):
//   1. Reflect: w = gray_code_inverse( rotr_9d(l XOR e, d) )
//   2. E_TABLE lookup: ew = e(w)   — base entry point of w-th sub-hypercube
//   3. D_TABLE lookup: dw = d(w)   — base axis shift of w-th sub-hypercube
//   4. State update:
//        e_next = e XOR rotr_9d(ew, d)
//        d_next = (d + dw + 1) % 9
//
// NOTE ON SKILLING (2004) BUG:
//   The Skilling in-place algorithm contains a published off-by-one error:
//     for (i = n-1; i >= 0; i--) X[i] ^= X[i-1];   // BUG: accesses X[-1]
//   The correct bound is i > 0 (strict greater-than, not >=).
//   This implementation uses Hamilton's approach and avoids the Skilling path.
//
// ARCHITECTURE — Dual 512-Entry L1-Cached Tables:
//   Rather than a monolithic 4608×512 matrix (4.7 MB, always L3), the FSM is
//   decomposed into two independent 512-entry arrays:
//     HILBERT_E_TABLE_9D[512]  — uint16_t each → 1,024 bytes
//     HILBERT_D_TABLE_9D[512]  — uint8_t  each →   512 bytes
//   Total: 1,536 bytes — fits entirely in L1 data cache (1-cycle fetch).
// =============================================================================

#include <array>
#include <bit>        // std::rotr (C++20), std::countr_zero (C++20)
#include <cstdint>

namespace nikola::math {

// ---------------------------------------------------------------------------
// Architectural constants
// ---------------------------------------------------------------------------
inline constexpr int     HILBERT_N          = 9;
inline constexpr int     HILBERT_NUM_CELLS  = 1 << HILBERT_N;   // 512
inline constexpr int     HILBERT_TOTAL_STATES = HILBERT_N * HILBERT_NUM_CELLS; // 4608

// ---------------------------------------------------------------------------
// Compound state type
// ---------------------------------------------------------------------------
struct HilbertState {
    uint16_t e;   ///< Reflection mask: entry point, 9 bits, range [0..511]
    uint8_t  d;   ///< Cyclic axis shift, range [0..8]
};

// ---------------------------------------------------------------------------
// Core bitwise transforms
// ---------------------------------------------------------------------------

/// Standard Binary Reflected Gray Code: gc(x) = x XOR (x >> 1)
[[nodiscard]] constexpr uint16_t gray_code(uint16_t x) noexcept {
    return static_cast<uint16_t>(x ^ (x >> 1));
}

/// Inverse Gray Code: recovers rank w from transformed coordinate t.
/// Iterative prefix-XOR scan — O(log n) for fixed word sizes.
[[nodiscard]] constexpr uint16_t gray_code_inverse(uint16_t x) noexcept {
    uint16_t result = x;
    for (uint16_t tmp = x; tmp >>= 1; result ^= tmp) {}
    return result;
}

/// Bitwise right-rotation confined strictly to HILBERT_N (9) bits.
/// Uses only the low 9 bits of val; result is masked to 9 bits.
[[nodiscard]] constexpr uint16_t rotr_9d(uint16_t val, int shift) noexcept {
    shift = ((shift % HILBERT_N) + HILBERT_N) % HILBERT_N;   // normalise, handle 0
    if (shift == 0) return val & (HILBERT_NUM_CELLS - 1);
    const uint16_t mask = static_cast<uint16_t>(HILBERT_NUM_CELLS - 1);
    val &= mask;
    return static_cast<uint16_t>(((val >> shift) | (val << (HILBERT_N - shift))) & mask);
}

// ---------------------------------------------------------------------------
// Hamilton base property functions
// ---------------------------------------------------------------------------

/// e(w): Base entry point of the w-th sub-hypercube.
/// e(0) = 0;  e(w) = gc(2 * floor((w-1)/2))  for w > 0.
[[nodiscard]] constexpr uint16_t calc_hilbert_e(uint16_t w) noexcept {
    if (w == 0) return 0;
    return gray_code(static_cast<uint16_t>(2u * ((w - 1u) / 2u)));
}

/// d(w): Principal axis shift of the w-th sub-hypercube.
/// d(0) = 0;  d(w) = ctz(w)  for w > 0, clamped to [0..8].
[[nodiscard]] constexpr uint8_t calc_hilbert_d(uint16_t w) noexcept {
    if (w == 0) return 0;
    // std::countr_zero is defined for unsigned types; clamp result to n-1
    int ctz = std::countr_zero(static_cast<unsigned int>(w));
    if (ctz >= HILBERT_N) ctz = HILBERT_N - 1;
    return static_cast<uint8_t>(ctz);
}

// ---------------------------------------------------------------------------
// Compile-time 512-entry table generators
// ---------------------------------------------------------------------------

/// Generates HILBERT_E_TABLE_9D[512] at compile time.
/// E_TABLE[w] = base entry point e(w).
[[nodiscard]] constexpr auto generate_hilbert_e_table() noexcept {
    std::array<uint16_t, HILBERT_NUM_CELLS> table{};
    for (int w = 0; w < HILBERT_NUM_CELLS; ++w) {
        table[static_cast<std::size_t>(w)] = calc_hilbert_e(static_cast<uint16_t>(w));
    }
    return table;
}

/// Generates HILBERT_D_TABLE_9D[512] at compile time.
/// D_TABLE[w] = base axis shift d(w).
[[nodiscard]] constexpr auto generate_hilbert_d_table() noexcept {
    std::array<uint8_t, HILBERT_NUM_CELLS> table{};
    for (int w = 0; w < HILBERT_NUM_CELLS; ++w) {
        table[static_cast<std::size_t>(w)] = calc_hilbert_d(static_cast<uint16_t>(w));
    }
    return table;
}

// Tables are instantiated once at compile time (.rodata) — zero runtime cost.
/// E_TABLE[w]: base entry point for the w-th sub-hypercube (512 × uint16_t = 1,024 B)
inline constexpr auto HILBERT_E_TABLE_9D = generate_hilbert_e_table();
/// D_TABLE[w]: base axis shift for the w-th sub-hypercube  (512 × uint8_t  =   512 B)
inline constexpr auto HILBERT_D_TABLE_9D = generate_hilbert_d_table();

// ---------------------------------------------------------------------------
// State transition (single step)
// ---------------------------------------------------------------------------

/// Given current state (e, d) and sub-hypercube cell coordinate l (9 bits),
/// returns the scalar rank w and the next FSM state.
///   1. t = rotr_9d(l XOR e, d)
///   2. w = gray_code_inverse(t)
///   3. e_next = e XOR rotr_9d(E_TABLE[w], d)
///   4. d_next = (d + D_TABLE[w] + 1) % 9
struct HilbertStepResult {
    uint16_t  w;          ///< Scalar rank of this sub-hypercube
    HilbertState next;    ///< State to use for the next recursion level
};

[[nodiscard]] constexpr HilbertStepResult hilbert_state_step(
        uint16_t l, HilbertState state) noexcept {
    const uint16_t t  = rotr_9d(static_cast<uint16_t>(l ^ state.e), state.d);
    const uint16_t w  = gray_code_inverse(t);
    const uint16_t ew = HILBERT_E_TABLE_9D[w];
    const uint8_t  dw = HILBERT_D_TABLE_9D[w];

    HilbertState next;
    next.e = static_cast<uint16_t>(state.e ^ rotr_9d(ew, state.d));
    next.d = static_cast<uint8_t>((state.d + dw + 1) % HILBERT_N);
    return {w, next};
}

// ---------------------------------------------------------------------------
// 9-Dimensional Hilbert Encoder
// ---------------------------------------------------------------------------

/// Encodes a 9-dimensional spatial coordinate (array of 9 × uint32_t) into a
/// 1-dimensional Hilbert index (uint64_t).
///
/// Processes `bits_per_dim` hierarchical levels from most-significant bit down.
/// Default `bits_per_dim = 7` gives a curve of order 7 (128 grid steps per axis)
/// which fits comfortably in a 63-bit result (9 × 7 = 63 bits).
///
/// @param coords          Nine 32-bit unsigned spatial coordinates.
/// @param bits_per_dim    Number of bits of precision per axis (1..7, default 7).
/// @return                64-bit Hilbert rank (locality-preserving index).
[[nodiscard]] constexpr uint64_t hilbert_encode_9d(
        const std::array<uint32_t, 9>& coords,
        int bits_per_dim = 7) noexcept {
    uint64_t h = 0;
    HilbertState state{0, 0};   // (e=0, d=0) — canonical initial state

    for (int level = bits_per_dim - 1; level >= 0; --level) {
        // 1. Extract 9-bit cell coordinate 'l' at this bit level
        uint16_t l = 0;
        for (int dim = 0; dim < HILBERT_N; ++dim) {
            l = static_cast<uint16_t>(l | (((coords[static_cast<std::size_t>(dim)] >> level) & 1u) << dim));
        }

        // 2–4. Compute rank w and advance automaton state
        auto [w, next_state] = hilbert_state_step(l, state);
        state = next_state;

        // 5. Accumulate w into the final Hilbert index
        h = (h << HILBERT_N) | w;
    }
    return h;
}

} // namespace nikola::math
