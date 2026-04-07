/**
 * @file phase137_state_persistence_test.cpp
 * @brief Phase 137 — LMDB state persistence tests.
 *
 * Tests LmdbStateStore: NikolaState, Ψ checkpoints, autobiography.
 * 12 test cases covering all acceptance criteria from v0.0.6 release plan.
 */

#include <nikola/persistence/lmdb_state_store.hpp>
#include <nikola/interior/autobiography.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// Unique temp dir per test to avoid LMDB lock conflicts
static std::string make_temp_dir(const std::string& label) {
    std::string dir = "/tmp/nikola_test_phase137_" + label + "_" +
                      std::to_string(std::rand());
    fs::create_directories(dir);
    return dir;
}

static void cleanup(const std::string& dir) {
    fs::remove_all(dir);
}

// ============================================================================
// NikolaState persistence
// ============================================================================

TEST_CASE("Phase137: State save/load round-trip preserves all fields",
          "[phase137][state]") {
    auto dir = make_temp_dir("state_rt");

    nikola::autonomy::NikolaState s;
    s.time         = 42.5f;
    s.torus_energy = 1.35f;
    s.dopamine     = 0.88f;
    s.td_error     = -0.12f;
    s.atp          = 0.65f;
    s.boredom      = 0.3f;
    s.entropy      = 2.1f;
    s.last_action  = nikola::autonomy::ActionType::EMIT_THOUGHT;
    s.tokens       = {"hello", "nikola", "curious"};

    {
        nikola::persistence::LmdbStateStore store(dir);
        store.save_state(s, 100);
    }

    {
        nikola::persistence::LmdbStateStore store(dir);
        nikola::autonomy::NikolaState loaded;
        uint64_t tick = 0;
        REQUIRE(store.load_latest_state(loaded, tick));
        CHECK(tick == 100);
        CHECK(loaded.time == s.time);
        CHECK(loaded.torus_energy == s.torus_energy);
        CHECK(loaded.dopamine == s.dopamine);
        CHECK(loaded.td_error == s.td_error);
        CHECK(loaded.atp == s.atp);
        CHECK(loaded.boredom == s.boredom);
        CHECK(loaded.entropy == s.entropy);
        CHECK(loaded.last_action == s.last_action);
        REQUIRE(loaded.tokens.size() == 3);
        CHECK(loaded.tokens[0] == "hello");
        CHECK(loaded.tokens[1] == "nikola");
        CHECK(loaded.tokens[2] == "curious");
    }

    cleanup(dir);
}

TEST_CASE("Phase137: Multiple state saves, load_latest returns highest tick",
          "[phase137][state]") {
    auto dir = make_temp_dir("state_multi");

    nikola::autonomy::NikolaState s1, s2;
    s1.dopamine = 0.2f;
    s2.dopamine = 0.9f;

    {
        nikola::persistence::LmdbStateStore store(dir);
        store.save_state(s1, 10);
        store.save_state(s2, 20);
    }

    {
        nikola::persistence::LmdbStateStore store(dir);
        nikola::autonomy::NikolaState loaded;
        uint64_t tick = 0;
        REQUIRE(store.load_latest_state(loaded, tick));
        CHECK(tick == 20);
        CHECK(loaded.dopamine == 0.9f);
        CHECK(store.state_count() == 2);
    }

    cleanup(dir);
}

TEST_CASE("Phase137: State load returns false on empty database",
          "[phase137][state]") {
    auto dir = make_temp_dir("state_empty");

    nikola::persistence::LmdbStateStore store(dir);
    nikola::autonomy::NikolaState loaded;
    uint64_t tick = 0;
    CHECK_FALSE(store.load_latest_state(loaded, tick));

    cleanup(dir);
}

// ============================================================================
// Ψ wavefunction checkpoints
// ============================================================================

TEST_CASE("Phase137: Checkpoint save/load preserves wavefunction",
          "[phase137][checkpoint]") {
    auto dir = make_temp_dir("ckpt_rt");

    // Build a small wavefunction (n=3 → 3^9 = 19,683 nodes)
    nikola::physics::WaveFunction wf;
    wf.seed_manifold(3, 3, 1, 1.0f, 42);

    const double H_before = wf.total_probability() + wf.total_kinetic_energy();
    REQUIRE(wf.num_nodes() == 19683);

    {
        nikola::persistence::LmdbStateStore store(dir);
        store.save_checkpoint(wf, 50);
    }

    {
        nikola::persistence::LmdbStateStore store(dir);
        nikola::physics::WaveFunction restored;
        nikola::persistence::detail::CheckpointHeader hdr;
        REQUIRE(store.load_latest_checkpoint(restored, hdr));
        CHECK(hdr.n_nodes == 19683);
        CHECK(hdr.grid_n == 3);

        const double H_after = restored.total_probability() +
                               restored.total_kinetic_energy();

        // Hamiltonian must match within 1e-12
        CHECK_THAT(H_after, Catch::Matchers::WithinAbs(H_before, 1e-6));
        CHECK_THAT(hdr.hamiltonian, Catch::Matchers::WithinAbs(H_before, 1e-6));

        // Spot-check: psi_real[0] must be identical
        CHECK(restored.grid().psi_real()[0] == wf.grid().psi_real()[0]);
        CHECK(restored.grid().psi_imag()[100] == wf.grid().psi_imag()[100]);
        CHECK(restored.grid().vel_real()[1000] == wf.grid().vel_real()[1000]);
    }

    cleanup(dir);
}

TEST_CASE("Phase137: Checkpoint Hamiltonian within 1e-12",
          "[phase137][checkpoint]") {
    auto dir = make_temp_dir("ckpt_H");

    nikola::physics::WaveFunction wf;
    wf.seed_manifold(3, 3, 1, 1.0f, 123);
    const double H_original = wf.total_probability() + wf.total_kinetic_energy();

    {
        nikola::persistence::LmdbStateStore store(dir);
        store.save_checkpoint(wf, 1);
    }

    {
        nikola::persistence::LmdbStateStore store(dir);
        nikola::physics::WaveFunction restored;
        nikola::persistence::detail::CheckpointHeader hdr;
        REQUIRE(store.load_latest_checkpoint(restored, hdr));

        const double H_restored = restored.total_probability() +
                                  restored.total_kinetic_energy();

        // Exact byte-for-byte copy → should be within float rounding
        CHECK_THAT(H_restored, Catch::Matchers::WithinAbs(H_original, 1e-12));
    }

    cleanup(dir);
}

TEST_CASE("Phase137: Checkpoint load returns false on empty database",
          "[phase137][checkpoint]") {
    auto dir = make_temp_dir("ckpt_empty");

    nikola::persistence::LmdbStateStore store(dir);
    nikola::physics::WaveFunction wf;
    nikola::persistence::detail::CheckpointHeader hdr;
    CHECK_FALSE(store.load_latest_checkpoint(wf, hdr));

    cleanup(dir);
}

// ============================================================================
// Autobiographical memory persistence
// ============================================================================

TEST_CASE("Phase137: Autobiography events round-trip",
          "[phase137][autobiography]") {
    auto dir = make_temp_dir("auto_evt");

    nikola::interior::AutobiographicalMemory mem;
    nikola::autonomy::NikolaState snap;
    snap.dopamine = 0.8f;
    snap.atp = 0.6f;

    // Record 10+ events (acceptance criteria)
    for (int i = 0; i < 12; ++i) {
        mem.record_event(
            "Event " + std::to_string(i),
            snap,
            nikola::interior::Affect::CURIOSITY,
            0.5 + 0.04 * i,
            {"test", "phase137"}
        );
    }

    REQUIRE(mem.event_count() == 12);

    {
        nikola::persistence::LmdbStateStore store(dir);
        store.save_autobiography(mem);
    }

    {
        nikola::interior::AutobiographicalMemory loaded;
        nikola::persistence::LmdbStateStore store(dir);
        std::size_t count = store.load_autobiography(loaded);
        CHECK(count >= 12);  // 12 events + 0 skills + 0 values
        CHECK(loaded.event_count() == 12);

        // Verify event content
        const auto& events = loaded.events();
        CHECK(events[0].description == "Event 0");
        CHECK(events[11].description == "Event 11");
        CHECK(events[5].tags.size() == 2);
        CHECK(events[5].tags[0] == "test");
    }

    cleanup(dir);
}

TEST_CASE("Phase137: Autobiography skills round-trip",
          "[phase137][autobiography]") {
    auto dir = make_temp_dir("auto_skill");

    nikola::interior::AutobiographicalMemory mem;
    mem.update_skill("reasoning", true, 10);
    mem.update_skill("reasoning", true, 20);
    mem.update_skill("reasoning", false, 30);
    mem.update_skill("language", true, 15);

    REQUIRE(mem.skill_count() == 2);

    {
        nikola::persistence::LmdbStateStore store(dir);
        store.save_autobiography(mem);
    }

    {
        nikola::interior::AutobiographicalMemory loaded;
        nikola::persistence::LmdbStateStore store(dir);
        (void)store.load_autobiography(loaded);
        CHECK(loaded.skill_count() == 2);
        CHECK(loaded.best_skill() == "reasoning");
    }

    cleanup(dir);
}

TEST_CASE("Phase137: Autobiography values round-trip",
          "[phase137][autobiography]") {
    auto dir = make_temp_dir("auto_val");

    nikola::interior::AutobiographicalMemory mem;
    mem.update_value("truth", 1.0);
    mem.update_value("curiosity", 2.0);
    mem.update_value("kindness", 0.5);

    REQUIRE(mem.value_count() == 3);

    {
        nikola::persistence::LmdbStateStore store(dir);
        store.save_autobiography(mem);
    }

    {
        nikola::interior::AutobiographicalMemory loaded;
        nikola::persistence::LmdbStateStore store(dir);
        (void)store.load_autobiography(loaded);
        CHECK(loaded.value_count() == 3);

        auto values = loaded.get_values();
        CHECK(values.count("truth") == 1);
        CHECK(values.count("curiosity") == 1);
        CHECK(values.count("kindness") == 1);
    }

    cleanup(dir);
}

TEST_CASE("Phase137: Cross-session persistence",
          "[phase137][integration]") {
    auto dir = make_temp_dir("cross_session");

    // Session 1: save state + checkpoint + autobiography
    {
        nikola::persistence::LmdbStateStore store(dir);

        nikola::autonomy::NikolaState s;
        s.dopamine = 0.75f;
        s.atp = 0.5f;
        s.tokens = {"thinking", "about", "memory"};
        store.save_state(s, 42);

        nikola::physics::WaveFunction wf;
        wf.seed_manifold(3, 3, 1, 1.0f, 99);
        store.save_checkpoint(wf, 42);

        nikola::interior::AutobiographicalMemory mem;
        mem.record_event("First session", s, nikola::interior::Affect::CURIOSITY, 0.8);
        mem.update_skill("persistence", true, 42);
        mem.update_value("continuity", 1.5);
        store.save_autobiography(mem);
    }

    // Session 2: restore everything
    {
        nikola::persistence::LmdbStateStore store(dir);

        // State
        nikola::autonomy::NikolaState loaded_state;
        uint64_t tick = 0;
        REQUIRE(store.load_latest_state(loaded_state, tick));
        CHECK(tick == 42);
        CHECK(loaded_state.dopamine == 0.75f);
        CHECK(loaded_state.tokens.size() == 3);

        // Checkpoint
        nikola::physics::WaveFunction restored_wf;
        nikola::persistence::detail::CheckpointHeader hdr;
        REQUIRE(store.load_latest_checkpoint(restored_wf, hdr));
        CHECK(hdr.n_nodes == 19683);

        // Autobiography
        nikola::interior::AutobiographicalMemory loaded_mem;
        std::size_t count = store.load_autobiography(loaded_mem);
        CHECK(count >= 3);  // 1 event + 1 skill + 1 value
        CHECK(loaded_mem.event_count() == 1);
        CHECK(loaded_mem.skill_count() == 1);
        CHECK(loaded_mem.value_count() == 1);
    }

    cleanup(dir);
}

TEST_CASE("Phase137: State dump produces readable output",
          "[phase137][dump]") {
    auto dir = make_temp_dir("dump");

    nikola::autonomy::NikolaState s;
    s.dopamine = 0.5f;
    s.atp = 0.8f;

    {
        nikola::persistence::LmdbStateStore store(dir);
        store.save_state(s, 100);

        nikola::physics::WaveFunction wf;
        wf.seed_manifold(3, 3, 1, 1.0f, 0);
        store.save_checkpoint(wf, 100);

        std::string dump = store.dump_latest();
        CHECK(dump.find("tick 100") != std::string::npos);
        CHECK(dump.find("dopamine") != std::string::npos);
        CHECK(dump.find("Checkpoints: 1") != std::string::npos);
    }

    cleanup(dir);
}
