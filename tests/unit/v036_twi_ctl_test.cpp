#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/cli/twi_ctl.hpp>

#include <thread>
#include <zmq.hpp>

using namespace nikola::cli::twi_ctl;

TEST_CASE("v0.3.6 slice2 §1 parse query command", "[v036][twi-ctl]") {
    const char* argv[] = {
        "twi-ctl", "query", "Explain", "entanglement", "--threshold", "0.8", "--steps", "42", "--json"
    };
    auto p = parse_args(static_cast<int>(std::size(argv)), const_cast<char**>(argv));

    REQUIRE(p.valid);
    CHECK(p.command == Command::QUERY);
    CHECK(p.payload == "Explain entanglement");
    CHECK(p.threshold == Catch::Approx(0.8f));
    CHECK(p.steps == 42);
    CHECK(p.json);
}

TEST_CASE("v0.3.6 slice2 §2 parse ingest with file", "[v036][twi-ctl]") {
    const char* argv[] = {
        "twi-ctl", "ingest", "--file", "/tmp/doc.txt", "--type", "text"
    };
    auto p = parse_args(static_cast<int>(std::size(argv)), const_cast<char**>(argv));

    REQUIRE(p.valid);
    CHECK(p.command == Command::INGEST);
    CHECK(p.file_path == "/tmp/doc.txt");
    CHECK(p.ingest_type == "text");
}

TEST_CASE("v0.3.6 slice3 §2b parse endpoint and timeout options", "[v036][twi-ctl]") {
    const char* argv[] = {
        "twi-ctl", "status", "--endpoint", "tcp://127.0.0.1:6006", "--timeout-ms", "1234"
    };
    auto p = parse_args(static_cast<int>(std::size(argv)), const_cast<char**>(argv));

    REQUIRE(p.valid);
    CHECK(p.command == Command::STATUS);
    CHECK(p.endpoint == "tcp://127.0.0.1:6006");
    CHECK(p.timeout_ms == 1234);
}

TEST_CASE("v0.3.6 slice2 §3 invalid threshold is rejected", "[v036][twi-ctl]") {
    const char* argv[] = {"twi-ctl", "query", "hello", "--threshold", "1.5"};
    auto p = parse_args(static_cast<int>(std::size(argv)), const_cast<char**>(argv));

    CHECK_FALSE(p.valid);
    CHECK_FALSE(p.error.empty());
}

TEST_CASE("v0.3.6 slice2 §4 unknown command is rejected", "[v036][twi-ctl]") {
    const char* argv[] = {"twi-ctl", "dance"};
    auto p = parse_args(static_cast<int>(std::size(argv)), const_cast<char**>(argv));

    CHECK_FALSE(p.valid);
    CHECK(p.command == Command::UNKNOWN);
}

TEST_CASE("v0.3.6 slice2 §5 build query request maps to INJECT_STIMULUS", "[v036][twi-ctl]") {
    const char* argv[] = {"twi-ctl", "query", "What is resonance?"};
    auto p = parse_args(static_cast<int>(std::size(argv)), const_cast<char**>(argv));
    REQUIRE(p.valid);

    auto req = build_rcis_request(p);
    REQUIRE(req.has_value());
    CHECK(req->type() == ::nikola::spine::RCISRequest::INJECT_STIMULUS);
    CHECK(req->stimulus_text() == "What is resonance?");
    CHECK_FALSE(req->request_id().empty());
}

TEST_CASE("v0.3.6 slice2 §6 build status request maps to FETCH_STATE", "[v036][twi-ctl]") {
    const char* argv[] = {"twi-ctl", "status"};
    auto p = parse_args(static_cast<int>(std::size(argv)), const_cast<char**>(argv));
    REQUIRE(p.valid);

    auto req = build_rcis_request(p);
    REQUIRE(req.has_value());
    CHECK(req->type() == ::nikola::spine::RCISRequest::FETCH_STATE);
}

TEST_CASE("v0.3.6 slice2 §7 init command does not build RCIS request", "[v036][twi-ctl]") {
    const char* argv[] = {"twi-ctl", "init"};
    auto p = parse_args(static_cast<int>(std::size(argv)), const_cast<char**>(argv));
    REQUIRE(p.valid);

    auto req = build_rcis_request(p);
    CHECK_FALSE(req.has_value());
}

TEST_CASE("v0.3.6 slice3 §8 RCIS roundtrip helper exchanges request/response", "[v036][twi-ctl][zmq]") {
    const std::string ep = "tcp://127.0.0.1:62036";

    zmq::context_t server_ctx(1);
    zmq::socket_t server(server_ctx, zmq::socket_type::rep);
    server.set(zmq::sockopt::rcvtimeo, 1000);
    server.set(zmq::sockopt::sndtimeo, 1000);
    server.bind(ep);

    bool server_ok = false;
    std::thread responder([&]() {
        ::nikola::spine::RCISRequest req;
        if (!nikola::infrastructure::rcis::recv_proto(server, req)) {
            return;
        }

        ::nikola::spine::RCISResponse resp;
        resp.set_request_id(req.request_id());
        resp.set_timestamp_ns(nikola::infrastructure::rcis::now_ns());
        resp.set_status(::nikola::spine::RCISResponse::OK);
        resp.set_message("pong");
        nikola::infrastructure::rcis::send_proto(server, resp);
        server_ok = true;
    });

    const char* argv[] = {"twi-ctl", "ping", "--endpoint", "tcp://127.0.0.1:62036", "--timeout-ms", "500"};
    auto p = parse_args(static_cast<int>(std::size(argv)), const_cast<char**>(argv));
    REQUIRE(p.valid);

    auto req = build_rcis_request(p);
    REQUIRE(req.has_value());

    ::nikola::spine::RCISResponse out;
    std::string err;
    CHECK(rcis_roundtrip(*req, out, p.endpoint, p.timeout_ms, &err));
    CHECK(out.status() == ::nikola::spine::RCISResponse::OK);
    CHECK(out.message() == "pong");

    responder.join();
    CHECK(server_ok);
}
