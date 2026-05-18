#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <nikola/infrastructure/rcis_protocol.hpp>

#include <rcis.pb.h>
#include <zmq.hpp>

using namespace nikola::infrastructure::rcis;

TEST_CASE("v0.3.5 §1 NeuralSpike validates required headers", "[v035][rcis]") {
    ::nikola::spine::NeuralSpike s;
    CHECK_FALSE(validate_neural_spike(s));

    s.set_request_id("r-1");
    s.set_timestamp_ns(123);
    CHECK(validate_neural_spike(s));
}

TEST_CASE("v0.3.5 §2 RCISRequest validates per-type payload rules", "[v035][rcis]") {
    auto ping = make_ping_request("ping-1");
    CHECK(validate_request(ping));

    ::nikola::spine::RCISRequest stim;
    stim.set_request_id("s-1");
    stim.set_timestamp_ns(1);
    stim.set_type(::nikola::spine::RCISRequest::INJECT_STIMULUS);
    CHECK_FALSE(validate_request(stim));
    stim.set_stimulus_text("hello");
    CHECK(validate_request(stim));

    ::nikola::spine::RCISRequest fw;
    fw.set_request_id("f-1");
    fw.set_timestamp_ns(1);
    fw.set_type(::nikola::spine::RCISRequest::FORWARD_SPIKE);
    CHECK_FALSE(validate_request(fw));
    auto* spike = fw.mutable_spike();
    spike->set_request_id("spk");
    spike->set_timestamp_ns(5);
    CHECK(validate_request(fw));
}

TEST_CASE("v0.3.5 §3 RCISResponse validates status and embedded spike", "[v035][rcis]") {
    auto ok = make_ok_response("abc", "done");
    CHECK(validate_response(ok));

    ::nikola::spine::RCISResponse bad;
    bad.set_request_id("abc");
    bad.set_timestamp_ns(1);
    CHECK_FALSE(validate_response(bad)); // status unspecified

    bad.set_status(::nikola::spine::RCISResponse::OK);
    auto* sp = bad.mutable_spike();
    sp->set_request_id("");
    sp->set_timestamp_ns(10);
    CHECK_FALSE(validate_response(bad));
}

TEST_CASE("v0.3.5 §4 protobuf wire roundtrip for request", "[v035][rcis]") {
    auto req = make_ping_request("wire-1");
    req.set_stimulus_text(std::string(RCIS_TOPIC_REQUEST));
    req.mutable_spike()->set_request_id("sp-1");
    req.mutable_spike()->set_timestamp_ns(99);

    const std::string wire = serialize(req);
    REQUIRE_FALSE(wire.empty());

    ::nikola::spine::RCISRequest out;
    REQUIRE(deserialize(wire, out));
    CHECK(out.request_id() == "wire-1");
    CHECK(out.type() == ::nikola::spine::RCISRequest::PING);
    CHECK(out.has_spike());
}

TEST_CASE("v0.3.5 §5 deserialize rejects invalid wire", "[v035][rcis]") {
    ::nikola::spine::RCISRequest out;
    CHECK_FALSE(deserialize(std::string("not-a-proto"), out));
}

TEST_CASE("v0.3.5 §6 ZMQ transport send/recv for RCISRequest", "[v035][rcis][zmq]") {
    zmq::context_t ctx(1);
    zmq::socket_t rx(ctx, zmq::socket_type::pair);
    zmq::socket_t tx(ctx, zmq::socket_type::pair);

    rx.set(zmq::sockopt::rcvtimeo, 1000);
    tx.set(zmq::sockopt::sndtimeo, 1000);

    const std::string ep = "inproc://v035-req";
    rx.bind(ep);
    tx.connect(ep);

    auto req = make_ping_request("z-1");
    send_proto(tx, req);

    ::nikola::spine::RCISRequest got;
    REQUIRE(recv_proto(rx, got));
    CHECK(got.request_id() == "z-1");
    CHECK(got.type() == ::nikola::spine::RCISRequest::PING);
}

TEST_CASE("v0.3.5 §7 ZMQ transport send/recv for RCISResponse", "[v035][rcis][zmq]") {
    zmq::context_t ctx(1);
    zmq::socket_t rx(ctx, zmq::socket_type::pair);
    zmq::socket_t tx(ctx, zmq::socket_type::pair);

    rx.set(zmq::sockopt::rcvtimeo, 1000);
    tx.set(zmq::sockopt::sndtimeo, 1000);

    const std::string ep = "inproc://v035-resp";
    rx.bind(ep);
    tx.connect(ep);

    auto resp = make_ok_response("r-77", "ok");
    send_proto(tx, resp);

    ::nikola::spine::RCISResponse got;
    REQUIRE(recv_proto(rx, got));
    CHECK(got.request_id() == "r-77");
    CHECK(got.status() == ::nikola::spine::RCISResponse::OK);
}

TEST_CASE("v0.3.5 §8 Curve client helper rejects empty server key", "[v035][rcis][curve]") {
    zmq::context_t ctx(1);
    zmq::socket_t sock(ctx, zmq::socket_type::pair);
    const auto kp = nikola::security::generate_ironhouse_keypair();

    CHECK_THROWS_AS(nikola::infrastructure::rcis::configure_curve_client(sock, kp, ""), std::invalid_argument);
}

TEST_CASE("v0.3.5 §9 RCIS proto enums present expected values", "[v035][rcis]") {
    CHECK(static_cast<int>(::nikola::spine::RCISRequest::PING) == 1);
    CHECK(static_cast<int>(::nikola::spine::RCISRequest::FORWARD_SPIKE) == 4);
    CHECK(static_cast<int>(::nikola::spine::RCISResponse::OK) == 1);
    CHECK(static_cast<int>(::nikola::spine::RCISResponse::ERROR) == 3);
}

TEST_CASE("v0.3.5 §10 trace context map survives wire roundtrip", "[v035][rcis]") {
    ::nikola::spine::NeuralSpike s;
    s.set_request_id("t-1");
    s.set_timestamp_ns(12345);
    (*s.mutable_trace_context())["traceparent"] = "00-abc-def-01";
    (*s.mutable_trace_context())["baggage"] = "k=v";

    const std::string wire = serialize(s);
    ::nikola::spine::NeuralSpike out;
    REQUIRE(deserialize(wire, out));

    REQUIRE(out.trace_context().find("traceparent") != out.trace_context().end());
    REQUIRE(out.trace_context().find("baggage") != out.trace_context().end());
    CHECK(out.trace_context().at("baggage") == "k=v");
}
