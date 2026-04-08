# Peer Discovery & Handshake Protocol

## Overview

Nikola uses a **bootstrap-node-list** discovery model for initial peer contact,
with **CurveZMQ (Ironhouse)** providing authenticated encryption for all peer
communication. After discovery, peers exchange `PeerAnnouncement` protobufs
(defined in `proto/social/irsp.proto`) and register each other in their local
`PeerRegistry`.

## Discovery Model

### Bootstrap Node List

Each Nikola instance ships with (or is configured with) a list of known peer
endpoints. On startup, it iterates through the list and attempts a handshake
with each. This is deliberately simple — DHT-based discovery is a future
enhancement once the network grows beyond manual configuration.

```
bootstrap_nodes:
  - tcp://nikola-alpha.example.com:9876
  - tcp://nikola-beta.example.com:9876
  - tcp://10.0.0.5:9876
```

### Discovery Flow

```
1. Read bootstrap node list from config
2. For each endpoint:
   a. Generate ephemeral CurveZMQ keypair (or use persistent Tier 1 key)
   b. Attempt handshake (see below)
   c. On success → add to PeerRegistry
   d. On failure → log, skip, try next
3. Periodically re-scan for new/recovered peers
```

## Handshake Protocol

### Transport

- **Socket pattern:** REQ/REP over CurveZMQ
- **Encryption:** Curve25519 (via ZeroMQ Ironhouse pattern)
- **Authentication:** Mutual — both sides must present valid CurveZMQ keypairs

### Key Hierarchy (from Ironhouse)

| Tier | Purpose | Rotation |
|------|---------|----------|
| 0 — SPINE_BROKER | Root of trust | Emergency only |
| 1 — CORE_COMPONENT | Peer identity | 30 days |
| 2 — EPHEMERAL_AGENT | Tools, CLI | 24 hours |

Peer-to-peer handshakes use **Tier 1 (CORE_COMPONENT)** keys.

### Message Format

Both messages use the `PeerAnnouncement` protobuf (from `irsp.proto`):

```protobuf
message PeerAnnouncement {
    string peer_id          = 1;  // Self-declared unique ID
    string public_key_z85   = 2;  // CurveZMQ public key (Z85 encoded)
    string zmq_endpoint     = 3;  // Reachable endpoint
    repeated string capabilities = 4;  // Offered services
    int64 timestamp         = 5;  // Unix millis
}
```

### Sequence Diagram

```
  Initiator (REQ client)              Responder (REP server)
  ──────────────────────              ──────────────────────
          │                                     │
          │  1. zmq_connect() with CurveZMQ     │
          │  ─────────────────────────────────►  │  (bind already active)
          │                                     │
          │  2. Send PeerAnnouncement            │
          │  ─────────────────────────────────►  │
          │                                     │  3. Deserialize, validate
          │                                     │  4. Add initiator to PeerRegistry
          │  5. Receive PeerAnnouncement         │
          │  ◄─────────────────────────────────  │
          │                                     │
          │  6. Deserialize, validate            │
          │  7. Add responder to PeerRegistry    │
          │                                     │
        [Both registries now contain each other]
```

### Error Handling

| Failure Mode | Behavior |
|---|---|
| CurveZMQ handshake failure | ZMQ returns EAGAIN; `HandshakeResult.error` set |
| Timeout (default: 5000ms) | Socket `rcvtimeo` / `sndtimeo` fire; error returned |
| Bad protobuf | `ParseFromString` fails; "bad announcement" error |
| Duplicate peer | `PeerRegistry::add_peer` deduplicates by peer_id; key updated |

### Linger Policy

All sockets set `ZMQ_LINGER = 0` to ensure clean shutdown on failure paths.
No messages are buffered after socket close.

## API Reference

### Headers

- `include/nikola/social/peer_handshake.hpp`

### Functions

```cpp
// Serialize a PeerAnnouncement to protobuf wire format
std::string serialize_announcement(
    const std::string& peer_id,
    const std::string& pub_key_z85,
    const std::string& endpoint,
    const std::vector<std::string>& capabilities = {});

// Deserialize a PeerAnnouncement from protobuf wire format
bool deserialize_announcement(
    const std::string& data,
    std::string& peer_id, std::string& pub_key_z85,
    std::string& endpoint, std::vector<std::string>& capabilities);

// Client-side handshake (connect → send → recv → register)
HandshakeResult initiate_handshake(
    zmq::context_t& ctx,
    const IronhouseKeypair& own_kp,
    std::string_view remote_pub_z85,
    const std::string& remote_endpoint,
    PeerRegistry& registry,
    const std::string& self_id,
    int timeout_ms = 5000);

// Server-side handshake (bind → recv → send → register)
HandshakeResult accept_handshake(
    zmq::context_t& ctx,
    const IronhouseKeypair& own_kp,
    const std::string& bind_endpoint,
    PeerRegistry& registry,
    const std::string& self_id,
    int timeout_ms = 5000);
```

## Future Work

1. **DHT Discovery** — Replace bootstrap list with Kademlia-style DHT for
   decentralized peer finding (planned for multi-agent scaling phase)
2. **Capability Matching** — Use the `capabilities` field to route service
   requests to the most capable peer
3. **Key Rotation Handshake** — Extend protocol with re-keying messages
   using the Ironhouse tier system
4. **NAT Traversal** — ZMQ doesn't handle NAT natively; consider a
   relay/TURN-style intermediary for public network deployment
