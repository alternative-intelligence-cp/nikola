// SPDX-License-Identifier: GPL-2.0
/**
 * @file bpf/nikola_sandbox.bpf.c
 * @brief Nikola KVM sandbox escape detection — eBPF tracepoint program
 *
 * Attaches to:
 *   - tracepoint/syscalls/sys_enter_execve    → catch process execution
 *   - tracepoint/syscalls/sys_enter_openat    → catch file access outside safe prefix
 *   - tracepoint/syscalls/sys_enter_socket    → catch network creation
 *   - tracepoint/syscalls/sys_enter_clone     → catch fork/clone attempts
 *   - tracepoint/syscalls/sys_enter_ptrace    → catch debugging/injection
 *
 * Events are pushed to a BPF ring buffer ("events") consumed by
 * EbpfMonitor::drain_ring_buffer() via ring_buffer__poll().
 *
 * Compile with:
 *   clang -g -O2 -target bpf -D__TARGET_ARCH_x86 \
 *         -I/usr/include/bpf \
 *         -c bpf/nikola_sandbox.bpf.c \
 *         -o bpf/nikola_sandbox.bpf.o
 *
 * v0.2.6 — Phase 1
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

// Must match EbpfEventType enum in ebpf_monitor.hpp
#define EVENT_EXECVE     0
#define EVENT_FILE_OPEN  1
#define EVENT_NETWORK    2
#define EVENT_CLONE      3
#define EVENT_PTRACE     4

// Must match BpfRawEvent struct in ebpf_monitor.hpp
struct event {
    __u32 pid;
    __u32 event_type;
    char  comm[16];
    char  filename[128];
};

// Ring buffer for events — consumed by userspace ring_buffer__poll()
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);  // 256 KB (matches EBPF_RING_BUFFER_PAGES)
} events SEC(".maps");

// ============================================================================
// Tracepoint: sys_enter_execve — process execution attempt
// ============================================================================

SEC("tracepoint/syscalls/sys_enter_execve")
int trace_execve(struct trace_event_raw_sys_enter *ctx)
{
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) return 0;

    e->pid = bpf_get_current_pid_tgid() >> 32;
    e->event_type = EVENT_EXECVE;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    // Read first argument (filename) from syscall args
    const char *fn_ptr = (const char *)ctx->args[0];
    bpf_probe_read_user_str(e->filename, sizeof(e->filename), fn_ptr);

    bpf_ringbuf_submit(e, 0);
    return 0;
}

// ============================================================================
// Tracepoint: sys_enter_openat — file access outside safe prefix
// ============================================================================

SEC("tracepoint/syscalls/sys_enter_openat")
int trace_openat(struct trace_event_raw_sys_enter *ctx)
{
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) return 0;

    e->pid = bpf_get_current_pid_tgid() >> 32;
    e->event_type = EVENT_FILE_OPEN;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    // Read filename from second argument (dirfd is first)
    const char *fn_ptr = (const char *)ctx->args[1];
    bpf_probe_read_user_str(e->filename, sizeof(e->filename), fn_ptr);

    bpf_ringbuf_submit(e, 0);
    return 0;
}

// ============================================================================
// Tracepoint: sys_enter_socket — network creation attempt
// ============================================================================

SEC("tracepoint/syscalls/sys_enter_socket")
int trace_socket(struct trace_event_raw_sys_enter *ctx)
{
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) return 0;

    e->pid = bpf_get_current_pid_tgid() >> 32;
    e->event_type = EVENT_NETWORK;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    e->filename[0] = '\0';

    bpf_ringbuf_submit(e, 0);
    return 0;
}

// ============================================================================
// Tracepoint: sys_enter_clone — fork/clone attempt
// ============================================================================

SEC("tracepoint/syscalls/sys_enter_clone")
int trace_clone(struct trace_event_raw_sys_enter *ctx)
{
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) return 0;

    e->pid = bpf_get_current_pid_tgid() >> 32;
    e->event_type = EVENT_CLONE;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    e->filename[0] = '\0';

    bpf_ringbuf_submit(e, 0);
    return 0;
}

// ============================================================================
// Tracepoint: sys_enter_ptrace — debugging/injection attempt
// ============================================================================

SEC("tracepoint/syscalls/sys_enter_ptrace")
int trace_ptrace(struct trace_event_raw_sys_enter *ctx)
{
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) return 0;

    e->pid = bpf_get_current_pid_tgid() >> 32;
    e->event_type = EVENT_PTRACE;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    e->filename[0] = '\0';

    bpf_ringbuf_submit(e, 0);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
