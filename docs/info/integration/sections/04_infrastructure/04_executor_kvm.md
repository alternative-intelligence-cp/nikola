# EXECUTOR AND KVM VIRTUALIZATION

## 13.1 Ubuntu 24.04 KVM Architecture

**Purpose:** Sandboxed execution of untrusted code.

### Architecture

- **Host:** Docker container running Nikola core
- **Hypervisor:** KVM (kernel-based virtual machine)
- **Management:** libvirt C++ API
- **VMs:** Transient domains (destroyed after task completion)

### Benefits

- Complete isolation from host
- No network access (air-gapped)
- Disposable (perfect cleanup)
- Fast (hardware virtualization)

## 13.2 Mini-VM Lifecycle

### Lifecycle States

```
UNDEFINED → DEFINED → RUNNING → SHUTOFF → UNDEFINED
            ↑___________________________|
                    (Transient)
```

### Transient Domain

- Created from XML template
- Runs task
- Auto-destroyed on shutdown (no persistent config)

## 13.3 Gold Image Strategy

### Read-Only Base Image

- **Path:** `/var/lib/nikola/gold/ubuntu-24.04.qcow2`
- **Size:** ~2GB
- **Contents:** Minimal Ubuntu 24.04 Cloud image
- **State:** Immutable (never modified)

### QCOW2 Overlay (Copy-on-Write)

- **Created per task:** `/tmp/nikola/overlays/task_<ID>.qcow2`
- **Backing file:** Gold image
- **Size:** Sparse (grows as needed, max ~10GB)
- **Lifetime:** Deleted after task completion

### Creation

```bash
qemu-img create -f qcow2 \
  -b /var/lib/nikola/gold/ubuntu-24.04.qcow2 \
  -F qcow2 \
  /tmp/nikola/overlays/task_12345.qcow2
```

## 13.4 Virtio-Serial Communication

### Why Not Network?

- **Security:** VMs have no network stack → cannot attack host or internet
- **Simplicity:** Direct channel, no TCP/IP overhead
- **Performance:** Near-native speed

### Architecture

```
Host Side:                      Guest Side:
┌──────────────┐               ┌──────────────┐
│ Unix Socket  │ <───────────> │ Character    │
│ /tmp/task.sock│   virtio     │ Device       │
│              │   -serial     │ /dev/vport0p1│
└──────────────┘               └──────────────┘
      ↓                              ↓
┌──────────────┐               ┌──────────────┐
│ ZeroMQ Spine │               │ Nikola Agent │
│ Integration  │               │ (systemd)    │
└──────────────┘               └──────────────┘
```

**Protocol:** JSON Lines (newline-delimited JSON)

## 13.5 Execution Protocol

### Request (Host → Guest)

```json
{
  "cmd": "exec",
  "bin": "gcc",
  "args": ["-O3", "-o", "output", "input.c"],
  "env": {"LC_ALL": "C"},
  "cwd": "/tmp/workspace",
  "timeout": 30000
}
```

### Streaming Response (Guest → Host)

```json
{"stream": "stdout", "data": "Compiling input.c...\n"}
{"stream": "stderr", "data": ""}
```

### Completion (Guest → Host)

```json
{
  "status": "exit",
  "code": 0,
  "usage": {
    "cpu_ms": 1250,
    "mem_kb": 8192,
    "io_kb": 512
  }
}
```

## 13.6 Implementation

### VM XML Template Generator

```cpp
std::string generate_vm_xml(const std::string& task_id,
                              const std::string& overlay_path) {
    return R"(
<domain type='kvm'>
  <name>nikola_task_)" + task_id + R"(</name>
  <memory unit='KiB'>1048576</memory>
  <vcpu placement='static'>2</vcpu>
  <os>
    <type arch='x86_64'>hvm</type>
    <kernel>/var/lib/nikola/kernels/vmlinuz-6.8.0</kernel>
    <initrd>/var/lib/nikola/kernels/initrd.img-6.8.0</initrd>
    <cmdline>console=ttyS0 root=/dev/vda rw quiet</cmdline>
  </os>
  <features>
    <acpi/>
    <apic/>
  </features>
  <cpu mode='host-passthrough'/>
  <devices>
    <disk type='file' device='disk'>
      <driver name='qemu' type='qcow2' cache='unsafe'/>
      <source file=')" + overlay_path + R"('/>
      <target dev='vda' bus='virtio'/>
    </disk>
    <channel type='unix'>
      <source mode='bind' path='/tmp/nikola/sockets/)" + task_id + R"(.sock'/>
      <target type='virtio' name='org.nikola.agent.0'/>
    </channel>
    <serial type='pty'>
      <target port='0'/>
    </serial>
    <console type='pty'>
      <target type='serial' port='0'/>
    </console>
  </devices>
</domain>
)";
}
```

### Executor Class

```cpp
#include <libvirt/libvirt.h>

class KVMExecutor {
    virConnectPtr conn = nullptr;
    std::string gold_image_path = "/var/lib/nikola/gold/ubuntu-24.04.qcow2";

public:
    KVMExecutor() {
        conn = virConnectOpen("qemu:///system");
        if (!conn) {
            throw std::runtime_error("Failed to connect to KVM");
        }
    }

    ~KVMExecutor() {
        if (conn) virConnectClose(conn);
    }

    std::string execute(const CommandRequest& cmd) {
        std::string task_id = cmd.task_id();

        // 1. Create overlay
        std::string overlay_path = create_overlay(task_id);

        // 2. Generate XML
        std::string xml = generate_vm_xml(task_id, overlay_path);

        // 3. Create and start VM
        virDomainPtr dom = virDomainCreateXML(conn, xml.c_str(), VIR_DOMAIN_NONE);
        if (!dom) {
            throw std::runtime_error("Failed to create VM: " +
                                      std::string(virGetLastErrorMessage()));
        }

        // 4. Connect to virtio-serial socket
        std::string socket_path = "/tmp/nikola/sockets/" + task_id + ".sock";
        auto agent_conn = wait_for_socket(socket_path, 30000);  // 30s timeout

        // 5. Send command
        nlohmann::json request = {
            {"cmd", "exec"},
            {"bin", cmd.command()},
            {"args", std::vector<std::string>(cmd.args().begin(), cmd.args().end())},
            {"timeout", cmd.timeout_ms()}
        };

        send_json_line(agent_conn, request);

        // 6. Receive response (streaming)
        std::string stdout_data;
        std::string stderr_data;
        int exit_code = -1;

        while (true) {
            auto response = recv_json_line(agent_conn);

            if (response["stream"] == "stdout") {
                stdout_data += response["data"].get<std::string>();
            } else if (response["stream"] == "stderr") {
                stderr_data += response["data"].get<std::string>();
            } else if (response["status"] == "exit") {
                exit_code = response["code"].get<int>();
                break;
            }
        }

        // 7. Destroy VM
        virDomainDestroy(dom);
        virDomainFree(dom);

        // 8. Delete overlay
        std::filesystem::remove(overlay_path);

        // 9. Return result
        return stdout_data;
    }

private:
    std::string create_overlay(const std::string& task_id) {
        std::string overlay_path = "/tmp/nikola/overlays/task_" + task_id + ".qcow2";

        // SECURITY: Use fork/execv instead of system() to prevent shell injection
        // (Compliant with Section 17.3.1 CSVP - Code Safety Verification Protocol)
        pid_t pid = fork();

        if (pid == -1) {
            throw std::runtime_error("Failed to fork for qemu-img");
        }

        if (pid == 0) {
            // Child process: exec qemu-img
            const char* argv[] = {
                "qemu-img",
                "create",
                "-f", "qcow2",
                "-b", gold_image_path.c_str(),
                "-F", "qcow2",
                overlay_path.c_str(),
                nullptr
            };

            execvp("qemu-img", const_cast<char**>(argv));

            // If execvp returns, it failed
            std::cerr << "execvp failed: " << strerror(errno) << std::endl;
            _exit(1);
        } else {
            // Parent process: wait for child
            int status;
            waitpid(pid, &status, 0);

            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                throw std::runtime_error("Failed to create overlay image (qemu-img returned " +
                                          std::to_string(WEXITSTATUS(status)) + ")");
            }
        }

        return overlay_path;
    }

    int wait_for_socket(const std::string& path, int timeout_ms) {
        auto start = std::chrono::steady_clock::now();

        while (true) {
            if (std::filesystem::exists(path)) {
                // Socket exists, try to connect
                int sock = socket(AF_UNIX, SOCK_STREAM, 0);

                struct sockaddr_un addr;
                memset(&addr, 0, sizeof(addr));
                addr.sun_family = AF_UNIX;
                strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

                if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                    return sock;  // Success
                }

                close(sock);
            }

            // Check timeout
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (elapsed > timeout_ms) {
                throw std::runtime_error("Timeout waiting for VM socket");
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
};
```

### Guest Agent (runs inside VM)

```cpp
// File: nikola-agent.cpp (compiled and installed in gold image)

#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sys/wait.h>
#include <nlohmann/json.hpp>

void execute_command(const nlohmann::json& request) {
    std::string bin = request["bin"];
    std::vector<std::string> args = request["args"];

    // CSVP COMPLIANCE: Validate binary against permissions whitelist
    // Prevents unauthorized command execution as required by Audit 2 Item #9
    std::vector<std::string> allowed_perms = request.value("permissions", std::vector<std::string>{});

    if (std::find(allowed_perms.begin(), allowed_perms.end(), bin) == allowed_perms.end()) {
        // Binary not in whitelist - reject execution
        nlohmann::json error = {
            {"status", "error"},
            {"code", -1},
            {"message", "CSVP: Permission denied - " + bin + " not in whitelist"}
        };
        std::cout << error.dump() << std::endl;
        return;
    }

    // Create pipes for stdout/stderr
    int stdout_pipe[2], stderr_pipe[2];
    pipe(stdout_pipe);
    pipe(stderr_pipe);

    pid_t pid = fork();

    if (pid == 0) {
        // Child process
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);

        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);

        // Prepare argv
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(bin.c_str()));
        for (auto& arg : args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        argv.push_back(nullptr);

        execvp(bin.c_str(), argv.data());
        exit(1);  // execvp failed
    } else {
        // Parent process
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        // Read and stream output
        char buffer[4096];
        fd_set readfds;

        while (true) {
            FD_ZERO(&readfds);
            FD_SET(stdout_pipe[0], &readfds);
            FD_SET(stderr_pipe[0], &readfds);

            int max_fd = std::max(stdout_pipe[0], stderr_pipe[0]);

            if (select(max_fd + 1, &readfds, NULL, NULL, NULL) > 0) {
                if (FD_ISSET(stdout_pipe[0], &readfds)) {
                    ssize_t n = read(stdout_pipe[0], buffer, sizeof(buffer) - 1);
                    if (n > 0) {
                        buffer[n] = '\0';
                        nlohmann::json response = {
                            {"stream", "stdout"},
                            {"data", std::string(buffer)}
                        };
                        std::cout << response.dump() << std::endl;
                    }
                }

                if (FD_ISSET(stderr_pipe[0], &readfds)) {
                    ssize_t n = read(stderr_pipe[0], buffer, sizeof(buffer) - 1);
                    if (n > 0) {
                        buffer[n] = '\0';
                        nlohmann::json response = {
                            {"stream", "stderr"},
                            {"data", std::string(buffer)}
                        };
                        std::cout << response.dump() << std::endl;
                    }
                }
            } else {
                break;  // No more data
            }
        }

        // Wait for child
        int status;
        waitpid(pid, &status, 0);
        int exit_code = WEXITSTATUS(status);

        // Send completion
        nlohmann::json response = {
            {"status", "exit"},
            {"code", exit_code}
        };
        std::cout << response.dump() << std::endl;
    }
}

int main() {
    // Open virtio-serial port
    std::ifstream input("/dev/vport0p1");

    std::string line;
    while (std::getline(input, line)) {
        auto request = nlohmann::json::parse(line);

        if (request["cmd"] == "exec") {
            execute_command(request);
        }
    }

    return 0;
}
```

---

**Cross-References:**
- See Section 10 for ZeroMQ Spine integration with executor commands
- See Section 11 for Orchestrator integration
- See Appendix C for CommandRequest/CommandResponse Protocol Buffer schemas
