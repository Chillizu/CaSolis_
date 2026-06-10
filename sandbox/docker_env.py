"""Docker sandbox environment manager.

Provides container lifecycle (create/start/stop/destroy) and three sandbox APIs:
  - execute_bash(cmd) → ExecResult
  - file_edit(path, content) → ExecResult
  - finish() → ExecResult
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from docker.models.containers import Container


# ── Types (mirrors CANONICAL-SPECS.md) ──────────────────────────────

@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float


@dataclass
class EnvironmentState:
    cwd: str
    files: dict[str, int]  # path → size (bytes)
    changes: list[str]     # delta since last observation
    last_cmd: str
    last_exit: int
    last_stdout_tail: str  # last 200 chars
    last_stderr_tail: str
    step: int


@dataclass
class FileInfo:
    path: str
    size: int
    mtime: float
    is_dir: bool


@dataclass
class DockerSandboxConfig:
    image: str = "ubuntu:22.04"
    network: str = "bridge"
    memory_limit: str = "8g"
    cpu_limit: int = 4
    timeout_per_action: int = 60
    workspace_dir: str = "/home/agent_user/workspace"
    user_name: str = "agent_user"


# ── Docker Sandbox ──────────────────────────────────────────────────

class DockerSandbox:
    """Manages a single Docker container for Agent exploration."""

    def __init__(self, config: DockerSandboxConfig | None = None):
        self.config = config or DockerSandboxConfig()
        self._client = None
        self.container: "Container | None" = None
        self._prev_files: dict[str, int] = {}
        self._last_cmd = ""
        self._last_exit = 0
        self._last_stdout = ""
        self._last_stderr = ""
        self._step = 0

    @property
    def client(self):
        if self._client is None:
            import docker as _docker
            self._client = _docker.from_env()
        return self._client

    # ── Container lifecycle ────────────────────────────────────────

    def ensure_image(self) -> None:
        """Pull the base image if not present."""
        import docker as _docker
        try:
            self.client.images.get(self.config.image)
        except _docker.errors.ImageNotFound:
            self.client.images.pull(self.config.image)

    def start(self) -> None:
        """Create and start the sandbox container."""
        self.ensure_image()

        self.container = self.client.containers.run(
            self.config.image,
            detach=True,
            tty=True,
            stdin_open=True,
            network=self.config.network,
            mem_limit=self.config.memory_limit,
            nano_cpus=int(self.config.cpu_limit * 1e9),
            working_dir=self.config.workspace_dir,
            environment={
                "DEBIAN_FRONTEND": "noninteractive",
                "HOME": f"/home/{self.config.user_name}",
            },
            command="sleep infinity",
        )

        # Setup: create user + workspace (idempotent)
        self._exec_raw(
            f"id -u {self.config.user_name} 2>/dev/null || "
            f"useradd -m -s /bin/bash {self.config.user_name}; "
            f"mkdir -p {self.config.workspace_dir}; "
            f"chown -R {self.config.user_name}:{self.config.user_name} "
            f"{self.config.workspace_dir}"
        )

    def stop(self) -> None:
        """Stop and remove the container."""
        if self.container:
            try:
                self.container.stop(timeout=5)
                self.container.remove(force=True)
            except Exception:
                pass
            self.container = None

    def reset(self) -> None:
        """Reset container for a fresh episode."""
        self.stop()
        self._prev_files = {}
        self._last_cmd = ""
        self._last_exit = 0
        self._last_stdout = ""
        self._last_stderr = ""
        self._step = 0
        self.start()

    # ── Sandbox API ─────────────────────────────────────────────────

    async def execute_bash(self, cmd: str) -> ExecResult:
        start = time.monotonic()
        exit_code, stdout, stderr = self._exec(
            cmd,
            user=self.config.user_name,
            workdir=self.config.workspace_dir,
        )
        duration = (time.monotonic() - start) * 1000

        self._last_cmd = cmd
        self._last_exit = exit_code
        self._last_stdout = stdout
        self._last_stderr = stderr

        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_ms=duration,
        )

    async def file_edit(self, path: str, content: str) -> ExecResult:
        """Create or overwrite a file in the container."""
        # Escape content for shell
        import base64
        encoded = base64.b64encode(content.encode()).decode()
        full_path = f"{self.config.workspace_dir}/{path}"
        script = (
            f"mkdir -p $(dirname '{full_path}') && "
            f"echo {encoded} | base64 -d > '{full_path}'"
        )
        return await self.execute_bash(script)

    async def finish(self) -> ExecResult:
        """Mark episode as complete."""
        return ExecResult(stdout="episode finished", stderr="", exit_code=0, duration_ms=0)

    async def execute(self, action_type: str, action_content: str, path: str | None = None) -> ExecResult:
        """Dispatch to the appropriate sandbox API."""
        if action_type == "bash":
            return await self.execute_bash(action_content)
        elif action_type == "file_edit":
            assert path is not None, "file_edit requires a path"
            return await self.file_edit(path, action_content)
        elif action_type == "finish":
            return await self.finish()
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    # ── State observation ───────────────────────────────────────────

    def get_state(self) -> EnvironmentState:
        """Extract current environment state."""
        self._step += 1

        # Use shell commands for state extraction (works even without Python)
        # Get CWD
        _, cwd_out, _ = self._exec(
            "pwd",
            user=self.config.user_name,
            workdir=self.config.workspace_dir,
        )
        cwd = cwd_out.strip()

        # Get file list with sizes
        _, ls_out, _ = self._exec(
            "find . -maxdepth 5 -type f -printf '%p %s\n' 2>/dev/null | head -100",
            user=self.config.user_name,
            workdir=self.config.workspace_dir,
        )

        current_files: dict[str, int] = {}
        for line in ls_out.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                try:
                    current_files[parts[0]] = int(parts[1])
                except ValueError:
                    current_files[parts[0]] = 0

        # Compute delta
        changes: list[str] = []
        for path, size in current_files.items():
            if path not in self._prev_files:
                changes.append(f"+{path}({size}B)")
            elif self._prev_files[path] != size:
                changes.append(f"~{path}({size}B)")
        for path in self._prev_files:
            if path not in current_files:
                changes.append(f"-{path}")

        changes = changes[:5]
        self._prev_files = current_files

        return EnvironmentState(
            cwd=cwd,
            files=current_files,
            changes=changes,
            last_cmd=self._last_cmd,
            last_exit=self._last_exit,
            last_stdout_tail=self._last_stdout[-200:] if self._last_stdout else "",
            last_stderr_tail=self._last_stderr[-200:] if self._last_stderr else "",
            step=self._step,
        )

    # ── Internal helpers ────────────────────────────────────────────

    def _exec(
        self,
        cmd: str,
        user: str = "root",
        workdir: str = "/",
    ) -> tuple[int, str, str]:
        """Execute a command in the container with timeout (uses subprocess for reliability)."""
        import subprocess
        if self.container is None:
            raise RuntimeError("Container not started. Call start() first.")

        try:
            result = subprocess.run(
                ["docker", "exec", "-u", user, "-w", workdir,
                 self.container.id, "bash", "-c", cmd],
                capture_output=True,
                timeout=self.config.timeout_per_action,
                text=True,
            )
            exit_code = result.returncode
            stdout = result.stdout or ""
            stderr = result.stderr or ""
        except subprocess.TimeoutExpired:
            return 124, "", f"command timed out after {self.config.timeout_per_action}s"
        except Exception as e:
            return 1, "", str(e)

        return exit_code, stdout, stderr

    def _exec_raw(
        self,
        cmd: str,
        user: str = "root",
        workdir: str = "/",
    ) -> tuple[int, str]:
        """Execute a command without demux (for setup). Returns (exit_code, output)."""
        if self.container is None:
            raise RuntimeError("Container not started. Call start() first.")

        try:
            result = self.container.exec_run(
                cmd=["bash", "-c", cmd],
                user=user,
                workdir=workdir,
                demux=False,
            )
            exit_code = result.exit_code or 0
            output = (result.output or b"").decode("utf-8", errors="replace")
        except Exception as e:
            return 1, str(e)

        return exit_code, output


# ── Convenience factory ─────────────────────────────────────────────

def create_sandbox(**kwargs: Any) -> DockerSandbox:
    """Create a DockerSandbox with default config, optionally overridden."""
    config = DockerSandboxConfig(**kwargs)
    return DockerSandbox(config)
