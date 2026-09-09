from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from typing import Any

from rlraft.core.supervisor import ClusterSupervisor


class DashboardServer:
    def __init__(self, supervisor: ClusterSupervisor, host: str, port: int):
        self.supervisor = supervisor
        self.host = host
        self.port = port
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        handler = self._make_handler()
        self.httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        supervisor = self.supervisor

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                if self.path == "/" or self.path == "/index.html":
                    self._serve_file("frontend/index.html", "text/html")
                elif self.path == "/style.css":
                    self._serve_file("frontend/style.css", "text/css")
                elif self.path == "/app.js":
                    self._serve_file("frontend/app.js", "application/javascript")
                elif self.path.startswith("/api/snapshot"):
                    self._send_json(supervisor.snapshot())
                else:
                    self.send_error(404)

            def _serve_file(self, filepath: str, content_type: str) -> None:
                from pathlib import Path
                path = Path(filepath)
                if not path.exists():
                    self.send_error(404)
                    return
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("content-type", f"{content_type}; charset=utf-8")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self) -> None:
                if not self.path.startswith("/api/control"):
                    self.send_error(404)
                    return
                length = int(self.headers.get("content-length", "0"))
                body = self.rfile.read(length).decode("utf-8") if length else "{}"
                payload = json.loads(body or "{}")
                result = self._handle_control(payload)
                self._send_json({"ok": True, "result": result})

            def _handle_control(self, payload: dict[str, Any]) -> str:
                action = payload.get("action")
                if action == "crash":
                    supervisor.crash_node(int(payload["node_id"]))
                    return "crash requested"
                if action == "restart":
                    supervisor.restart_node(int(payload["node_id"]))
                    return "restart requested"
                if action == "network":
                    supervisor.set_network(
                        base_latency_ms=payload.get("base_latency_ms"),
                        latency_jitter_ms=payload.get("latency_jitter_ms"),
                        packet_loss=payload.get("packet_loss"),
                    )
                    return "network updated"
                if action == "partition":
                    supervisor.partition(payload.get("groups", []))
                    return "partition applied"
                if action == "heal":
                    supervisor.heal()
                    return "network healed"
                if action == "client_command":
                    supervisor.client_command(payload.get("value", "demo-command"))
                    return "client command sent"
                return "unknown action ignored"

            def _send_json(self, obj: Any) -> None:
                data = json.dumps(obj).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler
