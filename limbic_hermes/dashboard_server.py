"""
limbic_hermes/dashboard_server.py
================================
Small Flask/FastAPI-optional HTTP server that serves the limbic state JSON
and accepts cofactor adjustments.

If no web framework is installed, falls back to http.server.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from limbic_hermes.core import LimbicSkillBridge, LimbicSystem
from limbic_hermes.storage import default_state_path


STATE_PATH = os.environ.get("LIMBIC_STATE_PATH", str(default_state_path()))


def _json_response(data: Dict[str, Any], status: int = 200) -> tuple:
    body = json.dumps(data, indent=2, default=str)
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Cache-Control", "no-store"),
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type"),
    ]
    return (status, headers, body.encode("utf-8"))


def _serve_file(path: Path, content_type: str) -> tuple:
    if not path.exists():
        return (404, [("Content-Type", "text/plain")], b"Not found")
    body = path.read_bytes()
    headers = [("Content-Type", content_type), ("Cache-Control", "no-store")]
    return (200, headers, body)


def _get_bridge() -> LimbicSkillBridge:
    return LimbicSkillBridge(STATE_PATH)


# ---------------------------------------------------------------------------
# Built-in http.server fallback handlers
# ---------------------------------------------------------------------------

class DashboardHTTPHandler:
    """Pluggable request handler for dashboard endpoints."""

    def handle(self, method: str, path: str, query: str, body: bytes) -> tuple:
        if path == "/api/state":
            return self._handle_state(method, body)
        if path == "/api/cofactors":
            return self._handle_cofactors(method, body)
        if path == "/api/snp":
            return self._handle_snp(method, body)
        if path == "/" or path == "/index.html":
            dashboard_html = Path(__file__).with_name("dashboard.html")
            return _serve_file(dashboard_html, "text/html; charset=utf-8")
        if path == "/limbic_bridge.js":
            bridge_js = Path(__file__).with_name("limbic_bridge.js")
            return _serve_file(bridge_js, "application/javascript")
        return (404, [("Content-Type", "text/plain")], b"Not found")

    def _handle_state(self, method: str, body: bytes) -> tuple:
        bridge = _get_bridge()
        if method == "POST":
            try:
                data = json.loads(body or b"{}")
            except Exception as e:
                return _json_response({"error": f"Invalid JSON: {e}"}, 400)
            # Handle state import: if the payload looks like a full state dump, load it
            if data.get("kind") == "import_state" or ("profile" in data and "vad" in data):
                try:
                    bridge.limbic = LimbicSystem.from_dict(data)
                    bridge.save()
                except Exception as e:
                    return _json_response({"error": f"Import failed: {e}"}, 400)
            else:
                kind = data.get("kind", "user_message")
                description = data.get("description", "")
                raw_valence = data.get("raw_valence", 0.0)
                raw_arousal = data.get("raw_arousal", 0.0)
                raw_dominance = data.get("raw_dominance", 0.0)
                importance = data.get("importance", 0.5)
                if kind == "circadian_set":
                    bridge.limbic.set_circadian_hour(data.get("circadian_hour", 12.0))
                    bridge.save()
                elif kind == "profile_set":
                    from limbic_hermes.profiles import full_remedy_library
                    profile_name = data.get("profile", "default")
                    bridge.limbic.profile = full_remedy_library.get(profile_name, bridge.limbic.profile)
                    bridge.save()
                else:
                    bridge.observe(
                        kind,
                        description=description,
                        raw_valence=raw_valence,
                        raw_arousal=raw_arousal,
                        raw_dominance=raw_dominance,
                        importance=importance,
                    )
        return _json_response(bridge.state())

    def _handle_cofactors(self, method: str, body: bytes) -> tuple:
        bridge = _get_bridge()
        if method == "POST":
            try:
                data = json.loads(body or b"{}")
            except Exception as e:
                return _json_response({"error": f"Invalid JSON: {e}"}, 400)
            cofactors = data.get("cofactors", {})
            if cofactors:
                bridge.limbic.apply_cofactors(cofactors)
                bridge.save()
        return _json_response({
            "cofactors": bridge.limbic.get_cofactors(),
            "state": bridge.state(),
        })

    def _handle_snp(self, method: str, body: bytes) -> tuple:
        bridge = _get_bridge()
        if method == "GET":
            from limbic_hermes.metabolic_snp import get_snp_presets, format_snp_profile_for_dashboard
            presets = get_snp_presets()
            active = bridge.limbic.snp_profile
            return _json_response({
                "active": format_snp_profile_for_dashboard(active) if active else None,
                "available": list(presets.keys()),
            })
        if method == "POST":
            try:
                data = json.loads(body or b"{}")
            except Exception as e:
                return _json_response({"error": f"Invalid JSON: {e}"}, 400)
            snp_profile_name = data.get("snp_profile")
            if snp_profile_name:
                try:
                    bridge.limbic.set_snp_profile(snp_profile_name)
                    bridge.save()
                except ValueError as e:
                    return _json_response({"error": str(e)}, 400)
            return _json_response({
                "snp_profile": (
                    {"name": bridge.limbic.snp_profile.name, "effects": bridge.limbic.snp_profile.get_variant_effects()}
                    if bridge.limbic.snp_profile else None
                ),
                "state": bridge.state(),
            })
        return _json_response({"error": "Method not allowed"}, 405)


class _BaseHTTPRequestHandler:
    """Minimal request handler shim for http.server."""

    def __init__(self, request, client_address, server):
        self.request = request
        self.client_address = client_address
        self.server = server
        self.dashboard = DashboardHTTPHandler()

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _handle(self, method: str):
        from urllib.parse import urlparse
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        status, headers, response = self.dashboard.handle(method, parsed.path, parsed.query, body)
        self.send_response(status)
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(response)


def run_http_server(port: int = 8787) -> None:
    """Run the dashboard server using stdlib http.server."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            _BaseHTTPRequestHandler(self.request, self.client_address, self.server)._handle("GET")

        def do_POST(self):
            _BaseHTTPRequestHandler(self.request, self.client_address, self.server)._handle("POST")

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Limbic dashboard server running at http://127.0.0.1:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    run_http_server(port)
