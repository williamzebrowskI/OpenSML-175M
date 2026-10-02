#!/usr/bin/env python3
"""Launch the animation as a rack-only native panel, not a browser tab."""

import argparse
import fcntl
import hashlib
import http.client
from pathlib import Path
import plistlib
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/rack-panel"
SOURCE = ROOT / "dashboard/RackPanel.swift"
BINARY = BUILD / "Rack Panel.app/Contents/MacOS/RackPanel"


def build_panel():
    fingerprint = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    stamp = BUILD / "source.sha256"
    if BINARY.is_file() and stamp.is_file() and stamp.read_text() == fingerprint:
        return BINARY
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    plist = dict(CFBundleIdentifier="local.smlmlx.rack-panel", CFBundleName="Rack Panel",
                 CFBundleDisplayName="Rack Panel", CFBundleExecutable="RackPanel",
                 CFBundlePackageType="APPL", CFBundleVersion="1", NSHighResolutionCapable=True,
                 NSAppTransportSecurity=dict(NSAllowsLocalNetworking=True))
    (BINARY.parent.parent / "Info.plist").write_bytes(plistlib.dumps(plist))
    temporary = BINARY.with_name("RackPanel.building")
    print("[panel] Building native window (first launch or source update only)...", flush=True)
    subprocess.run(["/usr/bin/xcrun", "swiftc", "-O", "-framework", "AppKit", "-framework", "WebKit",
                    str(SOURCE), "-o", str(temporary)], check=True, timeout=120)
    temporary.replace(BINARY)
    stamp.write_text(fingerprint)
    return BINARY


def dashboard_running(port):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", "/rack")
        response = connection.getresponse()
        page = response.read(65536)
        if response.status != 200 or b"<title>Five Mac Synthesizer</title>" not in page:
            raise RuntimeError(f"Port {port} is occupied by something other than the rack dashboard")
        return True
    except ConnectionRefusedError:
        return False
    finally:
        connection.close()


def stop_owned(process, timeout=5, graceful=False):
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT if graceful else signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_panel(binary, port, display_id):
    command = [str(binary), "--port", str(port)]
    if display_id is not None:
        command += ["--display-id", str(display_id)]
    # Fail before creating a server if the rack monitor cannot be identified.
    subprocess.run(command + ["--check-display"], check=True, timeout=15)
    server = panel = None
    try:
        if dashboard_running(port):
            print("[panel] Reusing your running dashboard; it will stay running when this panel closes.", flush=True)
        else:
            print("[panel] Starting a dashboard for this session only; closing the panel stops it.", flush=True)
            server = subprocess.Popen([sys.executable, str(ROOT / "scripts/rack_dashboard.py"), "--port", str(port)],
                                      start_new_session=True)
            deadline = time.monotonic() + 10
            while not dashboard_running(port):
                if server.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("The local dashboard did not start")
                time.sleep(.2)
            if server.poll() is not None:
                raise RuntimeError("The dashboard process exited during startup")
        panel = subprocess.Popen(command, start_new_session=True)
        return panel.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        stop_owned(panel)
        # Only this launcher's children are stopped; never a shared server or training.
        stop_owned(server, timeout=25, graceful=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--display-id", type=int, help="Select among multiple matching rack panels, never the main display")
    parser.add_argument("--list-displays", action="store_true", help="Read display identities without opening a window or server")
    parser.add_argument("--check-display", action="store_true", help="Verify the rack screen without opening a window or server")
    parser.add_argument("--build-only", action="store_true", help="Compile only; no window, server, or sensor activity")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("This native panel is for macOS")
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be from 1024 to 65535")
    BUILD.mkdir(parents=True, exist_ok=True)
    with (BUILD / "launcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, "A Rack Panel launch is already running. Close that panel first.\n")
        try:
            binary = build_panel()
            if args.build_only:
                print(f"[panel] Built: {binary}")
                return 0
            if args.list_displays or args.check_display:
                command = [str(binary), "--list-displays" if args.list_displays else "--check-display"]
                if args.display_id is not None:
                    command += ["--display-id", str(args.display_id)]
                return subprocess.run(command, timeout=15).returncode
            def interrupted(_signum, _frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, interrupted)
            return run_panel(binary, args.port, args.display_id)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            parser.exit(1, f"[panel] {exc}\n")
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
