#!/usr/bin/env python3
"""Read-only rack display. No MLX, sudo, training controls, or network changes."""

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
from scripts.rack_http import make_handler
from inspect_five_mac_ssh import PEERS, ssh_options
from scripts.rack_sensors import finite

DEFAULT_RUN = ROOT / "train/checkpoints/dolma_cosmo_152m_prose_continuation_v1"
NETWORK = ROOT / "diagnostics/m5_migration_20260922/network_20261002_142104"

# Executed over stdin, never installed on a worker. Only aggregate metrics leave it.
PROBE = r'''
import json, re, shutil, subprocess
def read(argv):
    return subprocess.check_output(argv, text=True, timeout=3).strip()
out = {}
for key, command in (
    ('chip', ['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string']),
    ('memory_bytes', ['/usr/sbin/sysctl', '-n', 'hw.memsize']),
):
    try:
        value = read(command)
        out[key] = int(value) if key == 'memory_bytes' else value[:80]
    except (OSError, ValueError, subprocess.SubprocessError):
        out[key] = None
try:
    processes = read(['/bin/ps', '-axo', 'comm=,args=']).splitlines()
    workers = []
    for process in processes:
        if not re.match(r'\S*[Pp]ython\S*\s', process):
            continue
        if '-m sml_v1.pretrain' in process or re.search(r'(?:/|\s|\.)(?:pretrain_jaccl|sft_cosmo|train_sft|train_(?:fresh_responses|full_response_pairs|grounded_transfer|natural_variation|nl_replay|alpaca_replay|preference_pilot|single_pass_sft|sft_varied))(?:\.py|\s|$)', process):
            workers.append('pretrain' if 'pretrain_jaccl' in process or '-m sml_v1.pretrain' in process else 'sft')
    out['workers'] = len(workers)
    out['workloads'] = sorted(set(workers))
except (OSError, subprocess.SubprocessError):
    out['workers'] = None
    out['workloads'] = []
try:
    match = re.search(r'used\s*=\s*([\d.]+)([MG])', read(['/usr/sbin/sysctl', '-n', 'vm.swapusage']))
    out['swap_gib'] = float(match[1]) / (1024 if match[2] == 'M' else 1) if match else None
except (OSError, subprocess.SubprocessError):
    out['swap_gib'] = None
try:
    disk = shutil.disk_usage('/')
    out['disk_free_gib'] = disk.free / 1024**3
except OSError:
    out['disk_free_gib'] = None
if 'sample_hardware' in globals():
    out['hardware'] = sample_hardware(out.get('chip'))
print(json.dumps(out, allow_nan=False))
'''
WIRE_PROBE = (ROOT / 'scripts/rack_sensors.py').read_text() + '\n' + PROBE


def clean_hardware(raw):
    """Only typed aggregate counters leave the probe boundary."""
    if not isinstance(raw, dict):
        return {}
    clean = {}
    for key in ('gpu_load', 'ecpu_load', 'pcpu_load'):
        clean[key] = finite(raw.get(key), 0, 100)
    for key in ('gpu_temp', 'cpu_temp'):
        clean[key] = finite(raw.get(key), 10, 125)
    for key in ('gpu_mhz', 'ecpu_mhz', 'pcpu_mhz', 'gpu_sensor_count', 'cpu_sensor_count'):
        clean[key] = finite(raw.get(key), 0, 20000)
    for key in ('gpu_power', 'cpu_power', 'sys_power', 'ane_power'):
        clean[key] = finite(raw.get(key), 0, 1000)
    for key in ('ram_usage', 'ram_total', 'swap_usage', 'network_time'):
        clean[key] = finite(raw.get(key))
    if clean['ram_total'] is not None and clean['ram_usage'] is not None and clean['ram_usage'] > clean['ram_total']:
        clean['ram_usage'] = None
    fans = raw.get('fan_rpm')
    clean['fan_rpm'] = [v for v in fans[:2] if finite(v, 0, 20000) is not None] if isinstance(fans, list) else []
    clean['load_kind'] = 'frequency_scaled'
    clean['network_counters'] = {}
    counters = raw.get('network_counters')
    if isinstance(counters, dict):
        for name, values in list(counters.items())[:32]:
            if not name.startswith('en') or not name[2:].isdigit() or not isinstance(values, dict):
                continue
            if all(finite(values.get(k), 0, 2**64) is not None for k in ('rx', 'tx')):
                clean['network_counters'][name] = {k: values[k] for k in ('rx', 'tx')}
    return clean


def network_rates(previous, current, ports):
    """Counter deltas over a host-local monotonic interval; not an RDMA estimate."""
    result = {}
    before, after = previous.get('network_time'), current.get('network_time')
    dt = after - before if finite(before) is not None and finite(after) is not None else 0
    for label, interfaces in (('tb', ports), ('lan', ['en0']), ('wifi', ['en1'])):
        for direction in ('rx', 'tx'):
            total = 0
            valid = bool(interfaces) and 0 < dt <= 45
            for interface in interfaces:
                old = previous.get('network_counters', {}).get(interface, {}).get(direction)
                new = current.get('network_counters', {}).get(interface, {}).get(direction)
                if old is None or new is None or new < old:
                    valid = False
                    break
                total += new - old
            result[f'{label}_{direction}_Bps'] = total / dt if valid else None
    return result


def probe_command(rank):
    if rank == 0:
        return [sys.executable, "-"]
    if rank not in range(1, 5):
        raise ValueError("Unknown rank")
    peer = PEERS[rank-1]
    return ssh_options(NETWORK / f"mac{rank+1}_known_hosts") + [
        f"williamzebrowski@{peer[1]}",
        shlex.quote(str(ROOT.parent / ".venv/bin/python")) + " -"]



def probe_node(rank):
    try:
        result = subprocess.run(probe_command(rank), input=WIRE_PROBE, text=True,
                                capture_output=True, timeout=20)
        if result.returncode:
            # Do not leak remote process arguments or arbitrary SSH error contents.
            reason = "Host key check failed" if "HOST IDENTIFICATION" in result.stderr or "Host key verification" in result.stderr else "Probe unavailable"
            return dict(rank=rank, reachable=None, checked_at=time.time(), error=reason)
        data = json.loads(result.stdout)
        if not isinstance(data, dict) or len(result.stdout) > 8192:
            raise ValueError("Invalid probe response")
        clean = {key: data.get(key) for key in ("chip", "memory_bytes", "workers", "workloads", "swap_gib", "disk_free_gib")}
        for key in ("memory_bytes", "workers", "swap_gib", "disk_free_gib"):
            value = clean[key]
            if value is not None and (not isinstance(value, (int, float)) or not 0 <= value < 1e15):
                raise ValueError("Invalid probe metric")
        if not isinstance(clean["workloads"], list) or any(v not in ("sft", "pretrain") for v in clean["workloads"]):
            raise ValueError("Invalid workloads")
        clean['hardware'] = clean_hardware(data.get('hardware'))
        clean.update(rank=rank, reachable=True, checked_at=time.time(), error=None)
        return clean
    except subprocess.TimeoutExpired:
        return dict(rank=rank, reachable=None, checked_at=time.time(), error="Connection timed out")
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return dict(rank=rank, reachable=None, checked_at=time.time(), error="Probe unavailable")


class Health:
    """One bounded five-host sampling round, shared by viewers; idle without viewers."""
    def __init__(self, enabled=True, interval=5, probe=probe_node):
        self.enabled, self.interval, self.probe = enabled, interval, probe
        self.nodes = [dict(rank=r, reachable=None, checked_at=None,
                           error="Checking" if enabled else "Checks disabled") for r in range(5)]
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.last_viewer = 0
        self.thread = None
        self.previous = [{} for _ in range(5)]
        self.history = [[] for _ in range(5)]
        self.ports = [[] for _ in range(5)]
        for link in ring_topology()['links']:
            self.ports[link['a']].append(link['port_a'])
            self.ports[link['b']].append(link['port_b'])

    def snapshot(self):
        with self.lock:
            self.last_viewer = time.monotonic()
            if self.enabled and self.thread is None:
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
            self.wake.set()
            return dict(enabled=self.enabled, interval_seconds=self.interval,
                        nodes=copy.deepcopy(self.nodes))

    def _sample(self, rank):
        try:
            record = self.probe(rank)
        except Exception:
            record = dict(rank=rank, reachable=None, checked_at=time.time(), error="Probe unavailable")
        hardware = record.get('hardware', {})
        hardware.update(network_rates(self.previous[rank], hardware, self.ports[rank]))
        self.previous[rank] = copy.deepcopy(hardware)
        self.history[rank].append(dict(time=record['checked_at'], gpu=hardware.get('gpu_load')))
        self.history[rank] = self.history[rank][-60:]
        hardware['gpu_history'] = self.history[rank]
        record['hardware'] = hardware
        with self.lock:
            self.nodes[rank] = record

    def _run(self):
        with ThreadPoolExecutor(max_workers=5, thread_name_prefix='rack-probe') as pool:
            self._rounds(pool)

    def _rounds(self, pool):
        next_poll = 0
        while not self.stop.is_set():
            now = time.monotonic()
            if now >= next_poll and now - self.last_viewer < 45:
                # No overlapping polls on a host; initialization latency is not multiplied by five.
                list(pool.map(self._sample, range(5)))
                next_poll = time.monotonic() + self.interval
            self.wake.wait(1)
            self.wake.clear()

    def close(self):
        self.stop.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=22)


def ring_topology():
    try:
        data = json.loads((NETWORK / "plan.json").read_text())["topology"]
        edges, order = data["edges"], data["physical_order"]
        if sorted(order) != list(range(5)) or len(edges) != 5:
            raise ValueError("Invalid ring inventory")
        for a, pa, b, pb in edges:
            if a not in order or b not in order or a == b or not all(isinstance(p, str) and p.startswith("en") for p in (pa, pb)):
                raise ValueError("Invalid edge")
        return dict(kind="Configured ring", order=order,
                    links=[dict(a=a, port_a=pa, b=b, port_b=pb) for a, pa, b, pb in edges],
                    verified_live=False)
    except (OSError, ValueError, TypeError, KeyError):
        return dict(kind="Topology unavailable", order=[0, 1, 2, 4, 3], links=[], verified_live=False)


class RackTelemetry:
    def __init__(self, run, health):
        # Keep the old run argument compatible, but the visualizer needs no training logs.
        self.health = health

    def snapshot(self):
        return {"rack": {"health": self.health.snapshot()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-host-checks", action="store_true", help="Disable local and SSH probes; instruments have no live readings")
    parser.add_argument("--check-once", action="store_true", help="Print one read-only host snapshot and exit; no server")
    args = parser.parse_args()
    if args.check_once:
        for rank in range(5):
            print(json.dumps(probe_node(rank)), flush=True)
        return
    if not 1024 <= args.port <= 65535:
        parser.error("Use an unprivileged port from 1024 to 65535")
    health = Health(enabled=not args.no_host_checks)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(RackTelemetry(args.run_dir, health), rack=True))
    except OSError as exc:
        parser.exit(1, f"Cannot start rack dashboard: {exc}. Try --port 8766\n")
    print(f"FIVE MAC SYNTHESIZER\nhttp://127.0.0.1:{args.port}/rack", flush=True)
    print("Animation only. Read-only sensors; no training logs or controls. Ctrl+C stops this display service.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        health.close()


if __name__ == "__main__":
    main()
