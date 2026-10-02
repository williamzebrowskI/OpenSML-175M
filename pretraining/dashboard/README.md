# Five Mac Synthesizer

A restored rack animation with five instruments, ordered Mac 1 through Mac 5 from left to right. GPU utilization sets the green liquid level and the lower strip shows RAM usage. Unknown or stale GPU readings show an amber outline; measured zero shows an empty green cube. Decorative movement does not represent extra utilization. The browser layout uses five columns, or two columns and three rows on narrow screens. The dedicated 1424×280 rack panel retains display compensation with five uniformly scaled instruments.

Run on Mac 1 from the workspace root:

```bash
.venv/bin/python pretraining/scripts/rack_dashboard.py --port 8766
```

Open http://127.0.0.1:8766/rack. For the dedicated external rack screen:

```bash
.venv/bin/python pretraining/scripts/rack_panel.py
```

The native launcher compiles its WebKit wrapper on first use and identifies the rack screen before opening a borderless window. It does not change display settings. Esc or Command-Q closes it. It reuses an existing rack server; otherwise it owns and stops its child server.

The backend samples all five physical Macs concurrently, at most once per five seconds while a viewer is active. It uses the existing pinned Ethernet SSH identities and the current network plan under `pretraining/diagnostics/m5_migration_20260922/network_20261002_142104`. The topology is 1 → 2 → 3 → 5 → 4 → 1, while display order remains 1–5. GPU and CPU load are macmon frequency-scaled IOReport measurements of the whole host, not training-only utilization. macmon must be available at `/opt/homebrew/bin/macmon`; version 0.8.2 was installed on Mac 1 during restoration. Other Macs already had working sensors. Invalid measurements remain unavailable.

The preserved SMC temperature helper is best-effort and recognizes the older M3/M4 chips; unsupported sensor keys remain unavailable. SMC key decoding is adapted from the original project's rack monitor, with the API and keys informed by the open-source smc/macmon implementations. Only GPU and RAM readings drive this animation. Interface byte counters are ordinary OS network counters, not RDMA throughput.

The HTTP service binds to loopback and performs read-only hardware probes. It neither launches nor controls training. Your playground remains on port 8767 with its existing UI.

Restored from this repository's commit `91eb73b`, then adapted to the current five-Mac cluster. Frontend tests cover all five instruments, stale readings, measured zero, liquid geometry, smoothing and panel scaling; sensor tests cover validation and sanitization.
