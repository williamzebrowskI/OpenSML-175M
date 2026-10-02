"""Bounded, read-only hardware probes; also sent over SSH stdin, without installation.

GPU load is macmon's frequency-scaled IOReport load, not per-core occupancy.
SMC protocol and sensor-key references are documented in dashboard/README.md.
"""

import ctypes as C
import json
import math
import re
import struct
import subprocess
import time


def finite(value, low=0, high=1e15):
    return value if type(value) in (int, float) and math.isfinite(value) and low <= value <= high else None


def parse_macmon(data):
    if not isinstance(data, dict):
        return {}
    out = {}
    for key, source in (("gpu", "gpu_usage"), ("ecpu", "ecpu_usage"), ("pcpu", "pcpu_usage")):
        pair = data.get(source)
        if isinstance(pair, list) and len(pair) == 2:
            ratio = finite(pair[1], 0, 1)
            out[key + "_load"] = ratio * 100 if ratio is not None else None
            out[key + "_mhz"] = finite(pair[0], 0, 20000)
    for source in ("gpu_power", "cpu_power", "sys_power", "ane_power"):
        out[source] = finite(data.get(source), 0, 1000)
    memory = data.get("memory") or {}
    if isinstance(memory, dict):
        for source in ("ram_usage", "ram_total", "swap_usage"):
            out[source] = finite(memory.get(source))
    # macmon 0.6.1 includes invalid M4 sensors in its temperature averages.
    # Temperatures below come only from explicitly selected, validated SMC keys.
    return out


class Version(C.Structure):
    _fields_ = [("major", C.c_uint8), ("minor", C.c_uint8), ("build", C.c_uint8),
                ("reserved", C.c_uint8), ("release", C.c_uint16)]


class Limits(C.Structure):
    _fields_ = [("version", C.c_uint16), ("length", C.c_uint16),
                ("cpu", C.c_uint32), ("gpu", C.c_uint32), ("mem", C.c_uint32)]


class KeyInfo(C.Structure):
    _fields_ = [("size", C.c_uint32), ("type", C.c_uint32), ("attributes", C.c_uint8)]


class KeyData(C.Structure):
    _fields_ = [("key", C.c_uint32), ("version", Version), ("limits", Limits),
                ("info", KeyInfo), ("result", C.c_uint8), ("status", C.c_uint8),
                ("command", C.c_uint8), ("data32", C.c_uint32), ("bytes", C.c_uint8 * 32)]


def decode_smc(kind, raw):
    if kind == b"flt " and len(raw) == 4:
        return struct.unpack("<f", raw)[0]
    if kind == b"sp78" and len(raw) == 2:
        return int.from_bytes(raw, "big", signed=True) / 256
    if kind == b"fpe2" and len(raw) == 2:
        return int.from_bytes(raw, "big") / 4
    return None


def smc_sensors(chip):
    """Only SMC key-info (9) and read-bytes (5); no write or fan-control API."""
    if "M4" in chip:
        gpu_keys = "Tg1U Tg1k Tg0K Tg0L Tg0d Tg0e Tg0j Tg0k".split()
        cpu_keys = "Te05 Te0S Te09 Te0H Tp01 Tp05 Tp09 Tp0D Tp0V Tp0Y Tp0b Tp0e".split()
    elif "M3" in chip:
        gpu_keys = "Tf14 Tf18 Tf19 Tf1A Tf24 Tf28 Tf29 Tf2A".split()
        cpu_keys = "Te05 Te0L Te0P Te0S Tf04 Tf09 Tf0A Tf0B Tf0D Tf0E Tf44 Tf49 Tf4A Tf4B Tf4D Tf4E".split()
    else:
        return {}
    assert C.sizeof(KeyData) == 80
    io = C.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
    system = C.CDLL("/usr/lib/libSystem.B.dylib")
    signatures = {
        "IOServiceMatching": ([C.c_char_p], C.c_void_p),
        "IOServiceGetMatchingService": ([C.c_uint32, C.c_void_p], C.c_uint32),
        "IOServiceOpen": ([C.c_uint32, C.c_uint32, C.c_uint32, C.POINTER(C.c_uint32)], C.c_int),
        "IOConnectCallStructMethod": ([C.c_uint32, C.c_uint32, C.c_void_p, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)], C.c_int),
        "IOObjectRelease": ([C.c_uint32], C.c_int),
        "IOServiceClose": ([C.c_uint32], C.c_int),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(io, name)
        fn.argtypes, fn.restype = args, result
    system.mach_task_self.argtypes, system.mach_task_self.restype = [], C.c_uint32
    service = io.IOServiceGetMatchingService(0, io.IOServiceMatching(b"AppleSMC"))
    if not service:
        return {}
    connection = C.c_uint32()
    try:
        status = io.IOServiceOpen(service, system.mach_task_self(), 0, C.byref(connection))
    finally:
        io.IOObjectRelease(service)
    if status:
        return {}

    def read_key(key):
        request = KeyData()
        request.key = int.from_bytes(key.encode("ascii"), "big")
        request.command = 9
        answer = KeyData()
        size = C.c_size_t(C.sizeof(answer))
        if io.IOConnectCallStructMethod(connection, 2, C.byref(request), C.sizeof(request), C.byref(answer), C.byref(size)) or answer.result:
            return None
        if not 0 < answer.info.size <= 32:
            return None
        request.info, request.command = answer.info, 5
        size.value = C.sizeof(answer)
        if io.IOConnectCallStructMethod(connection, 2, C.byref(request), C.sizeof(request), C.byref(answer), C.byref(size)) or answer.result:
            return None
        return decode_smc(request.info.type.to_bytes(4, "big"), bytes(answer.bytes[:request.info.size]))

    out = {}
    try:
        for label, keys in (("gpu", gpu_keys), ("cpu", cpu_keys)):
            readings = [finite(read_key(key), 10, 125) for key in keys]
            values = [v for v in readings if v is not None]
            out[label + "_temp"] = sum(values) / len(values) if values else None
            out[label + "_sensor_count"] = len(values)
        out["fan_rpm"] = [v for key in ("F0Ac", "F1Ac") if (v := finite(read_key(key), 0, 20000)) is not None]
    finally:
        io.IOServiceClose(connection)
    return out


def parse_network(output):
    counters = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 10 or not fields[2].startswith("<Link#"):
            continue
        name = fields[0].rstrip("*")
        if not re.fullmatch(r"en\d{1,2}", name):
            continue
        try:
            rx, tx = int(fields[-5]), int(fields[-2])
        except ValueError:
            continue
        if rx >= 0 and tx >= 0:
            counters[name] = {"rx": rx, "tx": tx}
    return counters


def sample_hardware(chip):
    out = {"source": "macmon / IOReport + read-only SMC", "load_kind": "frequency_scaled",
           "network_scope": "OS interface bytes; RDMA bandwidth is not measured"}
    try:
        raw = subprocess.check_output(["/opt/homebrew/bin/macmon", "pipe", "--samples", "1", "--interval", "500"],
                                      text=True, stderr=subprocess.DEVNULL, timeout=7)
        out.update(parse_macmon(json.loads(raw.strip())))
    except (OSError, ValueError, subprocess.SubprocessError):
        out["sensor_error"] = "GPU load probe unavailable"
    try:
        out.update(smc_sensors(chip or ""))
    except (OSError, AttributeError, ValueError):
        out["temperature_error"] = "SMC sensors unavailable"
    try:
        raw = subprocess.check_output(["/usr/sbin/netstat", "-ibn"], text=True, timeout=2)
        out["network_counters"] = parse_network(raw)
        out["network_time"] = time.monotonic()
    except (OSError, subprocess.SubprocessError):
        pass
    return out
