"""Bounded normal-user battery and PSTR collector; no SMC writes or identifiers.

PSTR watts follow macmon's private SMC convention. Battery energy fields remain
raw until units and cadence are validated. All timestamps use Python monotonic.
"""

import argparse
import ctypes as C
import json
import math
import os
import plistlib
import struct
import subprocess
import time


class Version(C.Structure):
    _fields_ = [(k, C.c_uint8) for k in ("major", "minor", "build", "reserved")] + [
        ("release", C.c_uint16)
    ]


class Limits(C.Structure):
    _fields_ = [("version", C.c_uint16), ("length", C.c_uint16)] + [
        (k, C.c_uint32) for k in ("cpu", "gpu", "memory")
    ]


class KeyInfo(C.Structure):
    _fields_ = [("size", C.c_uint32), ("type", C.c_uint32), ("attributes", C.c_uint8)]


class KeyData(C.Structure):
    _fields_ = [
        ("key", C.c_uint32),
        ("version", Version),
        ("limits", Limits),
        ("info", KeyInfo),
        ("result", C.c_uint8),
        ("status", C.c_uint8),
        ("command", C.c_uint8),
        ("data", C.c_uint32),
        ("bytes", C.c_uint8 * 32),
    ]


class SystemPower:
    """Read only PSTR through macmon's AppleSMCKeysEndpoint protocol."""

    def __init__(self):
        self.io = C.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
        self.io.IOServiceMatching.argtypes = [C.c_char_p]
        self.io.IOServiceMatching.restype = C.c_void_p
        self.io.IOServiceGetMatchingServices.argtypes = [
            C.c_uint32,
            C.c_void_p,
            C.POINTER(C.c_uint32),
        ]
        self.io.IOConnectCallStructMethod.argtypes = [
            C.c_uint32,
            C.c_uint32,
            C.c_void_p,
            C.c_size_t,
            C.c_void_p,
            C.POINTER(C.c_size_t),
        ]
        iterator = C.c_uint32()
        status = self.io.IOServiceGetMatchingServices(
            0, self.io.IOServiceMatching(b"AppleSMC"), C.byref(iterator)
        )
        if status:
            raise RuntimeError(f"SMC matching failed: {status}")
        self.connection = C.c_uint32()
        try:
            while device := self.io.IOIteratorNext(iterator):
                try:
                    name = C.create_string_buffer(128)
                    self.io.IORegistryEntryGetName(device, name)
                    if name.value == b"AppleSMCKeysEndpoint":
                        task = C.c_uint32.in_dll(C.CDLL(None), "mach_task_self_").value
                        status = self.io.IOServiceOpen(
                            device, task, 0, C.byref(self.connection)
                        )
                        if status:
                            raise RuntimeError(f"SMC open failed: {status}")
                        break
                finally:
                    self.io.IOObjectRelease(device)
        finally:
            self.io.IOObjectRelease(iterator)
        if not self.connection.value:
            raise RuntimeError("SMC endpoint absent")
        self.key = int.from_bytes(b"PSTR", "big")
        self.info = self.read(KeyData(key=self.key, command=9)).info
        if self.info.size != 4 or self.info.type != int.from_bytes(b"flt ", "big"):
            raise RuntimeError("PSTR is not a four-byte float")

    def read(self, request):
        response, size = KeyData(), C.c_size_t(C.sizeof(KeyData))
        status = self.io.IOConnectCallStructMethod(
            self.connection,
            2,
            C.byref(request),
            C.sizeof(request),
            C.byref(response),
            C.byref(size),
        )
        if status or response.result:
            raise RuntimeError(
                f"SMC read failed: transport={status}, result={response.result}"
            )
        return response

    def watts(self):
        response = self.read(KeyData(key=self.key, command=5, info=self.info))
        value = struct.unpack("<f", bytes(response.bytes[:4]))[0]
        if not math.isfinite(value) or value < 0:
            raise RuntimeError("Invalid PSTR value")
        return value

    def close(self):
        self.io.IOServiceClose(self.connection)


BATTERY_KEYS = (
    "ExternalConnected",
    "IsCharging",
    "Voltage",
    "Amperage",
    "InstantAmperage",
    "CurrentCapacity",
    "MaxCapacity",
    "AppleRawCurrentCapacity",
    "AppleRawMaxCapacity",
    "DesignCapacity",
    "NominalChargeCapacity",
    "UpdateTime",
    "Temperature",
)
DATA_KEYS = (
    "RemainingCapacity",
    "FullChargeCapacity",
    "NominalChargeCapacity",
    "Voltage",
    "Current",
    "StateOfCharge",
)
TELEMETRY_KEYS = (
    "SystemLoad",
    "AccumulatedSystemLoad",
    "SystemLoadAccumulatorCount",
    "SystemEnergyConsumed",
    "AccumulatedSystemEnergyConsumed",
    "SystemPowerIn",
    "AccumulatedSystemPowerIn",
    "SystemPowerInAccumulatorCount",
    "BatteryPower",
    "PowerTelemetryErrorCount",
)


def selected(source, keys):
    return {k: source[k] for k in keys if isinstance(source.get(k), (int, float, bool))}


def battery_sample():
    output = subprocess.run(
        ["/usr/sbin/ioreg", "-r", "-c", "AppleSmartBattery", "-a"],
        capture_output=True,
        check=True,
        timeout=10,
    )
    batteries = plistlib.loads(output.stdout)
    if not batteries:
        raise RuntimeError("Battery unavailable")
    b = batteries[0]
    return {
        "battery": selected(b, BATTERY_KEYS),
        "battery_data": selected(b.get("BatteryData", {}), DATA_KEYS),
        "telemetry": selected(b.get("PowerTelemetryData", {}), TELEMETRY_KEYS),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--battery-every", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 3600 or not 1 <= args.battery_every <= 60:
        parser.error("seconds must be 1..3600 and battery-every 1..60")
    power = None
    try:
        power = SystemPower()
    except Exception as error:  # noqa: BLE001 — collector records unavailable telemetry.
        print(json.dumps({"event": "smc_unavailable", "error": str(error)}), flush=True)
    start = time.monotonic()
    try:
        for index in range(args.seconds):
            row = {
                "sample": index,
                "uid": os.getuid(),
                "monotonic_s": time.monotonic(),
                "wall_time_s": time.time(),
            }
            if power:
                try:
                    row["pstr_w_estimate"] = power.watts()
                except Exception as error:  # noqa: BLE001 — preserve sample failures in the trace.
                    row["smc_error"] = str(error)
            if index % args.battery_every == 0:
                try:
                    row.update(battery_sample())
                except Exception as error:  # noqa: BLE001 — battery errors do not erase SMC data.
                    row["battery_error"] = str(error)
            row["read_duration_s"] = time.monotonic() - row["monotonic_s"]
            print(json.dumps(row, sort_keys=True), flush=True)
            if index + 1 < args.seconds:
                time.sleep(max(0, start + index + 1 - time.monotonic()))
    finally:
        if power:
            power.close()


if __name__ == "__main__":
    main()
