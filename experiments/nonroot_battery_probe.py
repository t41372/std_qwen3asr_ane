"""Read selected battery power values; never persist the full registry response.

The derived watt value assumes the registry voltage/current use mV/mA.
It is a whole-machine battery estimate, not CPU/GPU/ANE attribution.
"""

import argparse
import json
import os
import plistlib
import subprocess
import time


def sample():
    result = subprocess.run(
        ["/usr/sbin/ioreg", "-r", "-c", "AppleSmartBattery", "-a"],
        check=True,
        capture_output=True,
    )
    objects = plistlib.loads(result.stdout)
    if not objects:
        raise RuntimeError("AppleSmartBattery unavailable")
    battery = objects[0]
    telemetry = battery.get("PowerTelemetryData", {})
    record = {
        "monotonic_s": time.monotonic(),
        "uid": os.getuid(),
        "external_connected": battery.get("ExternalConnected"),
        "voltage_raw": battery.get("Voltage"),
        "instant_amperage_raw": battery.get("InstantAmperage"),
        "telemetry": {
            key: telemetry[key]
            for key in (
                "SystemLoad",
                "BatteryPower",
                "SystemPowerIn",
                "SystemEnergyConsumed",
                "AccumulatedSystemEnergyConsumed",
                "SystemLoadAccumulatorCount",
                "PowerTelemetryErrorCount",
            )
            if key in telemetry
        },
    }
    voltage, current = record["voltage_raw"], record["instant_amperage_raw"]
    if isinstance(voltage, int) and isinstance(current, int):
        # Some versions expose the signed current as an unsigned 64-bit integer.
        if current >= 2**63:
            current -= 2**64
        record["battery_discharge_w_estimate"] = -voltage * current / 1e6
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.samples <= 120:
        parser.error("samples must be between 1 and 120")
    for index in range(args.samples):
        print(json.dumps(sample(), sort_keys=True), flush=True)
        if index + 1 < args.samples:
            time.sleep(1)


if __name__ == "__main__":
    main()
