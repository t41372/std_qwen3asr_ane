"""90-second PSTR active control: 30s baseline, 30s four CPU workers, 30s recovery."""

import json
import subprocess
import sys
import time

from whole_machine import SystemPower, battery_sample


def main():
    power = SystemPower()
    workers = []
    start = time.monotonic()
    try:
        for index in range(90):
            if index == 30:
                code = "import time\nend=time.monotonic()+30\nx=1\nwhile time.monotonic()<end:\n x=(x*1664525+1013904223)%2**32\n"
                workers = [
                    subprocess.Popen(
                        [sys.executable, "-c", code], stdout=subprocess.DEVNULL
                    )
                    for _ in range(4)
                ]
            if index == 60:
                for worker in workers:
                    if worker.poll() is None:
                        worker.terminate()
                for worker in workers:
                    worker.wait()
                workers = []
            row = {
                "sample": index,
                "monotonic_s": time.monotonic(),
                "wall_time_s": time.time(),
                "phase": "baseline"
                if index < 30
                else "cpu4"
                if index < 60
                else "recovery",
                "pstr_w_estimate": power.watts(),
            }
            if index % 5 == 0:
                row.update(battery_sample())
            print(json.dumps(row, sort_keys=True), flush=True)
            if index < 89:
                time.sleep(max(0, start + index + 1 - time.monotonic()))
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
            worker.wait()
        power.close()


if __name__ == "__main__":
    main()
