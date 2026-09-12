# Non-root power telemetry probe

Date: 2026-09-12. Host: Apple M5 Max, 64 GB, macOS 27.0 build 26A428.

## Outcome

Power telemetry is accessible as ordinary UID 501 without `sudo`, a password, installation, or settings changes. However, **this OS currently returns unusable CPU and ANE energy deltas, including during deliberate CPU and ANE activity**. A full CPU/GPU/ANE energy gate is therefore not justified yet.

Codex's execution sandbox blocks both IOReport subscriptions and the selected `ioreg` query. Running the same probes outside that sandbox succeeded after automatic approval; the process still had UID 501. This is a sandbox capability distinction, not root access. Do not assume a sandbox-only automation can collect these metrics.

## IOReport evidence

`experiments/nonroot_power_probe.m` is a small standalone Objective-C probe, compiled locally with Foundation and the system `libIOReport`. It subscribes only to CPU, GPU, ANE, and DRAM channels in `Energy Model`, samples adjacent one-second windows, and writes selected counters as JSONL. No full hardware registry dump is persisted.

| Channel | Reported unit | Five baseline windows | Five windows with four busy CPU workers |
| --- | --- | --- | --- |
| CPU Energy | mJ | All deltas zero | All deltas zero |
| ANE0 | mJ | All deltas zero | All deltas zero |
| DRAM0 | mJ | All deltas zero | All deltas zero |
| GPU Energy | nJ | All deltas positive, ~0.25–0.28 W | All deltas positive, ~0.24–0.25 W |

The CPU workload consisted of four independent integer-arithmetic Python processes, bounded to eight seconds. No complete ASR pipeline ran. CPU zeros under deliberate CPU activity are evidence that this counter cannot currently support energy claims. Initial ANE and DRAM zeros alone did not establish counter failure, so an explicitly authorized ANE active control followed. GPU counters advance, but no GPU workload or external meter was used to calibrate their accuracy.

Raw evidence: `artifacts/power-probe/idle.jsonl`, `cpu-load.jsonl`, and aggregate `summary.json`. These are whole-machine device counters, with background activity included. “Baseline” here means no deliberate probe workload; it is not a controlled idle benchmark.

### ANE active control

The existing `experiments/repeat_microbench.py` ran `artifacts/probes/decoder-layer-0-scale1.mlpackage` with `CPU_AND_NE` for 10.000692 seconds. This small decoder layer has prior ANE placement evidence from the main project; this run did not repeat hardware tracing. It completed 10,721 predictions, all outputs finite. The workload report records load and predict start/end timestamps in `ane-workload.json`; stdout/stderr are in `ane-workload.log`.

The collector ran continuously for 25 windows spanning model load, prediction, and recovery. **CPU Energy, ANE0, and DRAM0 returned zero in every window**. GPU returned positive deltas in 24/25 windows. This active control makes ANE0 unusable for the intended energy gate on this host; it does not imply zero ANE consumption. GPU energy is not a substitute for whole-machine or CPU+ANE energy.

`ane-active.jsonl` records window times; `ane-active-summary.json` aligns them with the workload. Python's monotonic clock uses `mach_absolute_time`, whereas the C collector uses `CLOCK_MONOTONIC`; they differed by 35.745904 seconds on this host. Ten paired readings measured the offset after the run. Adding that offset to Python times gives 4 windows before prediction, 11 overlapping prediction, and 10 after. This post-run alignment assumes no sleep-induced offset change during the run; exact attribution at the two boundary windows is approximate. The all-zero CPU/ANE conclusion is independent of this alignment because all 25 windows were zero. GPU means were 0.257/0.319/0.347 W for these respective phases, with background activity uncontrolled.

## Exact API and source provenance

Primary implementation inspected: [macmon sources.rs](https://github.com/vladkens/macmon/blob/6919d7781b6c55a6e3bedff83a210435837e1dfe/src_lib/sources.rs) and [metrics.rs](https://github.com/vladkens/macmon/blob/6919d7781b6c55a6e3bedff83a210435837e1dfe/src_lib/metrics.rs), upstream commit `6919d7781b6c55a6e3bedff83a210435837e1dfe`. Downloaded source hashes are in the local summary.

The private API sequence is `IOReportCopyAllChannels` → filtered `IOReportChannels` → `IOReportCreateSubscription` → two `IOReportCreateSamples` calls → `IOReportCreateSamplesDelta`. `IOReportSimpleGetIntegerValue(deltaChannel, 0)` returns energy in the channel's `IOReportChannelGetUnitLabel` unit. Convert mJ/uJ/nJ to J using 1e-3/1e-6/1e-9, then divide by measured monotonic elapsed seconds for mean W. Preserve joules as the primary integrated quantity; do not multiply average watts by an unrelated nominal interval. Unknown units and negative deltas must not become valid power values.

The probe follows upstream channel matching: CPU names ending in `CPU Energy`, exact `GPU Energy`, and names starting with `ANE`. This is a private interface with no compatibility guarantee.

An independent [upstream macOS 27 beta report](https://github.com/vladkens/macmon/issues/76) describes zero CPU energy and working GPU energy on an M3 Max. It is consistent with our observation but does not establish the root cause on this M5 Max. Its claims about powermetrics internals were not verified here.

## Battery / whole-machine fallback

`experiments/nonroot_battery_probe.py` reads only selected values from `AppleSmartBattery` and `PowerTelemetryData`; it retains no serial numbers, device identifiers, or complete registry output. While disconnected from external power, one observation was `Voltage=12091`, `InstantAmperage=-1026`, `SystemLoad=12405`. The current × voltage estimate is 12.405366 W; `SystemLoad/1000` is 12.405 W. A later observation similarly matched at 11.2978 W versus 11.297 W.

Apple's local SDK `IOKit/ps/IOPSKeys.h` documents `Voltage` in mV and the public `Current` property in mA. Apple's [PowerManagement source](https://github.com/apple-oss-distributions/PowerManagement/blob/main/pmconfigd/BatteryTimeRemaining.m) reads the registry `InstantAmperage` as a signed integer. The mA interpretation for that private key and the mW interpretation for `SystemLoad` are additionally supported by the numerical agreement above; they are not a newly verified specification of `PowerTelemetryData`.

Five baseline and ten subsequent load samples returned identical battery values and accumulator counts. Thus querying every second does not ensure one-second sensor freshness. The output keeps raw values and calls the derived value an estimate. `SystemEnergyConsumed` and its accumulator were zero/stale in the initial battery observation; their units and integration semantics are unverified and must not be used as joules.

A final 45-sample window, with four CPU workers bounded to 40 seconds, did show battery updates. It began at estimated 57.656 W (`SystemLoad=57546`, count 13913) and changed approximately 35.92 seconds later to 51.828 W (`SystemLoad=51600`, count 13972). See `battery-cpu-load-long.jsonl`. This demonstrates non-root access to changing whole-machine telemetry, while confirming stale repeated reads over intervals far longer than an utterance. The difference between the current × voltage estimate and SystemLoad also prevents treating their agreement as exact calibration. This single run does not establish a guaranteed sensor update interval.

This path measures battery discharge for the entire machine, including display, networking, and background processes. It cannot attribute ANE savings, and battery charging/AC operation requires a different interpretation. For long repeated workloads it may become a useful exploratory whole-machine estimate after cadence and calibration checks. It is unsuitable for a short utterance energy gate as currently validated.

## Reproduction

Run from the repository root outside the restrictive agent sandbox, as the normal user:

```sh
clang -O2 -framework Foundation -lIOReport experiments/nonroot_power_probe.m -o artifacts/power-probe/nonroot_power_probe
artifacts/power-probe/nonroot_power_probe 5
python3 experiments/nonroot_battery_probe.py --samples 5
```

No global dependency installation is needed. Automated reports should mark CPU and ANE energy invalid on this tested host, preserve raw evidence, and continue using measured latency/quality gates. Revalidate active counters after OS changes and calibrate against a trusted reference before introducing an energy gate. Never interpret unavailable or stalled counters as zero energy consumption.
