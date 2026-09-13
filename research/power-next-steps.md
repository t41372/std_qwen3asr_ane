# Power measurement follow-up — 2026-09-12

Host: Apple M5 Max, 64 GB, macOS 27 build 26A428. All probes ran as ordinary user, without sudo, installation, or settings changes. The agent sandbox required an approved escape for IOKit reads; that is not root access.

## New result

**SMC `PSTR` supplies changing whole-machine power estimates at one-second cadence on this host.** This is a usable path for exploratory long-workload energy comparisons, while IOReport CPU/ANE energy remains invalid. `experiments/power_v2/whole_machine.py` reads only this SMC key and allowlisted numeric battery fields. `integrate.py` integrates bracketed PSTR samples and computes gross joules per audio second; optional idle subtraction is explicitly exploratory. These are full-machine estimates, not CPU/ANE attribution or calibrated energy gates.

A 120-sample cadence observation returned 119/119 adjacent PSTR changes, range 9.52–23.54 W and arithmetic mean 12.09 W. Battery values refreshed at samples 31, 59, and 118. This observation was **not controlled idle**; concurrent development and possible compilation make it a cadence diagnostic only. Raw records and a compact summary are in `artifacts/power-v2/whole-machine-120s.jsonl` and `whole-machine-summary.json`.

The PSTR interpretation follows [macmon's system-power implementation](https://github.com/vladkens/macmon/blob/main/src_lib/metrics.rs), which reads PSTR as its system watt estimate. Its [SMC source](https://github.com/vladkens/macmon/blob/main/src_lib/sources.rs) opens AppleSMCKeysEndpoint and checks four-byte `flt ` values, decoded little-endian. We reproduced that narrowly with ctypes. Local downloaded source snapshots are retained under `artifacts/power-v2/`; this establishes software interpretation, not a published Apple sensor accuracy guarantee.

## Controlled CPU response

A separate 90-second run reserved a quiet window from the main ANE benchmark: 30 seconds baseline, 30 seconds four bounded integer-arithmetic CPU workers, then 30 seconds recovery. No model inference ran in this control. PSTR phase means were **12.18 / 53.95 / 15.86 W**; the last 20 seconds of each phase averaged **12.10 / 55.72 / 13.15 W**. The first load sample remained near baseline and the first recovery sample remained elevated, confirming transition/sensor delay. This validates gross responsiveness, not instantaneous attribution or calibration. Use long steady blocks and preserve boundary uncertainty. Raw evidence is `cpu-control-90s.jsonl`, summary `cpu-control-summary.json`; reproduce with `python3 experiments/power_v2/cpu_control.py`.

## Why the original counters fail

`channel_probe.m` broadens selection to every Energy Model channel plus CPU statistics and GPU performance states. Five adjacent windows were run with both the subscription-returned channel dictionary and the original dictionary. Each variant found 364 energy channels. Only GPU Energy produced positive deltas. All 358 mJ channels had **identical RawElements payloads before and after every sample**, including alternate MCPU, PACC, CPU SRAM/detail, ANE0, GPU0, and DRAM0 counters. All 18 CPU core performance-state channels advanced residency counters. The implementation emits raw integer values and payload-equality booleans, never provider IDs or full dictionaries.

This isolates the fault below delta conversion and channel-name filtering: neither alternate M5 channel names nor the two sampling dictionary variants restore power data. The energy-producing path is stalled while the general sampling/residency path works. Exact cause inside Apple's driver (beta regression, lifecycle bug, or another firmware condition) is still unproved; no reboot or sleep-state intervention was performed. Detailed evidence: `channels-subscribed.jsonl`, `channels-original.jsonl`, `channel-summary.json`.

[macmon issue 76](https://github.com/vladkens/macmon/issues/76) reports matching frozen energy payloads on an M3 Max running another macOS27 beta, with working GPU/residency/PSTR readings. Its claim that powermetrics bypasses energy channels is not established here. In fact, [upstream tracing on macOS26.5.2](https://github.com/vladkens/macmon/blob/main/research/powermetrics/readme.md) observes powermetrics subscribing to and sampling Energy Model as well as residency channels. That older trace does not settle macOS27 implementation. We did not follow its invasive tracing setup.

## Battery fallback: what the new fields establish

This host lacks top-level AppleRawCurrentCapacity; `BatteryData.RemainingCapacity` exists. The [Stats battery reader](https://github.com/exelban/stats/blob/master/Modules/Battery/readers.swift) implements that fallback, following a [macOS regression report](https://github.com/exelban/stats/issues/3392). Our collector retains both possibilities without silently substituting percent values for charge.

During the 120-sample observation, RemainingCapacity changed 3580 → 3561 → 3552 → 3535. SystemLoadAccumulatorCount changed 28255 → 28315 → 28343 → 28402. AccumulatedSystemLoad increased, but AccumulatedSystemEnergyConsumed remained 80701424 and SystemEnergyConsumed stayed zero throughout discharge. **Do not integrate the field named SystemEnergyConsumed.** Its apparent lack of battery discharge response contradicts assuming a general battery joule accumulator.

The ratio ΔAccumulatedSystemLoad / ΔSystemLoadAccumulatorCount gives 13425.25, 11936.18, and 11940.32 raw units for the observed update intervals. Interpreted as mean mW these values are plausible and counts advance approximately once per second, but cached publication occurred at 28–60-second intervals. This is empirical evidence for a sample-sum/count model, not documented units or continuous-time integration semantics. Unequal underlying intervals would invalidate directly calling the sum energy. A documented specification for these private PowerTelemetryData fields was not found in the primary sources inspected.

If RemainingCapacity is mAh, the charge estimate is ΔQ_mAh × mean_voltage_mV × 0.0036 J. That unit assumption also needs validation on this host; these numbers are not percentages, but magnitude alone cannot prove mAh. Charge-gauge quantization, recalibration, and voltage variation prevent treating a short endpoint difference as exact energy.

## Ready-to-run comparison protocol

1. Keep the same power source, display state, peripheral use, and workload configuration through paired runs. Battery cross-checks require ExternalConnected=false, IsCharging=false, and discharging current at every observed update. Do not switch AC during a pair. PSTR on AC can be collected, but its relation to wall input and charging has not been validated here.
2. Warm models before measurement if evaluating steady inference; separately report cold-load energy if startup is in scope. Use the same fixed audio set/order and completed valid output count for both implementations. Record Python `time.monotonic()` boundaries around the actual workload and total processed audio seconds. CLOCK_UPTIME_RAW in the new C probe avoids the old CLOCK_MONOTONIC offset issue, but Python timestamps are the primary integration clock.
3. Start collection at least 10 seconds before measurement and keep it running at least 10 seconds after. For battery corroboration use at least 10 minutes and 10 fresh publication intervals per phase; retain endpoint update timestamps, not just query times. Do not count cached repeats as independent measurements. Longer is useful only while conditions stay comparable.
4. Run A/B/B/A blocks (then reverse order in another session), with baseline periods before/after each block and sufficient cooling to comparable conditions. Compare equal useful audio work, allowing elapsed times to differ. Report gross J/audio-second and elapsed RTF; idle-adjusted energy is supplementary because background and cooling costs are real and may not subtract linearly.
5. Integrate PSTR with measured elapsed times and linear interpolation at both boundaries. Reject missing coverage or sample gaps above the configured threshold (default 2.5s); never fill absent readings with zero. The CLI preserves negative idle-adjusted results, which indicate noise/model failure instead of silently clamping them.
6. Cross-check long battery-only intervals: PSTR integrated joules versus current×voltage numerical integration, charge decrement with voltage, and the accumulator/count mean. Use exact common fresh intervals. Treat agreement as corroboration, not calibration. Before claiming small savings, establish repeated-run uncertainty smaller than the effect and preferably compare an external calibrated meter. No fixed tolerance is justified by current data.

```sh
python3 experiments/power_v2/whole_machine.py --seconds 900 --battery-every 1 > artifacts/power-v2/comparison.jsonl
python3 experiments/power_v2/integrate.py artifacts/power-v2/comparison.jsonl --start START_MONOTONIC --end END_MONOTONIC --audio-seconds TOTAL_AUDIO_SECONDS
```

Optional `--idle-watts BASELINE_MEAN` adds explicitly exploratory idle-adjusted fields. No admin access is necessary for this measurement path. Accurate per-device CPU/ANE attribution remains unresolved; a normal-user PSTR comparison can now move the whole-machine energy objective forward without waiting on that separate blocker.
