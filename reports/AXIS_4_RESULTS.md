# Axis 4 results — bottleneck self-attention: real, small win that fails its own gate

Per the pre-registered decision rule
(`reports/axis4_architecture_decision_rule_PREREGISTERED.md`): baseline
NAFNetSR vs. NAFNetSR + one lightweight self-attention block at the
bottleneck, 15 epochs each, same data/loss/seed/schedule.

| | Baseline | + Attention | Delta |
|---|---|---|---|
| PSNR | 23.476 | 23.510 | +0.034 dB |
| SSIM | 0.5958 | 0.6008 | +0.0050 |
| LPIPS | 0.1851 | 0.1845 | -0.0006 (better) |
| Composite | 0.6586 | 0.6612 | +0.0026 |
| Params | 6.82M | 7.87M | +15% |

**The attention variant wins on every individual metric - and still
fails the pre-registered gate.** The composite gain (0.0026) is roughly
a quarter of the required 0.01 margin. Per the rule agreed in advance
("adopt only if composite improves by at least 0.01... if either
condition fails, the attention block is dropped and the result is
reported as a negative/null finding - not tuned post-hoc"):

**Decision: DROP.** Not adopted for the final model. This is a real,
consistent-direction, small positive signal - not nothing - but "every
metric moved the right way by a small amount" is exactly the kind of
result a pre-registered margin exists to guard against overinterpreting.
15% more parameters for a quarter of the required improvement is not a
good trade, and the rule was agreed before the run specifically so this
call wouldn't need to be made by eye after seeing a result that "looks
promising."

**One real process lesson from this run, worth keeping:** the first
attempt at this comparison died silently partway through (a pod-level
disk-quota issue, unrelated to the model itself - see
`reports/HARDENING_DISK_QUOTA_INCIDENT.md`) after the baseline config had
already fully finished its 15 epochs - and that completed result was lost
because it was only held in memory, never written to disk until the very
end of the whole script. Fixed by saving each config's result to its own
file the moment it completes (`scripts/axis4_architecture_ab_test.py`),
so a crash mid-comparison no longer discards already-finished work.

Full results: `reports/axis4_architecture_ab_summary.json`.
