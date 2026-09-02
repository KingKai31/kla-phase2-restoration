# Disk-quota incident (fresh pod, technical hardening pass)

**A real infrastructure finding worth documenting, not just a footnote.**

## What happened

The fresh RunPod pod's `/workspace` is backed by a shared network
filesystem (`mfs#us-md-1.runpod.net`) reporting 404TB total / 111TB free
at the pool level. This was read as "disk is not a constraint on this
pod" - **incorrect.** `df` on a shared network mount reports the pool's
stats, not this pod's individual allocation. There is a real per-pod
quota, somewhere between 8.1GB (confirmed working) and 15.5GB (confirmed
failing) - the exact number was never pinned down.

Three concurrent background processes - the Axis 4 architecture
comparison (GPU training), the Axis 1a synthetic-pair generation (CPU),
and the Tier 1 NFFA download - all died silently around the same time,
with no exception traceback in any log. Diagnosed by attempting an
unrelated file transfer, which failed with an explicit
`scp: ... close: Disk quota exceeded` - the first error message that
actually named the real cause.

## Real cost of the mistake

- The Axis 4 comparison's baseline config had fully completed all 15
  epochs before dying - that result was lost because it was only held in
  memory, not persisted until the whole script finished (fixed, see
  `reports/AXIS_4_RESULTS.md`).
- The Axis 1a synthetic-pair generation (7.3GB, roughly 53% through the
  4th category) had to be deleted and restarted with a smaller footprint.
- The NFFA download stopped at 4/10 categories and was not resumed at
  the same scope - continuing it risked immediately re-hitting the same
  quota, since the 6 remaining categories add roughly 12.35GB more on
  their own.

## Fix and going-forward practice

- Freed space by deleting the partial synthetic-pair output, confirmed
  the quota theory by testing a small file transfer before and after
  (failed at 15.5GB used, succeeded at 8.1GB).
- Regenerated Axis 1a's synthetic pairs with a much smaller per-image
  tile cap to fit safely within the unknown-but-bounded quota, rather
  than assuming abundant space again.
- Any future long-running background process on this pod should persist
  partial results incrementally (same fix applied to Axis 4's script),
  since a silent quota-driven death gives no warning and no exception to
  catch.
- **Left the remaining 6/10 Tier 1 NFFA categories undownloaded** rather
  than risk repeating this - Axis 1a proceeds with the 4 categories
  already on disk (Biological, Fibres, Films_Coated_Surface,
  MEMS_devices_and_electrodes), reported honestly as a real scope
  limit, not a hidden one.
