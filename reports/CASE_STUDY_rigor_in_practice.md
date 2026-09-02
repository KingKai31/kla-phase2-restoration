# Case study: rigor in practice — the ROI-preservation loss

**For the "failures and successes" PPT section.** This is the single
strongest evidence of real methodological discipline in this project, not
something to downplay as "a feature we tried and cut."

## The sequence, in order, as it actually happened

1. **Built a 6th loss term** (`src/losses/roi_preservation.py`) grounded in
   real literature (Zhang et al. 2025), designed to preserve fine
   structure more aggressively than a uniform loss - a real, motivated
   engineering addition, not a throwaway idea.

2. **Pre-registered the decision rule BEFORE running any comparison**
   (`reports/roi_loss_decision_rule_PREREGISTERED.md`): keep the term only
   if it (a) doesn't regress PSNR/SSIM by more than a stated margin, AND
   (b) measurably improves real defect survival on a stress test, with
   both thresholds fixed and committed before a single result existed.

3. **Ran the comparison. Found a bug in our own evaluation before
   trusting its output.** The first version of the defect-preservation
   stress test's hallucination check compared a restored image to
   *itself* - which is trivially always exactly zero, not a real test. It
   would have silently validated nothing. Caught by inspecting the
   result rather than accepting a clean-looking number, fixed to compare
   two independently-noised restorations of the same clean signal
   instead.

4. **Re-ran with the fix. Got a real null result and a real negative
   signal.** None of three real perturbation types showed a statistically
   significant defect-survival improvement (paired Wilcoxon, p=0.062 /
   0.184 / 0.102, n=100 each) - the earlier positive-looking deltas were
   noise, not signal. The fixed hallucination check found a *significant*
   effect (p=0.039) in the wrong direction: the ROI term made the model
   slightly *more* sensitive to noise-realization differences at random,
   defect-free locations - precisely the risk the term's own design
   docstring had named as a real possibility before any result existed.

5. **Applied the pre-registered rule and dropped the term** - exactly as
   committed to in step 2, with no post-hoc tuning of the boost factor or
   threshold to try to rescue a result that didn't hold up.

## Why this is the strong story, not a weak one

- The rule was fixed **before** the data existed, so there was no room to
  move the goalposts once real numbers came in.
- A bug that would have manufactured a false "it works" result was
  **caught by the team that wrote both the feature and the test**, not by
  an external reviewer - the kind of self-checking discipline that's easy
  to describe and hard to actually do under deadline pressure.
- The final decision **cost the project a feature it spent real time
  building**, and it was dropped anyway because the evidence said so.
- The dead code was not deleted - it stays in the repo with its real
  negative result attached (`reports/roi_loss_FINAL_DECISION.md`), so the
  finding is auditable, not just asserted.

## The one-line version for a slide

*"We built a 6th loss term, pre-registered exactly what would justify
keeping it, caught a bug in our own test that would have falsely
validated it, and dropped it anyway once the real evidence came back
negative — the discipline held even when it cost us a feature."*
