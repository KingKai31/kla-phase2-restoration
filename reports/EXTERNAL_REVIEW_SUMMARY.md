# What this project does — a plain summary for someone new to it

## The problem

Electron-microscope images used for inspecting tiny manufactured
surfaces often come out noisy and blurry, at lower resolution than
you'd want. The task: build software that takes a degraded image and
produces a cleaner, sharper, higher-resolution version, automatically.

## The core idea

Instead of guessing what kind of noise these images have, we measured
the *actual* noise in real samples. It followed a known pattern from
physics: the randomness of counting individual particles of light (like
grain in a low-light photo), combined with noise from the camera's own
electronics. Our formula for that exact combination matched the real
data almost perfectly — better than any simpler version we tried — and
we used it to generate realistic practice data to train on.

## The result

On a real, official test set we were given, the finished model:

- Beats a standard sharpen-and-denoise approach by a wide margin,
  extremely unlikely to be a fluke (checked with proper statistics).
- Runs in well under a tenth of a second per image on modern hardware.
- Keeps working across genuinely different sample types it wasn't
  trained on, not just one narrow image set.

## The one honest weak spot

Where a real physical edge — a boundary between two structures — exists
in a sample, our model doesn't preserve it as sharply as a much simpler,
older technique does. In real inspection work, spotting exactly where a
boundary or defect sits can be the whole point.

We measured this, found the cause, and tried three fixes. Two changed
*what the model is rewarded for* during training; both sharpened edges
as intended, but both also made the image less accurate elsewhere, by
more than we'd agreed in advance to accept — so neither was used. The
third instead gave the model a little extra machinery for producing fine
detail. That improved edge sharpness for essentially no cost anywhere
else, and **it is the version we shipped**.

That closed roughly a sixth of the gap. **Our model still preserves real
boundaries less well than the simpler technique** — the most important
open weakness in this work.

## The overall approach

Decisions were tested, not assumed: before adopting any change we agreed
in advance what would count as a genuine improvement, then measured it
honestly — including the many cases where the answer was "no, this
didn't help," reported as clearly as the successes.
