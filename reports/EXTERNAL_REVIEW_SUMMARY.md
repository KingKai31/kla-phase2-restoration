# What this project does — a plain summary for someone new to it

## The problem

Electron-microscope images used for inspecting tiny manufactured
surfaces often come out noisy and blurry, at lower resolution than
you'd want. The task: build software that takes a degraded image and
produces a cleaner, sharper, higher-resolution version, automatically.

## The core idea

Instead of guessing what kind of noise these images have, we first
carefully measured the *actual* noise in real sample images. It followed
a specific, known pattern from physics: the randomness you get from
counting individual particles of light (like grain in a low-light
photo), combined with a second kind of noise from the camera's own
electronics. We built a formula for this exact combination and checked
it against real data — it matched almost perfectly, better than any
simpler version we tried. We used that formula to generate large amounts
of realistic practice data to train the cleanup model on.

## The result

On a real, official test set we were given, the finished model:

- Improves image quality by a wide, clearly measurable margin over a
  standard sharpen-and-denoise approach — an improvement extremely
  unlikely to be a fluke (checked with proper statistics).
- Runs in well under a tenth of a second per image on modern hardware.
- Keeps working sensibly across genuinely different sample types it
  wasn't specifically trained on, including real category labels we
  independently tracked down, not just one narrow image set.

## The one honest weak spot

Where a real physical edge — a boundary between two structures — exists
in a sample, our model doesn't preserve it quite as sharply as a much
simpler, older technique does. This matters because in real inspection
work, spotting exactly where a boundary or defect sits can be the whole
point. We measured this precisely, found the likely cause, and tried two
targeted fixes. Both sharpened the edges as intended, but each also made
the image slightly less accurate elsewhere, more than we'd accept — so
neither was used. This limitation is real and unresolved.

## The overall approach

Decisions were tested, not assumed: before adopting any change, we
agreed in advance what result would count as a genuine improvement, then
measured it honestly — including several cases where the honest answer
was "no, this didn't actually help," reported as clearly as the
successes.
