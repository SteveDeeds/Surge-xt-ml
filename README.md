# Surge XT ML: first sampling baseline

This project uses Surge XT 1.3.4's official Windows x64 portable distribution,
hosted headlessly by Spotify's Pedalboard. Audio is bounced offline, not played
through an audio device. No DAW, audio driver, or GUI is required.

## Installed environment

- Python 3.12, project-local `.venv`.
- Surge XT, plugins, and factory content under `.tools/SurgeXT-1.3.4`.
- `requirements-lock.txt` records the installed Python dependencies.
- `scripts/Install-SurgeXT.ps1` downloads the official portable release and records
  its URL and local SHA-256. The upstream release has no published API digest;
  the recorded hash is provenance, not independent authenticity verification.

To reproduce on Windows, from this directory:

```powershell
& .\scripts\Install-SurgeXT.ps1
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
& .\.venv\Scripts\python.exe surge_dataset.py
& .\.venv\Scripts\python.exe scripts\validate_dataset.py data\initial_patch
& .\.venv\Scripts\python.exe -m pytest -q
```

Skip the installation steps when already installed. The installer refuses an
existing destination. Rendering also refuses an existing output directory.
Choose a new one with `--output data\another_run`.

On Windows, Pedalboard needs the **inner binary**
`Surge XT.vst3\Contents\x86_64-win\Surge XT.vst3`, not the enclosing bundle.
The default path handles this.

## Sampling

The selected baseline is Surge XT's initial patch, a simple Classic oscillator
patch. Its complete state is saved, rather than relying on a patch name alone.
This initial patch has velocity VCA gain set to 0 dB: it is a useful rendering
and pitch baseline, but not a demonstration of expressive velocity response.
The four distinct MIDI velocities are still sent and recorded.

| Axis | Values |
|---|---|
| Pitch | C0 through C8 inclusive; scientific notation, MIDI 12, 24, ..., 108 |
| Velocity | 32, 64, 96, 127 |
| Note-on | 0 seconds |
| Note-off | 1 second |
| Recording ends | 2 seconds |
| Audio | 48 kHz, stereo, 32-bit float WAV; 96,000 frames per recording |

This produces 36 recordings, totaling 72 seconds of audio. No peak normalization
or loudness normalization is performed.

### Keeping recordings independent

One plugin instance is reused for efficient bouncing, but its **full saved patch
state is restored before every note**. A discarded 100 ms silent pre-roll lets
Surge apply the queued patch load before the note-on. It is not part of the WAV.
The recorded note then starts at time zero with no reset between note-on and tail.

MIDI CC123 (All Notes Off) alone is insufficient: releases and reverb can continue.
CC120 (All Sound Off/panic) is also not a portable guarantee of cleared FX buffers.
In Surge XT 1.3.4, the host reset callback alone only resets its block position.
The important operation here is patch restoration: `loadRaw()` stops voices and
forces `loadFx(false, true)`, which destroys and recreates the effect processors.
This clears delay/reverb history without disabling the effects in the patch.

The one-second tail is a fixed observation window, **not** a claim that every
patch has finished sounding. A longer tail may be truncated within its own WAV;
it must not be carried into the next recording. The manifest includes tail RMS
and last-50-ms RMS to make truncation visible.

The local integration test deliberately enables Reverb 1, requires a measurable
tail at the recording cutoff, then restores the patch and checks two seconds of
MIDI-free output for residual signal below a peak of `1e-7` (-140 dBFS).

Oscillator/noise randomness is not externally seeded. Patch restoration prevents
cross-note DSP history, but does not promise bit-identical repeated audio.

## Representations

`data/initial_patch/features.npz` contains:

| Key | Axes / expected shape |
|---|---|
| `stft_power` | pitch x velocity x frequency x time: `(9, 4, 4097, 188)` |
| `log_mel_db` | pitch x velocity x mel band x time: `(9, 4, 128, 188)` |
| `mfcc` | pitch x velocity x coefficient x time: `(9, 4, 20, 188)` |
| `rms` | pitch x velocity x time: `(9, 4, 188)` |

Coordinate arrays are included: `pitches_midi`, `velocities`, `time_seconds`,
`stft_frequency_hz`, and `mel_center_hz`.

- STFT: 8192-sample Hann window, hop 512, centered with zero padding.
  Powers are averaged across the two channels, avoiding phase cancellation from
  averaging stereo waveforms. These are uncalibrated squared STFT magnitudes,
  not a power spectral density in physical units. Original stereo WAVs are kept.
- Log-mel: 128 Slaney-normalized bands from 0 to 24 kHz, fixed power reference 1,
  floor `1e-10`, no per-recording dB clipping or normalization.
- MFCC: the additional compact timbral representation; 20 orthonormal type-II
  DCT coefficients, including coefficient zero, derived from that same log-mel.
- RMS: a shared-grid energy envelope, useful for attack/release comparisons.

The hop is 10.67 ms, but actual time resolution is limited by the 170.67 ms window.
Frequency bins are 5.86 Hz apart. This is only a baseline at C0 (16.35 Hz);
low-note harmonic comparisons may benefit from CQT or a second, longer window.
Centered windows also smear the note-off boundary. Coordinate times are frame
centers, not causal observation times.

Other outputs:

- `audio/`: all 36 WAVs.
- `manifest.json`: sampling/analysis settings, sample checksums, quality metrics,
  package and plugin provenance, and completion status. A failed run stays
  `in_progress`; do not treat it as a complete dataset.
- `patch.vstpreset`, `patch.raw`, `parameters.json`: saved patch and exposed controls.
- `preview.png`: C4/velocity-96 waveform, STFT, log-mel, and MFCC overview.

To render a saved native VST3 preset:

```powershell
& .\.venv\Scripts\python.exe surge_dataset.py --preset data\initial_patch\patch.vstpreset --output data\repeat
```

Pedalboard's `load_preset()` accepts `.vstpreset`, not Surge's factory `.fxp`.
The batch workflow below adds a version-specific native-state bridge for Surge
FXP files, including their embedded wavetables. The official `surgepy` bindings
remain another option for deeper engine control, but require a native build.

## Ten-patch paired repeat experiment

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'
& .\.venv\Scripts\python.exe surge_batch.py
& .\.venv\Scripts\python.exe -m scripts.validate_batch data\factory_ten
```

The batch selects ten factory patches: FM Bass 1, Artificial 1, Acidofil,
Alias Pornography, Brassy, Clarinet, 80s Gliss, Drum One, 1804, and Aggero.
The source filenames and embedded display names can differ; both are recorded.

For **each original and sanitized variant**, it records the complete 36-point
grid **three times**: 2,160 WAVs and 72 minutes of audio in total. Individual
recordings are kept; waveforms are never averaged. The original factory files,
engine-normalized original states, sanitized states, and exact sanitizer edits
are preserved separately.

### Sanitizer policy and limitations

The installed engine first loads/resaves each factory patch so old revisions and
modulation-source IDs are migrated before editing. The sanitizer:

- Enables retrigger on all six oscillator slots (including inactive slots).
- Sets scene A/B drift to zero and removes modulation routes targeting drift.
- Sets **all 24 LFO slots** (six voice plus six scene per scene) to Keytrigger.
- Removes Random Bipolar/Unipolar and Alternate Bipolar/Unipolar modulation routes.
- Retains noise/S&H/formula LFO shapes, noise oscillators, effects, and the original
  configured LFO start phase. These potential residual sources are audited.

The saved sanitizer report records before/after controls and removed routes.
The engine's saved state is read back to verify that requested edits persisted.
Original patch semantics are not conflated with their sanitized derivatives.
There is **no claim of deterministic audio** and no accessible external RNG seed.
Retrigger does not guarantee a zero-degree phase for every oscillator algorithm.
Freerun advances with DSP/transport state, not necessarily wall-clock time.

Before every recording, the complete variant state is restored and processed
through the same discarded 100 ms pre-roll. Each variant additionally undergoes
two-second MIDI-free silence checks before and after the grid. A residual peak
at or above `1e-7` fails the batch rather than silently accepting possible bleed.
These checks do not claim that every free-running FX LFO or PRNG has a resettable
phase; that remaining variation is measured.

Some factory patches can legitimately be silent at parts of the grid (key splits,
filtering, or velocity mapping). Such cells are retained with explicit silence
flags. Entirely silent variants fail. Above-full-scale float WAVs are preserved
without clipping or normalization and are flagged.

### Repeats, spectral uncertainty, and comparisons

Under `data/factory_ten/<patch-id>/<original|sanitized>/`:

- `features_r0.npz` through `features_r2.npz`: individual feature grids.
- `feature_mean.npz` and `feature_variance.npz`: per-cell arithmetic feature means
  and **sample variance (`ddof=1`)** across repeats, not across pitch or velocity.
- `repeatability.npz`: all three repeat-pair distances at every grid point.
- `audio/`, `patch.vstpreset`, `manifest.json`, and `preview.png`.

The feature keys from the first baseline are retained, plus `stft_short_power`
with FFT size 2048 and shape `(9, 4, 1025, 188)`. Both STFTs use the same hop and
frame centers. Their different windows provide a first multi-resolution comparison.

The repeatability metrics are:

1. **Multi-resolution magnitude distance:** for each STFT size,
   `||sqrt(P1)-sqrt(P2)|| / ((||sqrt(P1)||+||sqrt(P2)||)/2)`, averaged over the two
   resolutions. It is phase-insensitive but remains sensitive to spectral/loudness
   changes. This is a measurement, not a trained loss/model.
2. **Active-bin log-mel RMS difference in dB:** bins within 60 dB of that grid
   cell's mean-log-mel maximum, restricted to frames with mean RMS above `1e-5`.
   Cells without active signal are explicitly counted; their zero distance must
   not be interpreted as a successful timbral match.
3. **Waveform-identical cells:** hashes detect bit-identical repeats separately
   from the spectral metrics. Nonidentical waveforms do not imply perceptually
   significant differences.

The mean of log-mel values is **not** the log-mel of averaged power. Both are saved:
`log_mel_db` in `feature_mean.npz` is the former, while
`log_mel_of_mean_power` is computed from the averaged long-window STFT power.
MFCC means/variances refer to the individual-repeat MFCCs. They are not calculated
by taking the MFCC of an averaged waveform.

`batch.json` records source/plugin/code hashes, environment and settings. The
`pipeline/` directory freezes the exact rendering source files behind those hashes.
`validation.json` and `comparison.csv` are produced only after saved WAVs,
feature shapes, means, variances, checksums, and repeat metrics are verified.
The comparison lets us inspect improvement or degradation rather than assuming
sanitization always reduces variability.

Three repeats give a small-sample within-patch uncertainty estimate, not proof of
determinism or invariance across fresh processes, machines, or engine versions.
Keep all repeats and both variants of a source patch in the same ML dataset split.
Treating them as independent patch examples would leak patch identity.

## Repeatability-only training subset

Variation stays in the acquisition/verification data, **not** as an added training
tensor dimension. The approved initial eligibility limits are:

- Relative spectral-magnitude distance **at most 1% at each STFT resolution**.
- Active-bin log-mel RMS difference **at most 0.1 dB**.
- Both conditions must hold for **every repeat pair and all 36 pitch/velocity cells**.
  An average across resolutions, notes, or repeat pairs cannot hide a failure.
- A cell without active signal is insufficient evidence, not an automatic pass;
  that patch is excluded from this initial full-grid training subset.

These are engineering starting limits, not established perceptual thresholds.
Only sanitized variants are considered. No extra FX disabling or texture removal
is used to force marginal patches through the gate.

After the full source batch passes saved-data validation:

```powershell
& .\.venv\Scripts\python.exe surge_repeatability.py
& .\.venv\Scripts\python.exe -m scripts.validate_selection data\repeatable_ten
```

The selector first screens all three existing repeats. It then renders each
shortlisted patch's full grid in **two fresh Python processes**, each creating a
new plugin instance. It compares all pairs among canonical repeat zero and both
fresh runs using the same limits, with additional tail-isolation checks.
Passing here establishes measured repeatability for this engine and grid, not
universal determinism under every environment or host setting.

`data/repeatable_ten/selection.json` lists accepted and excluded patches, worst-case
errors, failure locations, hashes, and the fresh-process evidence. Each accepted
patch references **one canonical feature grid and 36 canonical WAVs** from repeat
zero. Resolve a recording's relative `path` against its `variant_directory`.
There is no repeat dimension: the training collection is
`patch x pitch x velocity x feature x time`.

The original and repeated diagnostic data remain untouched. Rejected patches do
not appear in `accepted`; they are not silently averaged into apparently stable
examples. Different patch variants and verification repeats must still remain in
the same dataset split if reused for later experiments.

`--resume` skips complete variants only when source, code, plugin, state hashes,
and numerical thread settings match. An incomplete variant requires inspection and deliberate relocation before
resuming; existing results are never silently overwritten.

## Scope and next experiment

No DNN/CNN, learned embedding, patch inference, or inverse synthesis is trained
yet. One patch validates the acquisition pipeline; it cannot establish a latent
space that distinguishes patches.

Next, compare the paired multi-patch representations and evaluate embeddings with patch-held-out
splits. Fit any dataset normalization on the training split only; do not remove
velocity-dependent loudness independently from each example.

The saved-dataset validator is specific to this simple initial patch: its pitch
test expects the fundamental to be the strongest sustained spectral peak.
Arbitrary patches may intentionally emphasize a harmonic or transpose oscillators,
so that test must not be blindly applied to them.

## Sources

- [Official Surge XT downloads](https://surge-synthesizer.github.io/downloads/)
- [Official native Python bindings](https://github.com/surge-synthesizer/surge#building-and-using-the-python-bindings)
- [Official Python examples](https://github.com/surge-synthesizer/surge-python)
- [Pedalboard instrument rendering API](https://spotify.github.io/pedalboard/reference/pedalboard.html#pedalboard.ExternalPlugin.process)
- [Windows VST3 bundle-path issue](https://github.com/spotify/pedalboard/issues/433)
- [Surge XT 1.3.4 patch loading](https://github.com/surge-synthesizer/surge/blob/f7b97c682ade0b87da85ca5968b63d5c7c98e68d/src/common/SurgeSynthesizerIO.cpp#L421)
- [Surge XT 1.3.4 FX recreation](https://github.com/surge-synthesizer/surge/blob/f7b97c682ade0b87da85ca5968b63d5c7c98e68d/src/common/SurgeSynthesizer.cpp#L2991)
- [Surge native patch serialization and modulation migration](https://github.com/surge-synthesizer/surge/blob/f7b97c682ade0b87da85ca5968b63d5c7c98e68d/src/common/SurgePatch.cpp)
- [Surge modulation-source IDs](https://github.com/surge-synthesizer/surge/blob/f7b97c682ade0b87da85ca5968b63d5c7c98e68d/src/common/ModulationSource.h)
- [JUCE MemoryBlock state encoding](https://github.com/juce-framework/JUCE/blob/7.0.12/modules/juce_core/memory/juce_MemoryBlock.cpp)
