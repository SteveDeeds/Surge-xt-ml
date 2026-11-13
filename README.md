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

Pedalboard accepts `.vstpreset`, not Surge's factory `.fxp` files. Native `.fxp`
loading and deeper modulation control are reasons to consider the official
`surgepy` bindings later; these require a native source build.

## Scope and next experiment

No DNN/CNN, learned embedding, patch inference, or inverse synthesis is trained
yet. One patch validates the acquisition pipeline; it cannot establish a latent
space that distinguishes patches.

Next, collect multiple patches (including velocity-sensitive and long-FX patches),
compare these representations, and evaluate embeddings with patch-held-out
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
