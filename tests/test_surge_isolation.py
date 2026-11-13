import numpy as np
import pytest
from pedalboard import VST3Plugin

from surge_dataset import (
    DEFAULT_PLUGIN, PRE_ROLL_SECONDS, SAMPLE_RATE, render_note,
)


@pytest.mark.skipif(not DEFAULT_PLUGIN.exists(), reason="Local Surge XT installation required")
def test_full_patch_restore_clears_audible_reverb_tail():
    plugin = VST3Plugin(str(DEFAULT_PLUGIN))
    plugin.fx_g1_fx_type = "Reverb 1"
    plugin.process(
        [], duration=PRE_ROLL_SECONDS, sample_rate=SAMPLE_RATE,
        num_channels=2, buffer_size=512, reset=True,
    )
    state = plugin.preset_data
    audio = render_note(plugin, state, 60, 96)
    remaining_rms = float(np.sqrt(np.mean(audio[:, -4800:] ** 2)))
    assert remaining_rms > 1e-5, "Test patch must have an audible reverb tail at 2 s"

    plugin.preset_data = state
    plugin.process(
        [], duration=PRE_ROLL_SECONDS, sample_rate=SAMPLE_RATE,
        num_channels=2, buffer_size=512, reset=True,
    )
    silence = plugin.process(
        [], duration=2.0, sample_rate=SAMPLE_RATE,
        num_channels=2, buffer_size=512, reset=False,
    )
    residual_peak = float(np.max(np.abs(silence)))
    assert residual_peak < 1e-7, "Previous note/reverb leaked past full patch restoration"
    print(f"Reverb RMS before restore={remaining_rms:.8g}; residual peak={residual_peak:.8g}")
