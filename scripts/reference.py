"""Full librosa mel spectrogram reference pipeline.

Runs the complete pipeline via ``librosa.feature.melspectrogram`` (one call)
and step-by-step (matching the three functions in ``src/main.c``), then
asserts both produce the same result.

This script is an external/reference validation aid and is not the source used
to generate exam/local-judge ground-truth files.

Run with::

    uv run python scripts/reference.py
"""

import librosa
import numpy as np


def main():
    audio, _ = librosa.load(librosa.example("trumpet"), sr=16000, duration=1.0)

    # ── Version 1: full pipeline ──────────────────────────────────────────────
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=16000,
        n_fft=512,
        hop_length=160,
        n_mels=40,
        center=True,
        power=2.0,
    )

    # ── Version 2: step-by-step (mirrors src/main.c) ─────────────────────────

    # Problem 1 — fft()
    # librosa.stft applies a Hann window then calls numpy.fft.fft per frame.
    D = librosa.stft(audio, n_fft=512, hop_length=160, center=True)

    # Problem 2 — power_spectrum()
    # power[bin, frame] = re² + im²
    power = np.abs(D) ** 2  # shape (257, num_frames)

    # Problem 3 — mel_filter_bank()
    # mel_basis @ power: (40, 257) @ (257, num_frames) → (40, num_frames)
    mel_basis = librosa.filters.mel(sr=16000, n_fft=512, n_mels=40)
    mel_spec = mel_basis @ power

    np.testing.assert_allclose(mel_spec, S, rtol=1e-4, atol=1e-5)

    print(f"OK — shape {S.shape}, max diff {np.max(np.abs(mel_spec - S)):.2e}")


if __name__ == "__main__":
    main()
