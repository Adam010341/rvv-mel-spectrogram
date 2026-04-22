"""Reference implementation of the three scored mel spectrogram kernels.

This module is the judge ground truth for ``src/main.c``.

It intentionally implements a simplified kernel-level model that is close to
the standard librosa mel-spectrogram pipeline, while remaining easier to audit
and grade for assessment purposes.

For the full librosa pipeline reference, see ``scripts/reference.py``.
"""

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE: int = 16000
N_FFT: int = 512
HOP_LENGTH: int = 160
N_MELS: int = 40
N_FREQ_BINS: int = N_FFT // 2 + 1  # 257


def fft(
    real: NDArray[np.float32],
    imag: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """DFT of `real + j*imag` via numpy. C version operates in-place.

    Args:
        real: Real part of the input signal, shape (n,).
        imag: Imaginary part of the input signal, shape (n,).

    Returns:
        Tuple of (real_out, imag_out), each shape (n,), dtype float32.
    """
    c = real.astype(np.complex128) + 1j * imag.astype(np.complex128)
    result = np.fft.fft(c)
    return result.real.astype(np.float32), result.imag.astype(np.float32)


def power_spectrum(
    stft_data: NDArray[np.float32],
    num_frames: int,
) -> NDArray[np.float32]:
    """Compute `re² + im²` per bin from interleaved STFT data.

    Input memory layout per frame (from `stft` in ``src/utils.c``)::

        [ re[bin=0], im[bin=0], re[bin=1], im[bin=1], ... ]

    Args:
        stft_data: Interleaved complex STFT output, length
            `num_frames * N_FREQ_BINS * 2`.
        num_frames: Number of STFT frames.

    Returns:
        Power values, shape (num_frames * N_FREQ_BINS,), dtype float32.
    """
    total = num_frames * N_FREQ_BINS
    re = stft_data[0::2]
    im = stft_data[1::2]
    return (re * re + im * im)[:total].astype(np.float32)


def mel_filter_bank(
    power: NDArray[np.float32],
    mel_bank: NDArray[np.float32],
    num_frames: int,
) -> NDArray[np.float32]:
    """Apply the mel filter bank: `power @ mel_bank.T`.

    Args:
        power: Power spectrum, shape (num_frames, n_freq_bins), flattened.
        mel_bank: Mel filter matrix, shape (n_mels, n_freq_bins).
        num_frames: Number of STFT frames.

    Returns:
        Mel-filtered output, shape (num_frames, n_mels), flattened, dtype float32.
        Row-major: `output[frame * n_mels + mel]`.
    """
    _, n_freq_bins = mel_bank.shape
    return (
        (power.reshape(num_frames, n_freq_bins) @ mel_bank.T).ravel().astype(np.float32)
    )
