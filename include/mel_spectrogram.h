#ifndef MEL_SPECTROGRAM_H
#define MEL_SPECTROGRAM_H

#include <stddef.h>
#include <stdint.h>

#define SAMPLE_RATE 16000
#define N_FFT 512
#define HOP_LENGTH 160
#define N_MELS 40
#define N_FREQ_BINS (N_FFT / 2 + 1)
#define MAX_FRAMES 200
#define MAX_FFT_N 4096

/**
 * @brief Cooley-Tukey radix-2 FFT, computed in-place.
 *
 * @param[in,out] real  Real part array of length @p n.
 * @param[in,out] imag  Imaginary part array of length @p n.
 * @param         n     Number of complex samples (must be a power of 2).
 */
void fft(float *__restrict real, float *__restrict imag, size_t n);

/**
 * @brief Compute power spectrum from interleaved complex STFT output.
 *
 * For each bin: output[i] = real[i]^2 + imag[i]^2.
 * Input is interleaved [re0, im0, re1, im1, ...] per frame.
 *
 * @param[in]  stft_data   Interleaved complex STFT data, length
 *                         num_frames * N_FREQ_BINS * 2.
 * @param      num_frames  Number of STFT frames.
 * @param[out] output      Power values, length num_frames * N_FREQ_BINS.
 */
void power_spectrum(const float *__restrict stft_data, size_t num_frames,
                    float *__restrict output);

/**
 * @brief Apply mel filter bank via matrix multiplication.
 *
 * For each frame: output[frame][mel] = sum over freq of
 * mel_bank[mel][freq] * power[frame][freq].
 *
 * @param[in]  power       Power spectrum, row-major (num_frames, N_FREQ_BINS).
 * @param[in]  mel_bank    Mel filter bank matrix, row-major (n_mels,
 * n_freq_bins).
 * @param      num_frames  Number of frames.
 * @param      n_mels      Number of mel bands (rows of mel_bank).
 * @param      n_freq_bins Number of frequency bins (cols of mel_bank).
 * @param[out] output      Mel-filtered output, length num_frames * n_mels.
 */
void mel_filter_bank(const float *__restrict power,
                     const float *__restrict mel_bank, size_t num_frames,
                     size_t n_mels, size_t n_freq_bins,
                     float *__restrict output);

/**
 * @brief Generate a periodic Hann window.
 *
 * w[i] = 0.5 * (1 - cos(2 * pi * i / n)).
 *
 * @param[out] window  Output array of length @p n.
 * @param      n       Window size.
 */
void hann_window(float *__restrict window, size_t n);

/**
 * @brief Short-Time Fourier Transform with center=True zero-padding.
 *
 * Pads the signal with N_FFT/2 zeros on each side before framing.
 * num_frames = 1 + audio_len / HOP_LENGTH.
 *
 * @param[in]  audio      Input audio samples.
 * @param      audio_len  Length of audio array.
 * @param[in]  window     Hann window of length N_FFT.
 * @param[out] output     Interleaved complex output, length
 *                        num_frames * N_FREQ_BINS * 2.
 */
void stft(const float *__restrict audio, size_t audio_len,
          const float *__restrict window, float *__restrict output);

/**
 * @brief Full mel spectrogram pipeline: hann -> stft -> power -> mel filter.
 *
 * @param[in]  audio      Input audio samples.
 * @param      audio_len  Length of audio array.
 * @param[in]  mel_bank   Mel filter bank matrix, row-major (N_MELS,
 * N_FREQ_BINS).
 * @param[out] output     Mel spectrogram output, length num_frames * N_MELS.
 */
void melspectrogram(const float *__restrict audio, size_t audio_len,
                    const float *__restrict mel_bank, float *__restrict output);

void mel_init(void);

#endif
