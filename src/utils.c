/**
 * @file utils.c
 * @brief Pre-implemented utility functions for mel spectrogram computation.
 *
 * Students should NOT modify this file. It provides hann_window, stft
 * (center=True), and the melspectrogram pipeline that calls into the
 * three student-implemented functions (fft, power_spectrum, mel_filter_bank).
 */

#include "mel_spectrogram.h"
#include <math.h>
#include <string.h>

static float hann_buf[N_FFT];

void mel_init(void) { hann_window(hann_buf, N_FFT); }

void hann_window(float *__restrict window, size_t n) {
  for (size_t i = 0; i < n; i++) {
    window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / n));
  }
}

void stft(const float *__restrict audio, size_t audio_len,
          const float *__restrict window, float *__restrict output) {
  size_t pad = N_FFT / 2;
  size_t num_frames = 1 + audio_len / HOP_LENGTH;
  size_t n_freq_bins = N_FFT / 2 + 1;

  float fft_real[N_FFT], fft_imag[N_FFT];

  for (size_t frame = 0; frame < num_frames; frame++) {
    size_t center_pos = frame * HOP_LENGTH;

    for (size_t i = 0; i < (size_t)N_FFT; i++) {
      size_t padded_index = center_pos + i;
      float sample;
      if (padded_index < pad || padded_index >= pad + audio_len) {
        sample = 0.0f;
      } else {
        sample = audio[padded_index - pad];
      }
      fft_real[i] = sample * window[i];
      fft_imag[i] = 0.0f;
    }

    fft(fft_real, fft_imag, N_FFT);

    size_t out_offset = frame * n_freq_bins * 2;
    for (size_t bin = 0; bin < n_freq_bins; bin++) {
      output[out_offset + bin * 2 + 0] = fft_real[bin];
      output[out_offset + bin * 2 + 1] = fft_imag[bin];
    }
  }
}

void melspectrogram(const float *__restrict audio, size_t audio_len,
                    const float *__restrict mel_bank,
                    float *__restrict output) {
  size_t num_frames = 1 + audio_len / HOP_LENGTH;

  static float stft_buf[MAX_FRAMES * N_FREQ_BINS * 2];
  static float power_buf[MAX_FRAMES * N_FREQ_BINS];

  stft(audio, audio_len, hann_buf, stft_buf);
  power_spectrum(stft_buf, num_frames, power_buf);
  mel_filter_bank(power_buf, mel_bank, num_frames, N_MELS, N_FREQ_BINS, output);
}
