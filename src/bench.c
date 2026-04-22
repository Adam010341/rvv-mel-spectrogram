/**
 * @file bench.c
 * @brief Test runner and benchmark harness for mel spectrogram functions.
 *
 * Reads test data from data/ directory, runs each function under test,
 * measures cycle counts via the RISC-V cycle CSR, and outputs results as CSV.
 *
 * Output format: test_name,pass,cycles
 */

/* Students: you do not need to read or modify this file. */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "mel_spectrogram.h"

#define read_csr(reg)                                                          \
  ({                                                                           \
    uint64_t __tmp;                                                            \
    asm volatile("csrr %0, " #reg : "=r"(__tmp));                              \
    __tmp;                                                                     \
  })
#define rdcycle() read_csr(cycle)

#define MAX_SIZE ((size_t)100000)
#define MAX_TESTS ((size_t)100)

static float audio_buf[MAX_SIZE];
static float expected_buf[MAX_SIZE];
static float stft_buf[MAX_SIZE];
static float output_buf[MAX_SIZE];
static float power_buf[MAX_SIZE];
static float mel_matrix_buf[MAX_SIZE];

typedef struct {
  char name[64];
  bool pass;
  uint64_t cycles;
} TestResult;

/**
 * @brief Read a single integer from @p file, exit on failure.
 *
 * @param file  Input file stream.
 * @return      Parsed integer value.
 */
static int read_int(FILE *file) {
  int value;
  if (fscanf(file, "%d", &value) != 1) {
    fprintf(stderr, "error: failed to read int\n");
    exit(1);
  }
  return value;
}

/**
 * @brief Read a whitespace-delimited string from @p file into @p buf.
 *
 * @param file  Input file stream.
 * @param buf   Destination buffer.
 * @param size  Size of @p buf in bytes.
 */
static void read_name(FILE *file, char *buf, size_t size) {
  char fmt[16];
  snprintf(fmt, sizeof(fmt), "%%%ds", (int)(size - 1));
  if (fscanf(file, fmt, buf) != 1) {
    fprintf(stderr, "error: failed to read name\n");
    exit(1);
  }
}

/**
 * @brief Read a length-prefixed float array from @p file.
 *
 * Format: <length> <val0> <val1> ... <valN>
 *
 * @param      file    Input file stream.
 * @param[out] buf     Destination float buffer (must hold at least MAX_SIZE
 *                     elements).
 * @param[out] length  Number of floats read.
 */
static void read_array(FILE *file, float *buf, size_t *length) {
  size_t len = (size_t)read_int(file);
  if (len > MAX_SIZE) {
    fprintf(stderr, "error: array length %zu out of range\n", len);
    exit(1);
  }
  *length = len;
  for (size_t i = 0; i < len; i++) {
    if (fscanf(file, "%f", &buf[i]) != 1) {
      fprintf(stderr, "error: failed to read element %zu\n", i);
      exit(1);
    }
  }
}

/**
 * @brief Element-wise comparison with combined relative and absolute tolerance.
 *
 * Tolerance per element: diff <= 1e-5 + 1e-4 * |expected|.
 *
 * @param a       First array.
 * @param b       Second array (reference).
 * @param length  Number of elements to compare.
 * @return        true if all elements are within tolerance.
 */
static bool all_close(const float *a, const float *b, size_t length) {
  for (size_t i = 0; i < length; i++) {
    float diff = fabsf(a[i] - b[i]);
    float tolerance = 1e-5f + 1e-4f * fabsf(b[i]);
    if (diff > tolerance)
      return false;
  }
  return true;
}

/**
 * @brief Run FFT test cases from data/fft.txt.
 *
 * File format per case: <name> <n> then length-prefixed interleaved
 * input and expected arrays (each 2*n floats).
 *
 * @param[out] results  Array to append results into.
 * @param[in,out] count Current number of results; incremented per case.
 */
static void test_fft(TestResult *results, size_t *count) {
  FILE *file = fopen("data/fft.txt", "r");
  if (!file) {
    fprintf(stderr, "error: cannot open data/fft.txt");
    exit(1);
  }

  size_t n_cases = (size_t)read_int(file);
  for (size_t i = 0; i < n_cases; i++) {
    TestResult *result = &results[(*count)++];
    read_name(file, result->name, sizeof(result->name));
    size_t n = (size_t)read_int(file);

    size_t input_len;
    read_array(file, audio_buf, &input_len);

    size_t expected_len;
    read_array(file, expected_buf, &expected_len);

    static float fft_real[MAX_FFT_N], fft_imag[MAX_FFT_N];
    for (size_t j = 0; j < n; j++) {
      fft_real[j] = audio_buf[j * 2];
      fft_imag[j] = audio_buf[j * 2 + 1];
    }

    uint64_t t0 = rdcycle();
    fft(fft_real, fft_imag, n);
    result->cycles = rdcycle() - t0;

    for (size_t j = 0; j < n; j++) {
      output_buf[j * 2] = fft_real[j];
      output_buf[j * 2 + 1] = fft_imag[j];
    }

    result->pass = all_close(output_buf, expected_buf, expected_len);
  }

  fclose(file);
}

/**
 * @brief Run power spectrum test cases from data/power_spectrum.txt.
 *
 * File format per case: <name> <num_frames> then length-prefixed
 * stft and expected arrays.
 *
 * @param[out] results  Array to append results into.
 * @param[in,out] count Current number of results; incremented per case.
 */
static void test_power_spectrum(TestResult *results, size_t *count) {
  FILE *file = fopen("data/power_spectrum.txt", "r");
  if (!file) {
    fprintf(stderr, "error: cannot open data/power_spectrum.txt");
    exit(1);
  }

  size_t n_cases = (size_t)read_int(file);
  for (size_t i = 0; i < n_cases; i++) {
    TestResult *result = &results[(*count)++];
    read_name(file, result->name, sizeof(result->name));
    size_t frames = (size_t)read_int(file);

    size_t stft_len, expected_len;
    read_array(file, stft_buf, &stft_len);
    read_array(file, expected_buf, &expected_len);

    uint64_t t0 = rdcycle();
    power_spectrum(stft_buf, frames, output_buf);
    result->cycles = rdcycle() - t0;

    result->pass = all_close(output_buf, expected_buf, expected_len);
  }

  fclose(file);
}

/**
 * @brief Run mel filter bank test cases from data/mel_filter_bank.txt.
 *
 * File format per case: <name> <frames> <n_mels> <n_bins> then
 * length-prefixed power, expected, and mel_matrix arrays.
 *
 * @param[out] results  Array to append results into.
 * @param[in,out] count Current number of results; incremented per case.
 */
static void test_mel_filter_bank(TestResult *results, size_t *count) {
  FILE *file = fopen("data/mel_filter_bank.txt", "r");
  if (!file) {
    fprintf(stderr, "error: cannot open data/mel_filter_bank.txt");
    exit(1);
  }

  size_t n_cases = (size_t)read_int(file);
  for (size_t i = 0; i < n_cases; i++) {
    TestResult *result = &results[(*count)++];
    read_name(file, result->name, sizeof(result->name));
    size_t frames = (size_t)read_int(file);
    size_t n_mels = (size_t)read_int(file);
    size_t n_bins = (size_t)read_int(file);

    size_t power_len, expected_len, mel_len;
    read_array(file, power_buf, &power_len);
    read_array(file, expected_buf, &expected_len);
    read_array(file, mel_matrix_buf, &mel_len);

    uint64_t t0 = rdcycle();
    mel_filter_bank(power_buf, mel_matrix_buf, frames, n_mels, n_bins,
                    output_buf);
    result->cycles = rdcycle() - t0;

    result->pass = all_close(output_buf, expected_buf, expected_len);
  }

  fclose(file);
}

/**
 * @brief Run mel spectrogram end-to-end test cases from
 * data/mel_spectrogram.txt.
 *
 * File format per case: <name> <n_mels> then length-prefixed audio,
 * expected, and mel_matrix arrays.
 *
 * @param[out] results  Array to append results into.
 * @param[in,out] count Current number of results; incremented per case.
 */
static void test_melspectrogram(TestResult *results, size_t *count) {
  FILE *file = fopen("data/mel_spectrogram.txt", "r");
  if (!file) {
    fprintf(stderr, "error: cannot open data/mel_spectrogram.txt");
    exit(1);
  }

  size_t n_cases = (size_t)read_int(file);
  for (size_t i = 0; i < n_cases; i++) {
    TestResult *result = &results[(*count)++];
    read_name(file, result->name, sizeof(result->name));
    size_t n_mels = (size_t)read_int(file);
    (void)n_mels;

    size_t audio_len, expected_len, mel_len;
    read_array(file, audio_buf, &audio_len);
    read_array(file, expected_buf, &expected_len);
    read_array(file, mel_matrix_buf, &mel_len);

    size_t n_bins = N_FFT / 2 + 1;
    (void)n_bins;

    uint64_t t0 = rdcycle();
    melspectrogram(audio_buf, audio_len, mel_matrix_buf, output_buf);
    result->cycles = rdcycle() - t0;

    result->pass = all_close(output_buf, expected_buf, expected_len);
  }

  fclose(file);
}

/**
 * @brief Print test results as CSV to stdout.
 *
 * @param results  Array of test results.
 * @param n        Number of results.
 */
static void print_csv(const TestResult *results, size_t n) {
  printf("test_name,pass,cycles\n");
  for (size_t i = 0; i < n; i++) {
    printf("%s,%d,%lu\n", results[i].name, results[i].pass ? 1 : 0,
           (unsigned long)results[i].cycles);
  }
}

int main(void) {
  mel_init();

  static TestResult results[MAX_TESTS];
  size_t count = 0;

  test_fft(results, &count);
  test_power_spectrum(results, &count);
  test_mel_filter_bank(results, &count);
  test_melspectrogram(results, &count);

  print_csv(results, count);
  return 0;
}
