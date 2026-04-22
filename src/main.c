/**
 * @file main.c
 * @brief Student implementation of mel spectrogram core functions.
 *
 * Implement the three functions below. The remaining pipeline functions
 * (hann_window, stft, melspectrogram) are provided in utils.c.
 *
 * Python source of truth: scripts/mel_spectrogram.py
 * Include RVV intrinsics via: #include <riscv_vector.h>
 */

#include "mel_spectrogram.h"
#include "riscv_vector.h"
#include "math.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* RVV hint: each butterfly stage is data-parallel across independent pairs. */
void fft(float *__restrict real, float *__restrict imag, size_t n) {
    for(size_t i=1, j=0 ; i<n ; i++){
        size_t bit=n>>1;
        for(;j & bit;bit>>=1) j^=bit;
        j^=bit;
        if(i<j){
            float temp_re = real[i];
            real[i]=real[j];
            real[j]=temp_re;
            float temp_im=imag[i];
            imag[i]=imag[j];
            imag[j]=temp_im;
        }
    }

    float re_table[2048];
    float im_table[2048];

    for(size_t len=2; len<=n; len<<=1){
        size_t half_len=len/2;

        double angle_step=(-2*M_PI)/len;
        for(size_t j=0;j<half_len;j++){
            double angle=angle_step*(double)j;
            re_table[j]=(float)cos(angle);
            im_table[j]=(float)sin(angle);
        }

        for(size_t i=0;i<n;i+=len){
            float* p_re=real+i;
            float* p_im=imag+i;
            float* p_re_half=real+i+half_len;
            float* p_im_half=imag+i+half_len;
            for(size_t j=0;j<half_len;){
                size_t vl=__riscv_vsetvl_e32m8(half_len-j);

                vfloat32m8_t u_re=__riscv_vle32_v_f32m8(p_re+j, vl);
                vfloat32m8_t u_im=__riscv_vle32_v_f32m8(p_im+j, vl);
                vfloat32m8_t v_re=__riscv_vle32_v_f32m8(p_re_half+j,vl);
                vfloat32m8_t v_im=__riscv_vle32_v_f32m8(p_im_half+j,vl);

                vfloat32m8_t w_re=__riscv_vle32_v_f32m8(re_table+j,vl);
                vfloat32m8_t w_im=__riscv_vle32_v_f32m8(im_table+j,vl);

                vfloat32m8_t t_re=__riscv_vfmul_vv_f32m8(v_re,w_re,vl);//(a+bj)*(c+dj)=(ac-bd)+j(ad+bc)
                t_re=__riscv_vfnmsac_vv_f32m8(t_re,v_im,w_im,vl);

                vfloat32m8_t t_im=__riscv_vfmul_vv_f32m8(v_re,w_im,vl);
                t_im=__riscv_vfmacc_vv_f32m8(t_im, v_im,w_re,vl);

                vfloat32m8_t res_re1=__riscv_vfadd_vv_f32m8(u_re,t_re,vl);
                vfloat32m8_t res_im1=__riscv_vfadd_vv_f32m8(u_im,t_im,vl);
                vfloat32m8_t res_re2=__riscv_vfsub_vv_f32m8(u_re,t_re,vl);
                vfloat32m8_t res_im2=__riscv_vfsub_vv_f32m8(u_im,t_im,vl);

                __riscv_vse32_v_f32m8(p_re+j,res_re1,vl);
                __riscv_vse32_v_f32m8(p_im+j,res_im1,vl);
                __riscv_vse32_v_f32m8(p_re_half+j,res_re2,vl);
                __riscv_vse32_v_f32m8(p_im_half+j,res_im2,vl);

                j+=vl;
            }
        }
    }
}

/* RVV hint: vlse32 with stride=8 bytes extracts all re (or im) values in one pass. */
void power_spectrum(const float *__restrict stft_data, size_t num_frames, float *__restrict output) {
    size_t total_element=num_frames*N_FREQ_BINS;
    
    for(size_t i=0;i<total_element;){
        size_t vl=__riscv_vsetvl_e32m8(total_element-i);
        
        vfloat32m8_t v_re=__riscv_vlse32_v_f32m8(&stft_data[i*2],8,vl);
        vfloat32m8_t v_im=__riscv_vlse32_v_f32m8(&stft_data[i*2+1],8,vl);

        vfloat32m8_t v_re2=__riscv_vfmul_vv_f32m8(v_re,v_re,vl);
        vfloat32m8_t power=__riscv_vfmacc_vv_f32m8(v_re2, v_im,v_im,vl);

        __riscv_vse32_v_f32m8(&output[i], power,vl);

        i+=vl;
    }
}

/* RVV hint: vfmul + vfredusum computes one dot product per (frame, mel) pair. */
void mel_filter_bank(const float *__restrict power,
                     const float *__restrict mel_bank, size_t num_frames,
                     size_t n_mels, size_t n_freq_bins,
                     float *__restrict output) {
        vfloat32m1_t v_zero=__riscv_vfmv_v_f_f32m1(0.0f,__riscv_vsetvlmax_e32m1());
        for(size_t f=0;f<num_frames;f++){
            const float* p_frames=&power[f*n_freq_bins];
            size_t m=0;
            for(;m+3<n_mels;m+=4){
                const float* row0=&mel_bank[(m+0)*n_freq_bins];
                const float* row1=&mel_bank[(m+1)*n_freq_bins];
                const float* row2=&mel_bank[(m+2)*n_freq_bins];
                const float* row3=&mel_bank[(m+3)*n_freq_bins];
                
                float sum0=0.0f, sum1=0.0f, sum2=0.0f, sum3=0.0f;
                
                for(size_t k=0;k<n_freq_bins;){
                    size_t vl=__riscv_vsetvl_e32m8(n_freq_bins-k);
                    
                    vfloat32m8_t v_frames=__riscv_vle32_v_f32m8(&p_frames[k],vl);
                    vfloat32m8_t v_row0=__riscv_vle32_v_f32m8(&row0[k],vl);
                    vfloat32m8_t v_row1=__riscv_vle32_v_f32m8(&row1[k],vl);
                    vfloat32m8_t v_row2=__riscv_vle32_v_f32m8(&row2[k],vl);
                    vfloat32m8_t v_row3=__riscv_vle32_v_f32m8(&row3[k],vl);

                    vfloat32m8_t product0=__riscv_vfmul_vv_f32m8(v_frames,v_row0,vl);
                    vfloat32m8_t product1=__riscv_vfmul_vv_f32m8(v_frames,v_row1,vl);
                    vfloat32m8_t product2=__riscv_vfmul_vv_f32m8(v_frames,v_row2,vl);
                    vfloat32m8_t product3=__riscv_vfmul_vv_f32m8(v_frames,v_row3,vl);
                    
                    sum0+=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m8_f32m1(product0, v_zero, vl));
                    sum1+=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m8_f32m1(product1, v_zero, vl));
                    sum2+=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m8_f32m1(product2, v_zero, vl));
                    sum3+=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m8_f32m1(product3, v_zero, vl));
                    
                    k+=vl;
                }
                output[f*n_mels+m+0]=sum0;
                output[f*n_mels+m+1]=sum1;
                output[f*n_mels+m+2]=sum2;
                output[f*n_mels+m+3]=sum3;
            }
            for(;m<n_mels;m++){
                const float* row0=&mel_bank[(m+0)*n_freq_bins];
                float sum0=0.0f;
                for(size_t k=0;k<n_freq_bins;){
                    size_t vl=__riscv_vsetvl_e32m8(n_freq_bins-k);
                    
                    vfloat32m8_t v_frames=__riscv_vle32_v_f32m8(&p_frames[k],vl);
                    vfloat32m8_t v_row0=__riscv_vle32_v_f32m8(&row0[k],vl);
                    vfloat32m8_t product0=__riscv_vfmul_vv_f32m8(v_frames,v_row0,vl);
                    sum0+=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m8_f32m1(product0, v_zero, vl));
                }
                output[f*n_mels+m+0]=sum0;
            }
        }
}
