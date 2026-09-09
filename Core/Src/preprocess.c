/**
  ******************************************************************************
  * @file    preprocess.c
  * @brief   Gravity rotation + suppression filter for HAR model input.
  *
  * Adapted from STMicroelectronics stm32ai-modelzoo-services
  * (application_code/sensing/STM32U5/Dpu/Src/filter_gravity.c),
  * Copyright (c) 2021 STMicroelectronics, BSD-3-Clause.
  *
  * Difference to the original: the filter state is owned by the caller's
  * window instead of a translation-unit static, and is re-initialized for
  * every window. This mirrors the Python training pipeline
  * (stm32ai-modelzoo-services human_activity_recognition
  * tf/src/preprocessing/preprocessing.py), which applies the filter to
  * each training segment independently.
  ******************************************************************************
  */

#include "preprocess.h"

#include <math.h>

#define FILT_ORDER 4U

typedef struct
{
  float z[FILT_ORDER];
} iir_state_t;

/*
 * butter(4, 0.4/26, 'highpass') — identical coefficients are hard-coded in
 * ST's training pipeline for all datasets, so they are not a tunable here.
 */
static const float kHighPassA[FILT_ORDER + 1U] =
{
  1.0f, -3.868656635f, 5.614526749f, -3.622760773f, 0.8768966198f
};

static const float kHighPassB[FILT_ORDER + 1U] =
{
  0.9364275932f, -3.745710373f, 5.618565559f, -3.745710373f, 0.9364275932f
};

/* scipy lfilter_zi(b, a): steady-state delays for a unit step input */
static const float kHighPassInit[FILT_ORDER] =
{
  -0.936528250873f, 2.809571532101f, -2.809559172096f, 0.936515859573f
};

static void iir_init(iir_state_t *s, float first)
{
  float scale = (first == 0.0f) ? 1.0f : first;

  for (unsigned int i = 0U; i < FILT_ORDER; ++i)
  {
    s->z[i] = scale * kHighPassInit[i];
  }
}

/* Transposed direct-form II step, as scipy.signal.lfilter */
static float iir_step(iir_state_t *s, float x)
{
  float y = kHighPassB[0] * x + s->z[0];

  for (unsigned int i = 1U; i < FILT_ORDER; ++i)
  {
    s->z[i - 1U] = s->z[i] + kHighPassB[i] * x - kHighPassA[i] * y;
  }
  s->z[FILT_ORDER - 1U] = kHighPassB[FILT_ORDER] * x - kHighPassA[FILT_ORDER] * y;

  return y;
}

void HAR_PreprocessWindow(const float *in, float *out, unsigned int n_samples)
{
  iir_state_t fx, fy, fz;

  if (n_samples == 0U)
  {
    return;
  }

  iir_init(&fx, in[0]);
  iir_init(&fy, in[1]);
  iir_init(&fz, in[2]);

  for (unsigned int i = 0U; i < n_samples; ++i)
  {
    float acc_x = in[i * HAR_AXES];
    float acc_y = in[i * HAR_AXES + 1U];
    float acc_z = in[i * HAR_AXES + 2U];

    /* dynamic component: high-pass filtered acceleration */
    float dyn_x = iir_step(&fx, acc_x);
    float dyn_y = iir_step(&fy, acc_y);
    float dyn_z = iir_step(&fz, acc_z);

    /* gravity versor: the slow-varying remainder, normalized */
    float grav_x = acc_x - dyn_x;
    float grav_y = acc_y - dyn_y;
    float grav_z = acc_z - dyn_z;

    float inv_norm = 1.0f / sqrtf(grav_x * grav_x + grav_y * grav_y + grav_z * grav_z);
    grav_x *= inv_norm;
    grav_y *= inv_norm;
    grav_z *= inv_norm;

    /* rotation aligning gravity with the z axis (Rodrigues' formula) */
    float sin_theta = sqrtf(1.0f - grav_z * grav_z);
    float cos_theta = -grav_z;

    float v_x = 0.0f;
    float v_y = 0.0f;
    if (sin_theta != 0.0f)
    {
      /* rotation axis: cross(z, gravity) / sin */
      v_x = -grav_y / sin_theta;
      v_y = grav_x / sin_theta;
    }
    float v_factor = (v_x * dyn_x + v_y * dyn_y) * (1.0f - cos_theta);

    out[i * HAR_AXES]      = dyn_x * cos_theta + v_y * dyn_z * sin_theta + v_x * v_factor;
    out[i * HAR_AXES + 1U] = dyn_y * cos_theta - v_x * dyn_z * sin_theta + v_y * v_factor;
    out[i * HAR_AXES + 2U] = dyn_z * cos_theta + (v_x * dyn_y - v_y * dyn_x) * sin_theta;
  }
}
