/**
  ******************************************************************************
  * @file    preprocess.h
  * @brief   Gravity rotation + suppression filter for HAR model input.
  *
  * Preprocessing contract of the st_ign model family (ST model zoo,
  * preprocessing option "gravity_rot_sup"): each window of raw 3-axis
  * accelerometer samples in m/s^2 is split into a slow-varying gravity
  * component and a dynamic component by a 4th-order high-pass IIR filter
  * (designed for fs = 26 Hz, fc = 0.4 Hz), then the dynamic component is
  * rotated so that the estimated gravity direction maps onto the z axis
  * and the gravity itself is suppressed from the output.
  *
  * Adapted from STMicroelectronics stm32ai-modelzoo-services
  * (application_code/sensing/STM32U5/Dpu/Src/filter_gravity.c).
  ******************************************************************************
  */

#ifndef PREPROCESS_H
#define PREPROCESS_H

#ifdef __cplusplus
extern "C" {
#endif

#define HAR_AXES 3U

/*
 * Convert one window of raw accelerometer samples into the model input.
 * in / out layout: [n_samples][3] = {x, y, z}, units m/s^2.
 * The IIR filter state is re-initialized from the first sample of the
 * window, matching the per-window preprocessing used at training time.
 * in and out may alias.
 */
void HAR_PreprocessWindow(const float *in, float *out, unsigned int n_samples);

#ifdef __cplusplus
}
#endif

#endif /* HAR_PREPROCESS_H */
