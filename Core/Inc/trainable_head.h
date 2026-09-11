#ifndef TRAINABLE_HEAD_H
#define TRAINABLE_HEAD_H

#include <stdint.h>
#include <stddef.h>
#include "tx_api.h"

// Base Model
#define HEAD_NUM_CLASSES        4
#define HEAD_CNN_FEATURES      12

// Extra Head
#define NUM_NEW_FEATURES 1
#define FEAT_GYRO       HEAD_CNN_FEATURES        // Gyroscope: index 12
// #define FEAT_PRESSURE   (HEAD_CNN_FEATURES + 1)  // Pressure: index 13

// Final Model
#define HEAD_TOTAL_FEATURES    (HEAD_CNN_FEATURES + NUM_NEW_FEATURES)
#define HEAD_FLAT_SIZE          (HEAD_NUM_CLASSES * HEAD_TOTAL_FEATURES + HEAD_NUM_CLASSES)


#define FL_PAYLOAD_FLOATS HEAD_FLAT_SIZE

#ifdef __cplusplus
extern "C" {
#endif

// Mutex-protected API
void  trainable_head_init(TX_MUTEX *shared_mutex);
void  trainable_head_forward(const float x[HEAD_TOTAL_FEATURES], float probs[HEAD_NUM_CLASSES]);
float trainable_head_train_step(const float x[HEAD_TOTAL_FEATURES], uint8_t label, float lr);

// Serialization for MQTT/FedAvg payload — HEAD_FLAT_SIZE floats (HEAD_FLAT_SIZE * sizeof(float) bytes)
void  trainable_head_export_weights(float *flat_buffer);
void  trainable_head_import_weights(const float *flat_buffer);

// Getters & Setters
size_t trainable_head_flat_size(void);

#ifdef __cplusplus
}
#endif

#endif // TRAINABLE_HEAD_H