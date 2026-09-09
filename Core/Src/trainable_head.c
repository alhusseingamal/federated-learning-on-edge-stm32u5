
#include "trainable_head.h"
#include "head_weights_init.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

static float W[HEAD_NUM_CLASSES][HEAD_TOTAL_FEATURES];
static float b[HEAD_NUM_CLASSES];
static TX_MUTEX *head_mutex = NULL;

void trainable_head_init(TX_MUTEX *shared_mutex) {
    head_mutex = shared_mutex;

    for (uint32_t c = 0; c < HEAD_NUM_CLASSES; c++) {
        b[c] = BASELINE_B_INIT[c];
        
        // 12 base CNN features directly from pretrained dense_1
        for (uint32_t f = 0; f < HEAD_CNN_FEATURES; f++) {
            W[c][f] = BASELINE_W_INIT[c][f];
        }
        
        // Zero-initialize secondary sensor feature column
        for (uint32_t f = HEAD_CNN_FEATURES; f < HEAD_TOTAL_FEATURES; f++) {
            W[c][f] = 0.0f;
        }
    }
}

void trainable_head_forward(const float x[HEAD_TOTAL_FEATURES], float probs[HEAD_NUM_CLASSES]) {
    if (head_mutex) tx_mutex_get(head_mutex, TX_WAIT_FOREVER);

    // 1. Separate array for raw linear logits (z = W*x + b)
    float logits[HEAD_NUM_CLASSES];
    float max_logit = -1e9f;

    for (uint32_t c = 0; c < HEAD_NUM_CLASSES; c++) {
        logits[c] = b[c];
        for (uint32_t f = 0; f < HEAD_TOTAL_FEATURES; f++) {
            logits[c] += W[c][f] * x[f];
        }
        if (logits[c] > max_logit) {
            max_logit = logits[c];
        }
    }

    // 2. Numerically stable Softmax: compute exp(z - max_z) into probs[]
    float sum_exp = 0.0f;
    for (uint32_t c = 0; c < HEAD_NUM_CLASSES; c++) {
        probs[c] = expf(logits[c] - max_logit);
        sum_exp += probs[c];
    }

    // 3. Normalize probabilities
    float inv_sum = 1.0f / (sum_exp > 1e-7f ? sum_exp : 1e-7f);
    for (uint32_t c = 0; c < HEAD_NUM_CLASSES; c++) {
        probs[c] *= inv_sum;
    }

    if (head_mutex) tx_mutex_put(head_mutex);
}

float trainable_head_train_step(const float x[HEAD_TOTAL_FEATURES], uint8_t label, float lr) {
    if (head_mutex) tx_mutex_get(head_mutex, TX_WAIT_FOREVER);

    float logits[HEAD_NUM_CLASSES];
    float probs[HEAD_NUM_CLASSES];
    float max_logit = -1e9f;

    // 1. Compute raw logits
    for (uint32_t c = 0; c < HEAD_NUM_CLASSES; c++) {
        logits[c] = b[c];
        for (uint32_t f = 0; f < HEAD_TOTAL_FEATURES; f++) {
            logits[c] += W[c][f] * x[f];
        }
        if (logits[c] > max_logit) {
            max_logit = logits[c];
        }
    }

    // 2. Softmax probabilities
    float sum_exp = 0.0f;
    for (uint32_t c = 0; c < HEAD_NUM_CLASSES; c++) {
        probs[c] = expf(logits[c] - max_logit);
        sum_exp += probs[c];
    }
    float inv_sum = 1.0f / (sum_exp > 1e-7f ? sum_exp : 1e-7f);
    for (uint32_t c = 0; c < HEAD_NUM_CLASSES; c++) {
        probs[c] *= inv_sum;
    }

    // 3. Loss = -ln(P(target))
    float p_target = probs[label] > 1e-7f ? probs[label] : 1e-7f;
    float loss = -logf(p_target);

    // 4. Closed-form gradient update: dL/dz = probs - one_hot
    for (uint32_t c = 0; c < HEAD_NUM_CLASSES; c++) {
        float y_true = (c == label) ? 1.0f : 0.0f;
        float dlogit = probs[c] - y_true;

        b[c] -= lr * dlogit;
        for (uint32_t f = 0; f < HEAD_TOTAL_FEATURES; f++) {
            W[c][f] -= lr * dlogit * x[f];
        }
    }

    if (head_mutex) tx_mutex_put(head_mutex);
    return loss;
}

void trainable_head_export_weights(float *flat_buffer) {
    if (head_mutex) tx_mutex_get(head_mutex, TX_WAIT_FOREVER);
    memcpy(flat_buffer, W, sizeof(W));
    memcpy(flat_buffer + (HEAD_NUM_CLASSES * HEAD_TOTAL_FEATURES), b, sizeof(b));
    if (head_mutex) tx_mutex_put(head_mutex);
}

void trainable_head_import_weights(const float *flat_buffer) {
    if (head_mutex) tx_mutex_get(head_mutex, TX_WAIT_FOREVER);
    memcpy(W, flat_buffer, sizeof(W));
    memcpy(b, flat_buffer + (HEAD_NUM_CLASSES * HEAD_TOTAL_FEATURES), sizeof(b));
    if (head_mutex) tx_mutex_put(head_mutex);
}


size_t trainable_head_flat_size(void) {
    return HEAD_FLAT_SIZE;
}