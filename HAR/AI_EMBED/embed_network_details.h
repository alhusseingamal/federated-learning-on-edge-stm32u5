/**
  ******************************************************************************
  * @file    embed_network.h
  * @date    2026-09-06T22:41:06+0200
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */
#ifndef STAI_EMBED_NETWORK_DETAILS_H
#define STAI_EMBED_NETWORK_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_embed_network_details = {
  .tensors = (const stai_tensor[6]) {
   { .size_bytes = 288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 24, 3, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "input_tensor_output" },
   { .size_bytes = 288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 24, 3, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "functional_1_conv2d_1_BiasAdd__70_to_chfirst_output" },
   { .size_bytes = 2592, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 9, 3, 24}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "functional_1_conv2d_1_BiasAdd0_output" },
   { .size_bytes = 2592, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 9, 3, 24}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "functional_1_activation_1_Relu0_output" },
   { .size_bytes = 864, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 3, 3, 24}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "functional_1_max_pooling2d_1_MaxPool2d0_output" },
   { .size_bytes = 48, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 12}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "Identity0_output" }
  },
  .nodes = (const stai_node_details[5]){
    {.id = 1, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* functional_1_conv2d_1_BiasAdd__70_to_chfirst */
    {.id = 2, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* functional_1_conv2d_1_BiasAdd0 */
    {.id = 3, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* functional_1_activation_1_Relu0 */
    {.id = 4, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* functional_1_max_pooling2d_1_MaxPool2d0 */
    {.id = 7, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} } /* Identity0 */
  },
  .n_nodes = 5
};
#endif

