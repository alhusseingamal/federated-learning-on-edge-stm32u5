/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    app_threadx.c
  * @author  MCD Application Team
  * @brief   ThreadX applicative file
  ******************************************************************************
    * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "app_threadx.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

#include "b_u585i_iot02a.h"

#include "b_u585i_iot02a_motion_sensors.h"
#include "ism330dhcx.h"

#include "b_u585i_iot02a_env_sensors.h"

// NN
// #include "network.h"
// #include "network_data.h"
#include "preprocess.h"
#include "embed_network.h"
#include "embed_network_data.h"
#include "trainable_head.h"

#include "tx_api.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
extern UART_HandleTypeDef huart1;

extern TX_SEMAPHORE fl_round_done_sem;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
#define LOG_QUEUE_DEPTH 16        // number of messages the queue can hold
#define LOG_MSG_WORDS 16          // message size in ULONG units
#define CMD_BUF_LEN 32            // command thread buffer length


#define SENSOR_SAMPLE_RATE_HZ  26

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
TX_THREAD tx_app_thread;
/* USER CODE BEGIN PV */
TX_THREAD press_thread;
TX_THREAD logger_thread;
TX_THREAD command_thread;
TX_THREAD weights_thread;


UCHAR *press_thread_stack_ptr;
UCHAR *logger_thread_stack_ptr;
UCHAR *command_thread_stack_ptr;
UCHAR *weights_thread_stack_ptr;


TX_QUEUE log_queue;
ULONG *queue_storage;

TX_MUTEX i2c_mutex;     // Shared across all I2C2 sensor threads (namely: IMU (ACC+GYRO) and Pressure Sensor)
TX_MUTEX uart_mutex;    // Used to manage access to UART
TX_MUTEX head_mutex;

volatile uint8_t logging_enabled = 1;

TX_SEMAPHORE command_sem;
uint8_t uart_rx_byte;
char cmd_buf[CMD_BUF_LEN];
volatile uint32_t cmd_idx = 0;


TX_SEMAPHORE fl_round_done_sem;

/*******************************************************************************************************************************/
// ST Edge AI Context & Buffers (Strict 8-byte alignment)
static __attribute__((aligned(8))) stai_network net_ctx[STAI_EMBED_NETWORK_CONTEXT_SIZE];
static __attribute__((aligned(8))) uint8_t ai_activations[STAI_EMBED_NETWORK_ACTIVATIONS_SIZE_BYTES];

// Feature & Window Tracking (Exact 50% Overlap)
#define WINDOW_SIZE 24    // must match the model's architecture/training; The model was trained on windows of 24 samples
#define OVERLAP     12    // overlap bet. old and new windows; 1 means too little overlap, WINDOW_SIZE-1 is too much overlap, overlap cannot be WINDOW_SIZe
                          // because that means no new samples are processed

#define ACC_AXIS_COUNT 3   // accelerometer has 3 axes (x,y,z)
#define ACC_TOTAL_WINDOW_LENGTH (WINDOW_SIZE * ACC_AXIS_COUNT)

// Accelerometer: 3 axes * 24 samples
static float acc_window[ACC_TOTAL_WINDOW_LENGTH];

// Gyroscope: Exact per-sample magnitude buffer (no heuristic decay)
static float gyro_mag_window[WINDOW_SIZE];

// model-related variables
static uint32_t sample_idx = 0;
static float full_features[HEAD_TOTAL_FEATURES];
static float class_probs[HEAD_NUM_CLASSES];

volatile uint8_t is_training_mode = 0;
volatile uint8_t target_label = 0;

static float preprocessed_input[ACC_TOTAL_WINDOW_LENGTH];




// New globals (PV section)
#define FL_PAYLOAD_BYTES (FL_PAYLOAD_FLOATS * sizeof(float))  // 224
volatile uint8_t awaiting_weights_payload = 0;
volatile uint32_t weights_bytes_received = 0;
uint8_t rx_weights_buffer[FL_PAYLOAD_BYTES];
TX_SEMAPHORE weights_ready_sem;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */
void tx_sensor_entry(ULONG thread_input);
void press_thread_entry(ULONG input);
void logger_thread_entry(ULONG input);

void command_thread_entry(ULONG input);
void process_command(void);

void weights_thread_entry(ULONG input);

// Preprocessing routine from Lab baseline (IIR gravity + Rodrigues)
extern void HAR_PreprocessWindow(const float *in, float *out, unsigned int n_samples);

/* USER CODE END PFP */

/**
  * @brief  Application ThreadX Initialization.
  * @param memory_ptr: memory pointer
  * @retval int
  */
UINT App_ThreadX_Init(VOID *memory_ptr)
{
  UINT ret = TX_SUCCESS;
  TX_BYTE_POOL *byte_pool = (TX_BYTE_POOL*)memory_ptr;

  /* USER CODE BEGIN App_ThreadX_MEM_POOL */

  /* USER CODE END App_ThreadX_MEM_POOL */
  CHAR *pointer;

  /* Allocate the stack for tx_sensor  */
  if (tx_byte_allocate(byte_pool, (VOID**) &pointer,
                       TX_APP_STACK_SIZE, TX_NO_WAIT) != TX_SUCCESS)
  {
    return TX_POOL_ERROR;
  }
  /* Create tx_sensor.  */
  if (tx_thread_create(&tx_app_thread, "tx_sensor", tx_sensor_entry, 0, pointer,
                       TX_APP_STACK_SIZE, TX_APP_THREAD_PRIO, TX_APP_THREAD_PREEMPTION_THRESHOLD,
                       TX_APP_THREAD_TIME_SLICE, TX_APP_THREAD_AUTO_START) != TX_SUCCESS)
  {
    return TX_THREAD_ERROR;
  }

  /* USER CODE BEGIN App_ThreadX_Init */

  tx_mutex_create(&i2c_mutex, "I2C Mutex", TX_INHERIT);   // I2C BUS Mutex
  tx_mutex_create(&uart_mutex, "UART Mutex", TX_INHERIT); // UART Mutex
  tx_semaphore_create(&command_sem, "Command SEM", 0);    // Semaphore

  // Initialize Head Mutex & Weights
  tx_mutex_create(&head_mutex, "Head_Mutex", TX_NO_INHERIT);
  trainable_head_init(&head_mutex);

  // 
  tx_semaphore_create(&fl_round_done_sem, "FL Round Done SEM", 0);

  // Message Queue
  tx_byte_allocate(byte_pool, (VOID **)&queue_storage, LOG_QUEUE_DEPTH * LOG_MSG_WORDS * sizeof(ULONG), TX_NO_WAIT);
  tx_queue_create(&log_queue, "Log Queue", LOG_MSG_WORDS, queue_storage, LOG_QUEUE_DEPTH * LOG_MSG_WORDS * sizeof(ULONG));

  // logger
  tx_byte_allocate(byte_pool, (VOID **)&logger_thread_stack_ptr, 1024, TX_NO_WAIT);
  tx_thread_create(&logger_thread, "Logger", logger_thread_entry, 0, logger_thread_stack_ptr, 1024, 15, 15, TX_NO_TIME_SLICE, TX_AUTO_START);

  // pressure sensor
  tx_byte_allocate(byte_pool, (VOID **)&press_thread_stack_ptr, 1024, TX_NO_WAIT);
  tx_thread_create(&press_thread, "Pressure", press_thread_entry, 0, press_thread_stack_ptr, 1024, 15, 15, TX_NO_TIME_SLICE, TX_AUTO_START);  

  // Command processing thread
  tx_byte_allocate(byte_pool, (VOID **)&command_thread_stack_ptr, 1024, TX_NO_WAIT);
  tx_thread_create(&command_thread, "Command", command_thread_entry, 0, command_thread_stack_ptr, 1024, 15, 15, TX_NO_TIME_SLICE, TX_AUTO_START);
  
  // weights processing thread
  // tx_byte_allocate(byte_pool, (VOID **)&weights_thread_stack_ptr, 1024, TX_NO_WAIT);
  // tx_thread_create(&weights_thread, "Weights", weights_thread_entry, 0, weights_thread_stack_ptr, 1024, 15, 15, TX_NO_TIME_SLICE, TX_AUTO_START);  

  /* USER CODE END App_ThreadX_Init */

  return ret;
}

void tx_sensor_entry(ULONG thread_input) {
    char log_buf[96];
    stai_return_code ret;

    // Initialize ST Edge AI Runtime
    ret = stai_embed_network_init(net_ctx);
    if (ret != STAI_SUCCESS) {
        snprintf(log_buf, sizeof(log_buf), "AI Init Failed: %d\r\n", ret);
        tx_queue_send(&log_queue, log_buf, TX_NO_WAIT);
        while (1) tx_thread_sleep(100);
    }

    stai_ptr act_buffers[1] = { (stai_ptr)ai_activations };
    stai_embed_network_set_activations(net_ctx, act_buffers, 1);

    // Acquire Input and Output Buffer Pointers
    stai_ptr in_buffers[1], out_buffers[1];
    stai_size n_in, n_out;
    stai_embed_network_get_inputs(net_ctx, in_buffers, &n_in);
    stai_embed_network_get_outputs(net_ctx, out_buffers, &n_out);
    

    // SENSOR_SAMPLE_RATE_HZ Hz Timing: Fractional-remainder accumulator setup
    // 1000 / SENSOR_SAMPLE_RATE_HZ = 38 ticks, remainder = 12 ticks per sample (12/SENSOR_SAMPLE_RATE_HZ)
    const ULONG base_ticks = TX_TIMER_TICKS_PER_SECOND / SENSOR_SAMPLE_RATE_HZ; // 38
    const ULONG remainder_step = TX_TIMER_TICKS_PER_SECOND % SENSOR_SAMPLE_RATE_HZ; // 12
    ULONG remainder_acc = 0;

    while (1) {
        ULONG tick_start = tx_time_get();
        
        // Thread-safe I2C Sensor Acquisition
        BSP_MOTION_SENSOR_Axes_t acc_raw, gyro_raw;
        tx_mutex_get(&i2c_mutex, TX_WAIT_FOREVER);
        int32_t acc_status  = BSP_MOTION_SENSOR_GetAxes(0, MOTION_ACCELERO, &acc_raw);
        int32_t gyro_status = BSP_MOTION_SENSOR_GetAxes(0, MOTION_GYRO, &gyro_raw);
        tx_mutex_put(&i2c_mutex);
        
        // ---------------------- DEBUG:START ----------------------
        // // Check if raw readings are actually arriving
        // static uint32_t debug_counter = 0;
        // if (++debug_counter % SENSOR_SAMPLE_RATE_HZ == 0) { // Print once per second
        //     snprintf(log_buf, sizeof(log_buf), 
        //             "ACC:[%ld, %ld, %ld] | GYRO: [%ld, %ld, %ld] (ret: %ld,%ld)\r\n",
        //             (long)acc_raw.xval, (long)acc_raw.yval, (long)acc_raw.zval,
        //             (long)gyro_raw.xval, (long)gyro_raw.yval, (long)gyro_raw.zval,
        //             (long)acc_status, (long)gyro_status);
        //     tx_queue_send(&log_queue, log_buf, TX_NO_WAIT);
        // }
        // ---------------------- DEBUG:END ----------------------

        // Normalize and store Accelerometer samples (mg -> g)
        acc_window[(sample_idx * 3) + 0] = (acc_raw.xval / 1000.0f) * 9.8f;
        acc_window[(sample_idx * 3) + 1] = (acc_raw.yval / 1000.0f) * 9.8f;
        acc_window[(sample_idx * 3) + 2] = (acc_raw.zval / 1000.0f) * 9.8f;

        // Compute and store instant Gyro rotational magnitude (mdps -> dps)
        float gx = gyro_raw.xval / 1000.0f;
        float gy = gyro_raw.yval / 1000.0f;
        float gz = gyro_raw.zval / 1000.0f;
        gyro_mag_window[sample_idx] = sqrtf(gx*gx + gy*gy + gz*gz);

        sample_idx++;

        // Process Window when 24 samples are ready
        if (sample_idx >= WINDOW_SIZE) {
            
            // a. Preprocess Accelerometer (Gravity removal + Rodrigues rotation)
            HAR_PreprocessWindow(acc_window, preprocessed_input, WINDOW_SIZE);
            float max_dyn = 0.0f;
            for(int i = 0; i < ACC_TOTAL_WINDOW_LENGTH; i++) {
              if (fabsf(preprocessed_input[i]) > max_dyn)
                max_dyn = fabsf(preprocessed_input[i]);
            }
            
            // snprintf(log_buf, sizeof(log_buf), "Max Preproc Dynamic: %.2f g\r\n", max_dyn);
            // tx_queue_send(&log_queue, log_buf, TX_NO_WAIT);

            // Re-acquire the input buffer pointer from net_ctx (ensuring valid pointer)
            stai_ptr in_buffers[1];
            stai_size n_in;
            stai_embed_network_get_inputs(net_ctx, in_buffers, &n_in);
            float *nn_input = (float *)in_buffers[0];

            // Copy ACC_TOTAL_WINDOW_LENGTH floats (ACC_TOTAL_WINDOW_LENGTH * 4 bytes) directly into the model's input buffer
            memcpy(nn_input, preprocessed_input, sizeof(preprocessed_input));

            // Execute Backbone
            stai_embed_network_run(net_ctx, STAI_MODE_SYNC);

            // Re-acquire the output pointer
            stai_ptr out_buffers[1];
            stai_size n_out;
            stai_embed_network_get_outputs(net_ctx, out_buffers, &n_out);
            float *nn_output = (float *)out_buffers[0];

            // Populate the 12 base CNN features into late fusion vector
            for (int i = 0; i < HEAD_CNN_FEATURES; i++) {
                full_features[i] = nn_output[i];
            }

            // Compute exact mean gyro rotational speed over the full window
            float gyro_sum = 0.0f;
            for (int i = 0; i < WINDOW_SIZE; i++) {
                gyro_sum += gyro_mag_window[i];
            }
            float gyro_mean_dps = gyro_sum / (float)WINDOW_SIZE;
            
            // Scaled feature: typical human limb rotation spans 0 to 250+ dps
            full_features[FEAT_GYRO] = gyro_mean_dps / 250.0f;


            // ---------------------- DEBUG:START ----------------------
            // // --- DIAGNOSTIC PRINT ---
            // static uint32_t step = 0;
            // if (++step % 2 == 0) {
            //   snprintf(log_buf, sizeof(log_buf), "OUT_PTR: %p | EMB[0..3]: [%.2f, %.2f, %.2f, %.2f]\r\n", (void*)nn_output,
            //   full_features[0], full_features[1], full_features[2], full_features[3]);
            
            //   tx_queue_send(&log_queue, log_buf, TX_NO_WAIT);
              
            //   // Also print what logits are generated before softmax
            //   // float z0 = 0.0f, z1 = 0.0f, z2 = 0.0f, z3 = 0.0f;
            //   // Inspect BASELINE_B_INIT and first logit accumulation
            //   snprintf(log_buf, sizeof(log_buf), "PREPRO_IN[0..2]: [%.2f, %.2f, %.2f] | NN_IN[0..2]: [%.2f, %.2f, %.2f]\r\n",
            //           acc_window[0], acc_window[1], acc_window[2], nn_input[0], nn_input[1], nn_input[2]);
              
            //   tx_queue_send(&log_queue, log_buf, TX_NO_WAIT);
            // }
            // ---------------------- DEBUG:END ----------------------

            // Trainable Head Step (one-shot corrective training, or inference)
            if (is_training_mode) {
                float loss = trainable_head_train_step(full_features, target_label, 0.01f);
                snprintf(log_buf, sizeof(log_buf), "TRAINED %u %.4f %.2f %.1f\r\n", target_label, loss, max_dyn, gyro_mean_dps);
                tx_queue_send(&log_queue, log_buf, TX_NO_WAIT);

                is_training_mode = 0;   // one-shot: auto-revert to inference after this window
            } else {
                trainable_head_forward(full_features, class_probs);
                uint8_t best_class = 0;
                float best_prob = 0.0f;
                for (uint8_t c = 0; c < HEAD_NUM_CLASSES; c++) {
                    if (class_probs[c] > best_prob) {
                        best_prob = class_probs[c];
                        best_class = c;
                    }
                }

                // ---------------------- DEBUG:START ----------------------
                snprintf(log_buf, sizeof(log_buf),
                        "PRED %u %.3f %.3f %.3f %.3f %.2f %.1f\r\n",
                        best_class, class_probs[0], class_probs[1], class_probs[2], class_probs[3],
                        max_dyn, gyro_mean_dps);
                tx_queue_send(&log_queue, log_buf, TX_NO_WAIT);
                // ---------------------- DEBUG:END ----------------------
            }

            // Shift historical data by 50% (12 samples)
            for (int i = 0; i < OVERLAP * ACC_AXIS_COUNT; i++) {
                acc_window[i] = acc_window[i + (OVERLAP * ACC_AXIS_COUNT)];
            }
            for (int i = 0; i < OVERLAP; i++) {
                gyro_mag_window[i] = gyro_mag_window[i + OVERLAP];
            }
            sample_idx = OVERLAP;
        }

        // Exact Pacing: Base ticks + fractional remainder accumulation
        ULONG sleep_target = base_ticks;
        remainder_acc += remainder_step;
        if (remainder_acc >= SENSOR_SAMPLE_RATE_HZ) {
            sleep_target += 1;
            remainder_acc -= SENSOR_SAMPLE_RATE_HZ;
        }

        ULONG elapsed = tx_time_get() - tick_start;
        if (elapsed < sleep_target) {
            tx_thread_sleep(sleep_target - elapsed);
        } else {
            tx_thread_sleep(1); // Yield when execution takes longer than nominal period
        }
    }
}

  /**
  * @brief  Function that implements the kernel's initialization.
  * @param  None
  * @retval None
  */
void MX_ThreadX_Init(void)
{
  /* USER CODE BEGIN Before_Kernel_Start */

  /* USER CODE END Before_Kernel_Start */

  tx_kernel_enter();

  /* USER CODE BEGIN Kernel_Start_Error */

  /* USER CODE END Kernel_Start_Error */
}

/* USER CODE BEGIN 1 */


void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  if (huart == &huart1)
  {
    uint8_t byte = uart_rx_byte;
    HAL_UART_Receive_IT(&huart1, &uart_rx_byte, 1); // re-arm immediately

    // --- TEMP DIAGNOSTIC: prove bytes are physically arriving at all ---
    // HAL_UART_Transmit(&huart1, &byte, 1, 10);
    // ---------------------------------------------------------------

    if (awaiting_weights_payload)
    {
      rx_weights_buffer[weights_bytes_received++] = byte;
      if (weights_bytes_received >= FL_PAYLOAD_BYTES)
      {
        awaiting_weights_payload = 0;
        tx_semaphore_put(&weights_ready_sem);
      }
      return;  // don't also treat this byte as a command character
    }

    if (byte == '\r' || byte == '\n')
    {
      if (cmd_idx > 0)
      {
        cmd_buf[cmd_idx] = '\0';
        tx_semaphore_put(&command_sem);
      }
    }
    else if (byte != ' ' && cmd_idx < (CMD_BUF_LEN - 1))
    {
      cmd_buf[cmd_idx++] = byte;
    }
  }
}

void command_thread_entry(ULONG input)
{
  while (1)
  {
    tx_semaphore_get(&command_sem, TX_WAIT_FOREVER);
    process_command();
  }
}

void process_command(void)
{
  for (uint32_t i = 0; cmd_buf[i] != '\0'; i++)
  {
    cmd_buf[i] = (char)tolower((unsigned char)cmd_buf[i]);
  }

  if (strcmp(cmd_buf, "start") == 0)
  {
    logging_enabled = 1;
    printf("[board] ok: start\r\n");
  }
  else if (strcmp(cmd_buf, "stop") == 0)
  {
    logging_enabled = 0;
    printf("[board] ok: stop\r\n");
  }
  else if (strcmp(cmd_buf, "led") == 0)
  {
    printf("[board] ok: led toggle\r\n");
    HAL_GPIO_TogglePin(GPIOH, GPIO_PIN_7);
  }
  else if (strcmp(cmd_buf, "infer") == 0)
  {
    is_training_mode = 0;
    printf("[board] ok: infer\r\n");
  }
  else if (cmd_buf[0] == 'l' && cmd_buf[1] >= '0' && cmd_buf[1] <= '3' && cmd_buf[2] == '\0')
  {
    target_label = (uint8_t)(cmd_buf[1] - '0');
    is_training_mode = 1;
    printf("[board] ok: train label %u (next window)\r\n", target_label);
  }
  else if (strcmp(cmd_buf, "getweights") == 0)
  {
    float flat_buffer[FL_PAYLOAD_FLOATS];       // 56 floats = 224 bytes
    trainable_head_export_weights(flat_buffer); // internally locks head_mutex

    const char *marker = "WEIGHTS_DUMP\r\n";
    tx_mutex_get(&uart_mutex, TX_WAIT_FOREVER);
    HAL_UART_Transmit(&huart1, (uint8_t*)marker, strlen(marker), HAL_MAX_DELAY);
    HAL_UART_Transmit(&huart1, (uint8_t*)flat_buffer, sizeof(flat_buffer), HAL_MAX_DELAY);
    tx_mutex_put(&uart_mutex);
  }
  else if (strcmp(cmd_buf, "setweights") == 0)
  {
    weights_bytes_received = 0;
    awaiting_weights_payload = 1;
    printf("[board] ok: send 224 bytes now\r\n");
  }
  else if (strcmp(cmd_buf, "done") == 0)
  {
    is_training_mode = 0;
    printf("[board] ok: round complete, notifying server\r\n");
    tx_semaphore_put(&fl_round_done_sem); // releases NetX Duo TCP thread
  }
  else
  {
    printf("[board] err: unknown command '%s'\r\n", cmd_buf);
  }

  cmd_idx = 0;
}



// // New dedicated thread (same style as command_thread/logger_thread)
// void weights_thread_entry(ULONG input)
// {
//   while (1)
//   {
//     tx_semaphore_get(&weights_ready_sem, TX_WAIT_FOREVER);
//     trainable_head_import_weights((float *)rx_weights_buffer);
//     printf("[board] ok: weights updated\r\n");
//   }
// }


void logger_thread_entry(ULONG input) 
{
  char line[64]; 
  while (1)
  {
    // always pull from the queue so producers never freeze
    tx_queue_receive(&log_queue, line, TX_WAIT_FOREVER);
    if (logging_enabled)
    {
      printf("%s", line);
      fflush(stdout);
    }
  }
}

void press_thread_entry(ULONG input) {
  // char line[64];
  float press_val;
  
  tx_thread_sleep(TX_TIMER_TICKS_PER_SECOND * 1);
    
  while (1)
  {
    tx_mutex_get(&i2c_mutex, TX_WAIT_FOREVER);
    BSP_ENV_SENSOR_GetValue(1, ENV_PRESSURE, &press_val);
    tx_mutex_put(&i2c_mutex);
    
    // snprintf(line, sizeof(line), "Live Pressure: %.2f hPa\r\n", press_val);
    // tx_queue_send(&log_queue, line, TX_NO_WAIT);
    
    // Sleep for 500 ms
    tx_thread_sleep(TX_TIMER_TICKS_PER_SECOND / 2); 
  }
}
/* USER CODE END 1 */
