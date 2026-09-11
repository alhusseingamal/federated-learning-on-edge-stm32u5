/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    app_netxduo.c
  * @author  MCD Application Team
  * @brief   NetXDuo applicative file
  ******************************************************************************
    * @attention
  *
  * Copyright (c) 2021 STMicroelectronics.
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
#include "app_netxduo.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "app_azure_rtos.h"
#include "nx_ip.h"
#include "stm32u5xx_hal_rtc.h"
#include <time.h>

#include "trainable_head.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
TX_THREAD AppMainThread;
TX_THREAD AppSNTPThread;

TX_SEMAPHORE Semaphore;

NX_PACKET_POOL           AppPool;

NX_IP                    IpInstance;
NX_DHCP                  DhcpClient;
NX_DNS                   DnsClient;
NX_SNTP_CLIENT           SntpClient;

TX_EVENT_FLAGS_GROUP     SntpFlags;

ULONG                    IpAddress;
ULONG                    NetMask;

CHAR                     buffer[64];

CHAR                     *pointer;

struct tm timeInfos;

/* RTC handler declaration */
extern RTC_HandleTypeDef hrtc;


NX_UDP_SOCKET UdpEchoSocket;  // @ali
NX_TCP_SOCKET FLClientSocket; // @ali

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
static UINT kiss_of_death_handler(NX_SNTP_CLIENT *client_ptr, UINT KOD_code);
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
static void display_rtc_time(RTC_HandleTypeDef *hrtc);
static void rtc_time_update(NX_SNTP_CLIENT *client_ptr);
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
/* set the SNTP network interface to the primary interface. */
UINT  iface_index =0;

extern TX_SEMAPHORE fl_round_done_sem;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */
static VOID App_Main_Thread_Entry(ULONG thread_input);
static VOID App_SNTP_Thread_Entry(ULONG thread_input);

static VOID ip_address_change_notify_callback(NX_IP *ip_instance, VOID *ptr);
static VOID time_update_callback(NX_SNTP_TIME_MESSAGE *time_update_ptr, NX_SNTP_TIME *local_time);

static UINT dns_create(NX_DNS *dns_ptr);
/* USER CODE END PFP */

/**
  * @brief  Application NetXDuo Initialization.
  * @param memory_ptr: memory pointer
  * @retval int
  */
UINT MX_NetXDuo_Init(VOID *memory_ptr)
{
  UINT ret = NX_SUCCESS;
  TX_BYTE_POOL *byte_pool = (TX_BYTE_POOL*)memory_ptr;

   /* USER CODE BEGIN App_NetXDuo_MEM_POOL */

  /* USER CODE END App_NetXDuo_MEM_POOL */
  /* USER CODE BEGIN 0 */

  /* USER CODE END 0 */

  /* USER CODE BEGIN MX_NetXDuo_Init */
#if (USE_STATIC_ALLOCATION  == 1)
  printf("Nx_SNTP_Client application started..\n");

  CHAR *pointer;

  /* Initialize the NetX system. */
  nx_system_initialize();

  /* Allocate the memory for packet_pool.  */
  if (tx_byte_allocate(byte_pool, (VOID **) &pointer,  NX_PACKET_POOL_SIZE, TX_NO_WAIT) != TX_SUCCESS)
  {
    return TX_POOL_ERROR;
  }

  /* Create the Packet pool to be used for packet allocation */
  ret = nx_packet_pool_create(&AppPool, "Main Packet Pool", PAYLOAD_SIZE, pointer, NX_PACKET_POOL_SIZE);

  if (ret != NX_SUCCESS)
  {
    return NX_NOT_ENABLED;
  }

  /* Allocate the memory for Ip_Instance */
  if (tx_byte_allocate(byte_pool, (VOID **) &pointer, 2 * DEFAULT_MEMORY_SIZE, TX_NO_WAIT) != TX_SUCCESS)
  {
    return TX_POOL_ERROR;
  }

  /* Create the main NX_IP instance */
  ret = nx_ip_create(&IpInstance, "Main Ip instance", NULL_ADDRESS, NULL_ADDRESS, &AppPool, nx_driver_emw3080_entry,
                     pointer, 2 * DEFAULT_MEMORY_SIZE, DEFAULT_MAIN_PRIORITY);

  if (ret != NX_SUCCESS)
  {
    return NX_NOT_ENABLED;
  }

  /* create the DHCP client */
  ret = nx_dhcp_create(&DhcpClient, &IpInstance, "DHCP Client");

  if (ret != NX_SUCCESS)
  {
    return NX_NOT_ENABLED;
  }

  /* Allocate the memory for ARP */
  if (tx_byte_allocate(byte_pool, (VOID **) &pointer, ARP_MEMORY_SIZE, TX_NO_WAIT) != TX_SUCCESS)
  {
    return TX_POOL_ERROR;
  }

  /* Enable the ARP protocol and provide the ARP cache size for the IP instance */
  ret = nx_arp_enable(&IpInstance, (VOID *)pointer, ARP_MEMORY_SIZE);

  if (ret != NX_SUCCESS)
  {
    return NX_NOT_ENABLED;
  }

  /* Enable the ICMP */
  ret = nx_icmp_enable(&IpInstance);

  if (ret != NX_SUCCESS)
  {
    return NX_NOT_ENABLED;
  }

  /* Enable the UDP protocol required for  DHCP communication */
  ret = nx_udp_enable(&IpInstance);

  if (ret != NX_SUCCESS)
  {
    return NX_NOT_ENABLED;
  }

    /* Enable the TCP protocol */
  ret = nx_tcp_enable(&IpInstance);

  if (ret != NX_SUCCESS)
  {
    return NX_NOT_ENABLED;
  }

  /* Allocate the memory for main thread   */
  if (tx_byte_allocate(byte_pool, (VOID **) &pointer, MAIN_THREAD_MEMORY, TX_NO_WAIT) != TX_SUCCESS)
  {
    return TX_POOL_ERROR;
  }

  /* Create the main thread */
  ret = tx_thread_create(&AppMainThread, "App Main thread", App_Main_Thread_Entry, 0, pointer, MAIN_THREAD_MEMORY,
                         DEFAULT_MAIN_PRIORITY, DEFAULT_MAIN_PRIORITY, TX_NO_TIME_SLICE, TX_AUTO_START);

  if (ret != TX_SUCCESS)
  {
    return NX_NOT_ENABLED;
  }

  /* Allocate the memory for SNTP client thread   */
  if (tx_byte_allocate(byte_pool, (VOID **) &pointer, SNTP_CLIENT_THREAD_MEMORY, TX_NO_WAIT) != TX_SUCCESS)
  {
    return TX_POOL_ERROR;
  }

  /* create the SNTP client thread */
  ret = tx_thread_create(&AppSNTPThread, "App SNTP Thread", App_SNTP_Thread_Entry, 0, pointer, SNTP_CLIENT_THREAD_MEMORY,
                         DEFAULT_PRIORITY, DEFAULT_PRIORITY, TX_NO_TIME_SLICE, TX_DONT_START);

  if (ret != TX_SUCCESS)
  {
    return NX_NOT_ENABLED;
  }

  /* Create the event flags. */
  ret = tx_event_flags_create(&SntpFlags, "SNTP event flags");

  /* Check for errors */
  if (ret != NX_SUCCESS)
  {
    return NX_NOT_ENABLED;
  }

  /* set DHCP notification callback  */
  tx_semaphore_create(&Semaphore, "DHCP Semaphore", 0);
#endif
  /* USER CODE END MX_NetXDuo_Init */

  return ret;
}

/* USER CODE BEGIN 1 */
/**
* @brief  ip address change callback.
* @param ip_instance: NX_IP instance
* @param ptr: user data
* @retval none
*/
static VOID ip_address_change_notify_callback(NX_IP *ip_instance, VOID *ptr)
{
  /* release the semaphore as soon as an IP address is available */
  tx_semaphore_put(&Semaphore);
}

/**
* @brief  Main thread entry.
* @param thread_input: ULONG user argument used by the thread entry
* @retval none
*/
static VOID App_Main_Thread_Entry(ULONG thread_input)
{
  UINT ret;

  ret = nx_ip_address_change_notify(&IpInstance, ip_address_change_notify_callback, NULL);
  if (ret != NX_SUCCESS)
  {
    Error_Handler();
  }

  ret = nx_dhcp_start(&DhcpClient);

  if (ret != NX_SUCCESS)
  {
    Error_Handler();
  }

  /* wait until an IP address is ready */
  if(tx_semaphore_get(&Semaphore, TX_WAIT_FOREVER) != TX_SUCCESS)
  {
    Error_Handler();
  }

  ret = nx_ip_address_get(&IpInstance, &IpAddress, &NetMask);

  if (ret != TX_SUCCESS)
  {
    Error_Handler();
  }

  PRINT_IP_ADDRESS(IpAddress);

  /* the network is correctly initialized, start the TCP server thread */
  tx_thread_resume(&AppSNTPThread);

  /* this thread is not needed any more, we relinquish it */
  tx_thread_relinquish();

  return;
}

/**
* @brief  DNS Create Function.
* @param dns_ptr
* @retval ret
*/
UINT dns_create(NX_DNS *dns_ptr)
{
  UINT ret = NX_SUCCESS;

  /* Create a DNS instance for the Client */
  ret = nx_dns_create(dns_ptr, &IpInstance, (UCHAR *)"DNS Client");

  if (ret)
  {
    Error_Handler();
  }

  /* Initialize DNS instance with the DNS server Address */
  ret = nx_dns_server_add(dns_ptr, USER_DNS_ADDRESS);
  if (ret)
  {
    Error_Handler();
  }

  return ret;
}

// @ali: 4th modification

static UINT fl_tcp_receive_exact(NX_TCP_SOCKET *socket_ptr, UCHAR *dest, ULONG n_bytes, ULONG wait_option)
{
  ULONG total_received = 0;
  NX_PACKET *packet_ptr;
  UINT status;
  ULONG bytes_copied;

  while (total_received < n_bytes)
  {
    status = nx_tcp_socket_receive(socket_ptr, &packet_ptr, wait_option);
    if (status != NX_SUCCESS)
    {
      printf("[fl_tcp_receive_exact] nx_tcp_socket_receive failed: 0x%02X\r\n", status);
      return status;
    }

    status = nx_packet_data_extract_offset(packet_ptr, 0, dest + total_received,
                                            n_bytes - total_received, &bytes_copied);
    nx_packet_release(packet_ptr);

    if (status != NX_SUCCESS)
    {
      printf("[fl_tcp_receive_exact] extract_offset failed: 0x%02X\r\n", status);
      return status;
    }
    if (bytes_copied == 0)
    {
      printf("[fl_tcp_receive_exact] 0 bytes extracted! Continuing wait...\r\n");
      return NX_NOT_SUCCESSFUL;  /* guard against an infinite loop if 0 bytes ever comes back */
    }

    total_received += bytes_copied;
    printf("[FL] recv chunk: %lu bytes (total %lu/%lu)\r\n", bytes_copied, total_received, n_bytes);
  }

  return NX_SUCCESS;
}

#define FL_WEIGHTS_BYTES        (HEAD_FLAT_SIZE * sizeof(float))
#define FL_SERVER_PORT          9999
#define FL_SERVER_IP            IP_ADDRESS(192, 168, 42, 1)

static void App_SNTP_Thread_Entry(ULONG info)
{
  UINT status;
  // NX_PACKET *recv_packet;
  NX_PACKET *send_packet;
  ULONG round_count = 0;
  ULONG bytes_copied = 0;
  float weights_buffer[HEAD_FLAT_SIZE];

  NX_PARAMETER_NOT_USED(info);

  /* 1. Setup & Connect TCP Socket */
  status = nx_tcp_socket_create(&IpInstance, &FLClientSocket, "FL Client Socket",
                                NX_IP_NORMAL, NX_FRAGMENT_OKAY, NX_IP_TIME_TO_LIVE,
                                1536, NX_NULL, NX_NULL);
  if (status != NX_SUCCESS) return;

  status = nx_tcp_client_socket_bind(&FLClientSocket, NX_ANY_PORT, TX_WAIT_FOREVER);
  if (status != NX_SUCCESS) return;

  printf("Connecting to FL server...\r\n");
  status = nx_tcp_client_socket_connect(&FLClientSocket, FL_SERVER_IP, FL_SERVER_PORT,
                                        NX_IP_PERIODIC_RATE * 10);
  if (status != NX_SUCCESS) {
    printf("FL connect failed (0x%02X)\r\n", status);
    nx_tcp_client_socket_unbind(&FLClientSocket);
    nx_tcp_socket_delete(&FLClientSocket);
    return;
  }
  printf("FL connected to server\r\n");

  /* 2. Receive Round Count Handshake */
  // status = nx_tcp_socket_receive(&FLClientSocket, &recv_packet, NX_IP_PERIODIC_RATE * 5);
  // if (status != NX_SUCCESS) return;
  status = fl_tcp_receive_exact(&FLClientSocket, (UCHAR *)&round_count, sizeof(round_count), NX_IP_PERIODIC_RATE * 5);
  if (status != NX_SUCCESS) return;
  printf("FL server requested %lu round(s)\r\n", (unsigned long)round_count);

  // status = nx_packet_data_extract_offset(recv_packet, 0, &round_count, sizeof(round_count), &bytes_copied);
  // nx_packet_release(recv_packet);
  // if (status != NX_SUCCESS || bytes_copied != sizeof(round_count)) return;

  printf("FL server requested %lu round(s)\r\n", (unsigned long)round_count);

  /* 3. FL Execution Rounds */
  for (ULONG r = 0; r < round_count; r++)
  {
    printf("--- Starting FL Round %lu/%lu ---\r\n", r + 1, round_count);

    /* 3a. Receive global weights */
    // status = nx_tcp_socket_receive(&FLClientSocket, &recv_packet, TX_WAIT_FOREVER);
    // if (status != NX_SUCCESS) break;
    /* Change Step 3a in App_SNTP_Thread_Entry */
    status = fl_tcp_receive_exact(&FLClientSocket, (UCHAR *)weights_buffer, FL_WEIGHTS_BYTES, NX_IP_PERIODIC_RATE * 10000);
    if (status != NX_SUCCESS) 
    {
      printf("[FL ERROR] Timed out waiting for weights: 0x%02X\r\n", status);
      break;
    }

    // status = nx_packet_data_extract_offset(recv_packet, 0, weights_buffer, FL_WEIGHTS_BYTES, &bytes_copied);
    // nx_packet_release(recv_packet);
    // if (status != NX_SUCCESS || bytes_copied != FL_WEIGHTS_BYTES) break;

    /* 3b. Overwrite local head with global parameters */
    trainable_head_import_weights(weights_buffer);
    printf("[FL] Global weights applied. Ready for local training.\r\n");

    /* 3c. Synchronized local training gate */
    /* Blocks until local training steps run or 'done' command is sent */
    tx_semaphore_get(&fl_round_done_sem, TX_WAIT_FOREVER);

    /* 3d. Export newly trained parameters */
    trainable_head_export_weights(weights_buffer);
    printf("[FL] Local training complete. Exporting weights...\r\n");

    /* 3e. Transmit updated weights back to aggregator */
    status = nx_packet_allocate(&AppPool, &send_packet, NX_TCP_PACKET, TX_WAIT_FOREVER);
    if (status != NX_SUCCESS) break;

    status = nx_packet_data_append(send_packet, weights_buffer, FL_WEIGHTS_BYTES, &AppPool, TX_WAIT_FOREVER);
    if (status != NX_SUCCESS) {
      nx_packet_release(send_packet);
      break;
    }

    status = nx_tcp_socket_send(&FLClientSocket, send_packet, NX_IP_PERIODIC_RATE * 5);
    if (status != NX_SUCCESS) {
      nx_packet_release(send_packet);
      break;
    }
    printf("[FL] Weights transmitted to server for round %lu\r\n", r + 1);
  }


  /* 4. Receive Final Aggregated Global Model */
  printf("[FL] Receiving final aggregated model from server...\r\n");
  status = fl_tcp_receive_exact(&FLClientSocket, (UCHAR *)weights_buffer, FL_WEIGHTS_BYTES, NX_IP_PERIODIC_RATE * 5);
  if (status == NX_SUCCESS)
  {
    trainable_head_import_weights(weights_buffer);
    printf("[FL] Final aggregated model applied successfully!\r\n");
  }
  else
  {
    printf("[FL WARNING] Failed to receive final model (status 0x%02X)\r\n", status);
  }

  printf("FL session complete. Closing connection.\r\n");
  nx_tcp_socket_disconnect(&FLClientSocket, NX_IP_PERIODIC_RATE * 2);
  nx_tcp_client_socket_unbind(&FLClientSocket);
  nx_tcp_socket_delete(&FLClientSocket);

  while (1) {
    tx_thread_sleep(NX_IP_PERIODIC_RATE);
  }
}


/* This application defined handler for handling a Kiss of Death packet is not
required by the SNTP Client. A KOD handler should determine
if the Client task should continue vs. abort sending/receiving time data
from its current time server, and if aborting if it should remove
the server from its active server list.

Note that the KOD list of codes is subject to change. The list
below is current at the time of this software release. */

static UINT kiss_of_death_handler(NX_SNTP_CLIENT *client_ptr, UINT KOD_code)
{
  UINT    remove_server_from_list = NX_FALSE;
  UINT    status = NX_SUCCESS;

  NX_PARAMETER_NOT_USED(client_ptr);

  /* Handle kiss of death by code group. */
  switch (KOD_code)
  {

  case NX_SNTP_KOD_RATE:
  case NX_SNTP_KOD_NOT_INIT:
  case NX_SNTP_KOD_STEP:

    /* Find another server while this one is temporarily out of service.  */
    status =  NX_SNTP_KOD_SERVER_NOT_AVAILABLE;

    break;

  case NX_SNTP_KOD_AUTH_FAIL:
  case NX_SNTP_KOD_NO_KEY:
  case NX_SNTP_KOD_CRYP_FAIL:

    /* These indicate the server will not service client with time updates
    without successful authentication. */

    remove_server_from_list =  NX_TRUE;

    break;


  default:

    /* All other codes. Remove server before resuming time updates. */

    remove_server_from_list =  NX_TRUE;
    break;
  }

  /* Removing the server from the active server list? */
  if (remove_server_from_list)
  {

    /* Let the caller know it has to bail on this server before resuming service. */
    status = NX_SNTP_KOD_REMOVE_SERVER;
  }

  return status;
}

/* This application defined handler for notifying SNTP time update event.  */
static VOID time_update_callback(NX_SNTP_TIME_MESSAGE *time_update_ptr, NX_SNTP_TIME *local_time)
{
  NX_PARAMETER_NOT_USED(time_update_ptr);
  NX_PARAMETER_NOT_USED(local_time);

  tx_event_flags_set(&SntpFlags, SNTP_UPDATE_EVENT, TX_OR);
}

/* This application updates Time from SNTP to STM32 RTC */
static void rtc_time_update(NX_SNTP_CLIENT *client_ptr)
{
  RTC_DateTypeDef sdatestructure ={0};
  RTC_TimeTypeDef stimestructure ={0};
  struct tm ts;
  CHAR  temp[32] = {0};

  /* convert SNTP time (seconds since 01-01-1900 to 01-01-1970)

  EPOCH_TIME_DIFF is equivalent to 70 years in sec
  calculated with www.epochconverter.com/date-difference
  This constant is used to delete difference between :
  Epoch converter (referenced to 1970) and SNTP (referenced to 1900) */
  time_t timestamp = client_ptr->nx_sntp_current_server_time_message.receive_time.seconds - EPOCH_TIME_DIFF;

  /* convert time in yy/mm/dd hh:mm:sec */
  ts = *localtime(&timestamp);

  /* convert date composants to hex format */
  sprintf(temp, "%d", (ts.tm_year - 100));
  sdatestructure.Year = strtol(temp, NULL, 16);
  sprintf(temp, "%d", ts.tm_mon + 1);
  sdatestructure.Month = strtol(temp, NULL, 16);
  sprintf(temp, "%d", ts.tm_mday);
  sdatestructure.Date = strtol(temp, NULL, 16);
  /* dummy weekday */
  sdatestructure.WeekDay =0x00;

  if (HAL_RTC_SetDate(&hrtc, &sdatestructure, RTC_FORMAT_BCD) != HAL_OK)
  {
    Error_Handler();
  }
  /* convert time composants to hex format */
  sprintf(temp,"%d", ts.tm_hour);
  stimestructure.Hours = strtol(temp, NULL, 16);
  sprintf(temp,"%d", ts.tm_min);
  stimestructure.Minutes = strtol(temp, NULL, 16);
  sprintf(temp, "%d", ts.tm_sec);
  stimestructure.Seconds = strtol(temp, NULL, 16);

  if (HAL_RTC_SetTime(&hrtc, &stimestructure, RTC_FORMAT_BCD) != HAL_OK)
  {
    Error_Handler();
  }

}

/* this application displays time from RTC */
static void display_rtc_time(RTC_HandleTypeDef *hrtc)
{
  RTC_TimeTypeDef RTC_Time = {0};
  RTC_DateTypeDef RTC_Date = {0};

  HAL_RTC_GetTime(hrtc, &RTC_Time,RTC_FORMAT_BCD);
  HAL_RTC_GetDate(hrtc, &RTC_Date,RTC_FORMAT_BCD);

  printf("%02x-%02x-20%02x / %02x:%02x:%02x\n",\
        RTC_Date.Date, RTC_Date.Month, RTC_Date.Year,RTC_Time.Hours,RTC_Time.Minutes,RTC_Time.Seconds);
}
/* USER CODE END 1 */
