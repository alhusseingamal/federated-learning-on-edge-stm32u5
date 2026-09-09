cmake_minimum_required(VERSION 3.22)

#
# This file is generated only once,
# and is not re-generated if converter is called multiple times.
#
# User is free to modify the file as much as necessary
#

# Setup compiler settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)


# Define the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# Set the project name
set(CMAKE_PROJECT_NAME Nx_SNTP_Client)

# Enable compile command to ease indexing with e.g. clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Core project settings
project(${CMAKE_PROJECT_NAME})
message("Build type: " ${CMAKE_BUILD_TYPE})

# Enable CMake support for ASM and C languages
enable_language(C ASM)

# Create an executable object type
add_executable(${CMAKE_PROJECT_NAME})

# Add STM32CubeMX generated sources
add_subdirectory(cmake/stm32cubemx)

# Link directories setup
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined library search paths
)

# Add sources to executable
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user sources here
    # General
    Drivers/BSP/b_u585i_iot02a.c
    Drivers/BSP/b_u585i_iot02a_bus.c

    # IMU Sensors (accelerometer + gyroscope)
    Drivers/BSP/b_u585i_iot02a_motion_sensors.c
    Drivers/BSP/Components/ism330dhcx/ism330dhcx.c
    Drivers/BSP/Components/ism330dhcx/ism330dhcx_reg.c
    Drivers/BSP/Components/iis2mdc/iis2mdc.c
    Drivers/BSP/Components/iis2mdc/iis2mdc_reg.c

    # Pressure sensor
    Drivers/BSP/Components/lps22hh/lps22hh.c
    Drivers/BSP/Components/lps22hh/lps22hh_reg.c
    
    # Environment sensor
    Drivers/BSP/b_u585i_iot02a_env_sensors.c
    Drivers/BSP/Components/hts221/hts221.c
    Drivers/BSP/Components/hts221/hts221_reg.c
    
    # NN files
    Core/Src/preprocess.c
    # HAR/AI/network.c
    # HAR/AI/network_data.c
    HAR/AI_EMBED/embed_network.c       # Replaces network.c
    HAR/AI_EMBED/embed_network_data.c  # Replaces network_data.c
    Core/Src/trainable_head.c          # Add this new module
)

# Add include paths
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined include paths
    Drivers/BSP
    
    Drivers/BSP/Components/ism330dhcx
    Drivers/BSP/Components/iis2mdc

    Drivers/BSP/Components/lps22hh

    Drivers/BSP/Components/hts221
    
    AI_Runtime
    AI_Runtime/Inc
    # HAR/AI
    HAR/AI_EMBED
)

# Add project symbols (macros)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined symbols
)

# Remove wrong libob.a library dependency when using cpp files
list(REMOVE_ITEM CMAKE_C_IMPLICIT_LINK_LIBRARIES ob)

# Add linked libraries
target_link_libraries(${CMAKE_PROJECT_NAME}
    stm32cubemx

    # Add user defined libraries
    ${CMAKE_CURRENT_SOURCE_DIR}/AI_Runtime/NetworkRuntime1201_CM33_GCC.a
    m
)

target_link_options(${CMAKE_PROJECT_NAME} PRIVATE 
    -u _printf_float
)