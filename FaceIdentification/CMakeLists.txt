set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_VERSION 1)
if(NOT DEFINED CMAKE_SYSTEM_PROCESSOR)
  set(CMAKE_SYSTEM_PROCESSOR arm)
else()
  #message("CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
endif()

SET(TOOLCHAIN_ROOT_PATH  $ENV{SDK_ROOT}/prebuilts/toolschain/arm-linux-gnueabihf)
SET(OPENCV_RK3288_PATH "/home/tchip/project/opencv/opencv-rk3288/out/system")

SET(CMAKE_FIND_ROOT_PATH ${TOOLCHAIN_ROOT_PATH} ${OPENCV_RK3288_PATH})

# search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

SET(CMAKE_C_COMPILER   $ENV{SDK_ROOT}/prebuilts/toolschain/bin/arm-linux-gnueabihf-gcc)
SET(CMAKE_CXX_COMPILER $ENV{SDK_ROOT}/prebuilts/toolschain/bin/arm-linux-gnueabihf-g++)
SET(CMAKE_LINKER       $ENV{SDK_ROOT}/prebuilts/toolschain/bin/arm-linux-gnueabihf-ld)
SET(CMAKE_AR           $ENV{SDK_ROOT}/prebuilts/toolschain/bin/arm-linux-gnueabihf-ar)

if(NOT DEFINED CMAKE_CXX_FLAGS)
  set(CMAKE_CXX_FLAGS           "" CACHE INTERAL "")
  set(CMAKE_C_FLAGS             "" CACHE INTERAL "")
  set(CMAKE_SHARED_LINKER_FLAGS "" CACHE INTERAL "")
  set(CMAKE_MODULE_LINKER_FLAGS "" CACHE INTERAL "")
  set(CMAKE_EXE_LINKER_FLAGS    "" CACHE INTERAL "")

# Use C++11
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  message(STATUS "C++11 support has been enabled by default.")

  set(CMAKE_CXX_FLAGS           "${CMAKE_CXX_FLAGS} -fdata-sections -Wa,--noexecstack -fsigned-char -Wno-psabi")
  set(CMAKE_C_FLAGS             "${CMAKE_C_FLAGS} -fdata-sections -Wa,--noexecstack -fsigned-char -Wno-psabi")
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL arm)
    set(CMAKE_CXX_FLAGS           "-mthumb ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS           "-mfpu=neon ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS             "-mthumb ${CMAKE_C_FLAGS}")
    set(CMAKE_C_FLAGS             "-mfpu=neon ${CMAKE_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,nocopyreloc")
  endif()
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL arm)
    set(ARM_LINKER_FLAGS "-Wl,--fix-cortex-a8 -Wl,--no-undefined -Wl,--gc-sections -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64)
    set(ARM_LINKER_FLAGS "-Wl,--no-undefined -Wl,--gc-sections -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
  endif()
  set(CMAKE_SHARED_LINKER_FLAGS "${ARM_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
  set(CMAKE_MODULE_LINKER_FLAGS "${ARM_LINKER_FLAGS} ${CMAKE_MODULE_LINKER_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS    "${ARM_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
else()
  #message(WARNING "CMAKE_CXX_FLAGS='${CMAKE_CXX_FLAGS}' is defined")
endif()

cmake_minimum_required (VERSION 2.6)

project (viplnet)

set (VIPLNET_VERSION_MAJOR 4)
set (VIPLNET_VERSION_MINOR 5)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -std=c++11 -O2")


MESSAGE(STATUS "other platform: ${CMAKE_SYSTEM_NAME}")

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DGTEST_USE_OWN_TR1_TUPLE=1")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DGTEST_USE_OWN_TR1_TUPLE=1")
endif (CMAKE_SYSTEM_NAME MATCHES "Darwin")

set(VIPLNET_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(VIPLNET_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
# set __VIOL_LOG__ macro
# add_definitions(-D__VIPL_LOG__)

include_directories(${VIPLNET_INCLUDE_DIR})
include_directories(${VIPLNET_SRC_DIR})

add_subdirectory(src)

aux_source_directory(./src SRC_LIST)
aux_source_directory(./tools TOOLS_LIST)
add_library(viplnet SHARED ${SRC_LIST} ${TOOLS_LIST})
set_target_properties(viplnet PROPERTIES 
  VERSION ${VIPLNET_VERSION_MAJOR}.${VIPLNET_VERSION_MINOR} 
  SOVERSION ${VIPLNET_VERSION_MAJOR}.${VIPLNET_VERSION_MINOR}) 
