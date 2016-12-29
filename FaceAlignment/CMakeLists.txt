set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_VERSION 1)
if(NOT DEFINED CMAKE_SYSTEM_PROCESSOR)
  set(CMAKE_SYSTEM_PROCESSOR arm)
else()
  #message("CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
endif()

SET(CMAKE_FIND_ROOT_PATH  $ENV{SDK_ROOT}/prebuilts/toolschain/arm-linux-gnueabihf)

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
    set(CMAKE_C_FLAGS             "-mthumb ${CMAKE_C_FLAGS}")
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

cmake_minimum_required(VERSION 2.8.4)

project(seeta_fa_lib)


# Build options
option(BUILD_EXAMPLES  "Set to ON to build examples"  OFF)

include_directories(include)

set(src_files 
    src/cfan.cpp
    src/face_alignment.cpp
    src/sift.cpp
    )

add_library(seeta_fa_lib SHARED ${src_files})
set(fa_required_libs seeta_fa_lib)

if (BUILD_EXAMPLES)
    message(STATUS "Build with examples.")
    find_package(OpenCV)
    if (NOT OpenCV_FOUND)
        message(WARNING "OpenCV not found. Test will not be built.")
    else()
        include_directories(${OpenCV_INCLUDE_DIRS} build)
	link_directories(build)
        list(APPEND fa_required_libs ${OpenCV_LIBS} seeta_facedet_lib)
        add_executable(fa_test src/test/face_alignment_test.cpp)
        target_link_libraries(fa_test ${fa_required_libs})
    endif()
endif()
