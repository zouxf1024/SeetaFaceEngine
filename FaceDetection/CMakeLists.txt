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

cmake_minimum_required(VERSION 3.1.0)

project(seeta_facedet_lib)

# Build options
option(BUILD_EXAMPLES  "Set to ON to build examples"  OFF)
option(USE_OPENMP      "Set to ON to build use openmp"  OFF)
option(USE_SSE         "Set to ON to build use SSE"  OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# Use C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++11 support has been enabled by default.")

# Use SSE
if (USE_SSE)
    add_definitions(-DUSE_SSE)
    message(STATUS "Use SSE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
endif()

# Use OpenMP
if (USE_OPENMP)
    find_package(OpenMP QUIET)
    if (OPENMP_FOUND)
        message(STATUS "Use OpenMP")
        add_definitions(-DUSE_OPENMP)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

include_directories(include)

set(src_files 
    src/util/nms.cpp
    src/util/image_pyramid.cpp
    src/io/lab_boost_model_reader.cpp
    src/io/surf_mlp_model_reader.cpp
    src/feat/lab_feature_map.cpp
    src/feat/surf_feature_map.cpp
    src/classifier/lab_boosted_classifier.cpp
    src/classifier/mlp.cpp
    src/classifier/surf_mlp.cpp
    src/face_detection.cpp
    src/fust.cpp
    )

# Build shared library
add_library(seeta_facedet_lib SHARED ${src_files})
set(facedet_required_libs seeta_facedet_lib)

# Build examples
if (BUILD_EXAMPLES)
    message(STATUS "Build with examples.")
    find_package(OpenCV)
    if (NOT OpenCV_FOUND)
        message(WARNING "OpenCV not found. Test will not be built.")
    else()
        include_directories(${OpenCV_INCLUDE_DIRS})
        list(APPEND facedet_required_libs ${OpenCV_LIBS})
        
        add_executable(facedet_test src/test/facedetection_test.cpp)
        target_link_libraries(facedet_test ${facedet_required_libs})
    endif()
endif()
