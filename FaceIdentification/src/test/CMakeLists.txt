aux_source_directory (. SRC_LIST)

message(${SRC_LIST})

# add external libraries
find_package(OpenCV REQUIRED)

enable_testing ()

list(APPEND test_libs ${OpenCV_LIBS} "seeta_facedet_lib" "seeta_fa_lib" "pthread")
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib)

foreach (f ${SRC_LIST})
  string(REGEX REPLACE "[.]cpp" ".bin" BIN ${f})
  add_executable(${BIN} ${f})
  target_link_libraries(${BIN} viplnet ${test_libs})
endforeach ()

