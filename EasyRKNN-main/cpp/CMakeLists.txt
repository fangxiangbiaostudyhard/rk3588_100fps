cmake_minimum_required(VERSION 3.13)
project(rknn_deploy LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-O3 -pthread)
add_link_options(-Wl,--allow-shlib-undefined)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_deploy_${CMAKE_SYSTEM_NAME})
set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

set(LIB_ARCH aarch64)
set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/include/librknnrt.so)
set(RGA_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/rga/RK3588)
set(RGA_LIB ${RGA_PATH}/lib/Linux/${LIB_ARCH}/librga.so)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/3rdparty)
include_directories(${RGA_PATH}/include)
find_package(OpenCV REQUIRED)

set(RKNN_DEPLOY_SOURCES
    src/utils/preprocess_yolo.cpp
    src/models/yolov5/rknn_yolov5.cpp
    src/models/yolov8/rknn_yolov8.cpp
    src/models/yolov11/rknn_yolov11.cpp
    src/api/rknn_detector.cpp
    src/api/rknn_detector_c.cpp
)

add_library(rknn_deploy STATIC ${RKNN_DEPLOY_SOURCES})
target_link_libraries(rknn_deploy PUBLIC ${RKNN_RT_LIB} ${OpenCV_LIBS} ${RGA_LIB} pthread)

add_executable(rknn_detector_demo examples/detector_demo.cpp)
target_link_libraries(rknn_detector_demo PRIVATE rknn_deploy)

install(TARGETS rknn_detector_demo DESTINATION bin)
install(TARGETS rknn_deploy DESTINATION lib)
install(PROGRAMS ${RKNN_RT_LIB} ${RGA_LIB} DESTINATION lib)
if(EXISTS ${CMAKE_SOURCE_DIR}/../model)
    install(DIRECTORY ../model DESTINATION ./)
endif()
install(DIRECTORY include/api include/core include/models include/utils DESTINATION include)
install(FILES include/rknn_api.h DESTINATION include)
