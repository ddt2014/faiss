cmake_minimum_required(VERSION 2.8.7)

# faiss project
project(faiss C CXX)

option(BUILD_TUTORIAL "Build tutorials" ON)
option(BUILD_TEST "Build tests" ON)
option(BUILD_WITH_GPU "Build faiss with gpu (cuda) support" OFF)
option(WITH_MKL "Build with MKL if ON (OpenBLAS if OFF)" OFF)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# OpenMP
find_package(OpenMP REQUIRED)

# BLAS (MKL os OpenBLAS)
if(WITH_MKL)
    find_package(MKL REQUIRED)
    include_directories(${MKL_INCLUDE_DIRS})
    set(BLAS_LIB ${MKL_LIBRARIES})
else()
    find_package(OpenBLAS REQUIRED)
    include_directories(${OpenBLAS_INCLUDE_DIR})
    set(BLAS_LIB ${OpenBLAS_LIB})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -m64 -Wall -g -O3  -msse4 -mpopcnt -fopenmp -Wno-sign-compare")
add_definitions(-DFINTEGER=int)

# specify output bin_path and lib_path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# specify header and cpp files
file(GLOB faiss_cpu_headers ${CMAKE_CURRENT_SOURCE_DIR}/c_api/*.h)
file(GLOB faiss_cpu_cpp ${CMAKE_CURRENT_SOURCE_DIR}/c_api/*.cpp)

set(LIBPATH "~/tool/3rdparty/")
add_definitions(-DCPU_ONLY)
set(Caffe_DIR "${LIBPATH}/mcaffe/latest_until0703/cpu")
set(Caffe_LIB caffe_face_cpu)
set(FACE_LIB tuniface_cpu)
set(OpenCV_DIR "${LIBPATH}/opencv-3.3.0/share/OpenCV")
set(TBB_DIR "${LIBPATH}/tbb2018_20171205oss/cmake")
set(faiss_lib faiss)

find_package(TBB COMPONENTS tbb tbbmalloc)
find_package(OpenCV REQUIRED )
find_package(Boost COMPONENTS system filesystem regex REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${LIBPATH}/yas/include")
include_directories("${LIBPATH}/SimpleJSON")


link_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib")
link_directories("${Caffe_DIR}/lib")

add_library(${faiss_lib} STATIC ${faiss_cpu_headers} ${faiss_cpu_cpp})

target_link_libraries(${faiss_lib} ${OpenMP_CXX_FLAGS} ${BLAS_LIB} ${FACE_LIB} ${Caffe_LIB} ${SEARCH_LIB} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_IMPORTED_TARGETS})

# build gpu lib
if(BUILD_WITH_GPU)
    include(cmake/Cuda.cmake)
    add_subdirectory(gpu)
endif(BUILD_WITH_GPU)

# build tutorial examples
if(BUILD_TUTORIAL)
    add_subdirectory(tutorial)
endif(BUILD_TUTORIAL)

# build tests
#if(BUILD_TEST)
#    add_subdirectory(tests)
#endif(BUILD_TEST)
# Install libraries
install(TARGETS ${faiss_lib}
        ARCHIVE DESTINATION lib
        )
install(FILES ${faiss_cpu_headers} DESTINATION include/faiss)
