# CMakeList.txt: 最上層的 CMake 專案檔，執行全域設定
# 並在這裡包含子專案。
#
cmake_minimum_required (VERSION 3.8)

project ("ncon" LANGUAGES CXX)

# 包含子專案。
#add_subdirectory ("Cytnx_test")

#execute_process(COMMAND echo $CYTNX_INC OUTPUT_VARIABLE CYTNX_INC)
#execute_process(COMMAND echo $CYTNX_LDFLAGS OUTPUT_VARIABLE CYTNX_LDFLAGS)
#execute_process(COMMAND echo $CYTNX_LIB OUTPUT_VARIABLE CYTNX_LIB)
#execute_process(COMMAND echo $CYTNX_CXXFLAGS OUTPUT_VARIABLE CYTNX_CXXFLAGS)



#set(CMAKE_CXX_COMPILER "g++")
#set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD_REQUIRED True)
#set(CMAKE_CXX_FLAGS ${CYTNX_CXXFLAGS})
#message(STATUS "CMAKE_CXX_FLAGS: " "${CMAKE_CXX_FLAGS}")
#string(REPLACE ";" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
#message(STATUS "CMAKE_CXX_FLAGS: " "${CMAKE_CXX_FLAGS}")

#include_directories(lib/ definition/)

aux_source_directory(. SOURCE_DIR)

#add_executable(run ${SOURCE_DIR})
add_executable(run main.cpp ncon.cpp)

# These explict includes are for Intellise
#set(CYTNX_INC "/mnt/c/Users/petjelinux/source/repos/Cytnx_Self/include")
#set(CYTNX_LIB "/mnt/c/Users/petjelinux/source/repos/Cytnx_Self/lib/libcytnx.a")
#set(CYTNX_CXXFLAGS "-I/home/petjelinux/anaconda3/envs/cytnx/include -fvisibility-inlines-hidden -std=c++17 -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /home/petjelinux/anaconda3/envs/cytnx/include -Wformat=0 -Wno-c++11-narrowing -w -fopenmp -std=c++11 -DUNI_OMP -DUNI_MKL -DUNI_HPTT")
#set(CYTNX_LDFLAGS "/home/petjelinux/anaconda3/envs/cytnx/lib/libmkl_intel_ilp64.so /home/petjelinux/anaconda3/envs/cytnx/lib/libmkl_intel_thread.so /home/petjelinux/anaconda3/envs/cytnx/lib/libmkl_core.so /home/petjelinux/anaconda3/envs/cytnx/lib/libiomp5.so -lpthread -lm -ldl -lpthread -lm -ldl -Wl,-rpath,/home/petjelinux/anaconda3/envs/cytnx/lib /mnt/c/Users/petjelinux/source/repos/Cytnx_Self/hptt/lib/libhptt.a")

#include_directories("/mnt/c/Users/petjelinux/source/repos/Cytnx_Self/include")

target_include_directories(run PUBLIC $ENV{CYTNX_INC})
#target_link_directories(run PUBLIC "${CYTNX_LIB}")
#target_link_libraries(run PUBLIC "$ENV{CYTNX_LIB}")
target_compile_options(run PUBLIC -g -ggdb -rdynamic -O0 $ENV{CYTNX_CXXFLAGS} -O0)
target_link_libraries(run PUBLIC $ENV{CYTNX_LIB})
set(LIBS "$ENV{CYTNX_LDFLAGS}")
string(REPLACE " " ";" LIBS_LIST ${LIBS})
target_link_libraries(run PUBLIC ${LIBS_LIST})

message(STATUS "$ENV{CYTNX_INC}")
message(STATUS "$ENV{CYTNX_LIB}")
message(STATUS "$ENV{CYTNX_CXXFLAGS}")
message(STATUS "$ENV{CYTNX_LDFLAGS}")

#target_link_libraries(run PUBLIC libs)
