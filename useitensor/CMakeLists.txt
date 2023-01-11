# CMakeList.txt: 最上層的 CMake 專案檔，執行全域設定
# 並在這裡包含子專案。
#
cmake_minimum_required (VERSION 3.8)

project ("ncon" LANGUAGES CXX)

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
add_executable(run main.cpp)

# These explict includes are for Intellise
#set(CYTNX_INC "/mnt/c/Users/petjelinux/source/repos/Cytnx_Self/include")
#set(CYTNX_LIB "/mnt/c/Users/petjelinux/source/repos/Cytnx_Self/lib/libcytnx.a")
#set(CYTNX_CXXFLAGS "-I/home/petjelinux/anaconda3/envs/cytnx/include -fvisibility-inlines-hidden -std=c++17 -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /home/petjelinux/anaconda3/envs/cytnx/include -Wformat=0 -Wno-c++11-narrowing -w -fopenmp -std=c++11 -DUNI_OMP -DUNI_MKL -DUNI_HPTT")
#set(CYTNX_LDFLAGS "/home/petjelinux/anaconda3/envs/cytnx/lib/libmkl_intel_ilp64.so /home/petjelinux/anaconda3/envs/cytnx/lib/libmkl_intel_thread.so /home/petjelinux/anaconda3/envs/cytnx/lib/libmkl_core.so /home/petjelinux/anaconda3/envs/cytnx/lib/libiomp5.so -lpthread -lm -ldl -lpthread -lm -ldl -Wl,-rpath,/home/petjelinux/anaconda3/envs/cytnx/lib /mnt/c/Users/petjelinux/source/repos/Cytnx_Self/hptt/lib/libhptt.a")

#include_directories("/mnt/c/Users/petjelinux/source/repos/Cytnx_Self/include")

######################################################################
### Get Gtest
######################################################################
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
######################################################################

target_include_directories(run PUBLIC $ENV{CYTNX_INC})
#target_link_directories(run PUBLIC "${CYTNX_LIB}")
#target_link_libraries(run PUBLIC "$ENV{CYTNX_LIB}")
target_compile_options(run PUBLIC -g -ggdb -rdynamic -O0 $ENV{CYTNX_CXXFLAGS} -O0)
target_link_libraries(run PUBLIC $ENV{CYTNX_LIB})
set(LIBS "$ENV{CYTNX_LDFLAGS}")
string(REPLACE " " ";" LIBS_LIST ${LIBS})
target_link_libraries(run PUBLIC ${LIBS_LIST})

#itensor
target_include_directories(run PUBLIC /home/petjelinux/itensor)
target_include_directories(run PUBLIC /home/petjelinux/itensor/itensor)
set(LIBS "-L/home/petjelinux/itensor/lib -litensor -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_rt -lmkl_core -liomp5 -lpthread")
string(REPLACE " " ";" LIBS_LIST2 ${LIBS})
target_link_libraries(run PUBLIC ${LIBS_LIST2})

target_link_libraries(run PUBLIC -lgcov --coverage -lrt)

#message(STATUS "${LIBS_LIST2}")

#message(STATUS "$ENV{CYTNX_INC}")
#message(STATUS "$ENV{CYTNX_LIB}")
#message(STATUS "$ENV{CYTNX_CXXFLAGS}")
#message(STATUS "$ENV{CYTNX_LDFLAGS}")

#target_link_libraries(run PUBLIC libs)
