#!/bin/bash
${CXX} -I${CYTNX_INC} ${CYTNX_CXXFLAGS} main.cpp ${CYTNX_LIB} ${CYTNX_LDFLAGS} -o run
