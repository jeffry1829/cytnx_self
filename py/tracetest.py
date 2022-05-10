import numpy as np
import os
import sys
sys.path.append('/home/petjelinux/Cytnx_lib')
import cytnx

os.environ["OMP_NUM_THREADS"] = "1"

#print(cytnx.__cpp_include__)
N,M = 2,1024
UT = cytnx.UniTensor(cytnx.from_numpy(np.random.rand(N,M,M,N)), rowrank=2)
UT.print_diagram()

I = cytnx.UniTensor(cytnx.from_numpy(np.eye(N)), rowrank=1)
UT_Trace_2 = UT.Trace(0,3)

UT_Trace_2.print_diagram()
