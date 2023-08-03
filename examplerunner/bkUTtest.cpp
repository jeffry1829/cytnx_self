#include "cytnx.hpp"
#include <iostream>

using namespace std;
using namespace cytnx;

int main() {
  Bond I = Bond(BD_IN,{Qs(-4),Qs(-2),Qs(0),Qs(2),Qs(4)},{2,7,10,8,3});
  Bond J = Bond(BD_OUT,{Qs(1),Qs(-1)},{1,1});
  Bond K = Bond(BD_OUT,{Qs(1),Qs(-1)},{1,1});
  Bond L = Bond(BD_OUT,{Qs(-4),Qs(-2),Qs(0),Qs(2),Qs(4),Qs(6)},{1,5,10,9,4,1});
  UniTensor cyT = UniTensor({I,J,K,L},{"a","b","c","d"},2,Type.Double,Device.cpu,false);
  cyT = UniTensor::Load("Svd_truncate3.cytnx");
  std::vector<UniTensor> res =  linalg::Svd_truncate(cyT, 30, 0, true, true, true);
  // cout<<res.size()<<endl;
  // for(size_t i=0;i<res.size();i++){
  //   res[i].print_diagram();
  //   // res[i].print_blocks();
  // }
  auto con_T1 = Contract(Contract(res[2], res[0]), res[1]);
  auto con_T2 = Contract(Contract(res[1], res[0]), res[2]);
}