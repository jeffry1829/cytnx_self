#include "main.h"

using namespace itensor;
using namespace cytnx;
using namespace std;

int main() {
  auto I = Index(QN(-1), 2, QN(0), 1, QN(1), 2, In, "I");
  auto J = Index(QN(-1), 4, QN(0), 3, QN(+1), 4, Out, "J");
  auto K = Index(QN(-1), 1, QN(0), 1, QN(+2), 1, In, "K");
  auto L = Index(QN(-1), 2, QN(0), 1, QN(+1), 2, Out, "L");
  auto T = randomITensor(QN(0), I, J, K, L);
  // auto T = readFromFile<ITensor>("testTensor.txt");
  // PrintData(T);
  // PrintData(delta(I, L));
  auto trT = T * delta(I.dag(), L.dag());
  // PrintData(trT);
  // PrintData(qn(inds(T)(1),1));

  writeToFile("tensor.txt", T);
  writeToFile("trace.txt", trT);

  Bond B1 = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {2, 1, 2});
  Bond B2 = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {4, 3, 4});
  Bond B3 = Bond(BD_IN, {Qs(-1), Qs(0), Qs(2)}, {1, 1, 1});
  Bond B4 = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {2, 1, 2});
  UniTensor BUT = UniTensor({B1, B2, B3, B4});
  UniTensor BUtrT = UniTensor({B2, B3});

  // Range rg = Range(2,1,2,4,3,4,1,1,1,2,1,2);
  // for(auto& ind : rg){
  //   vector<cytnx_uint64> vi,vi1;
  //   for(int i=0;i<(int)ind.size();i++)vi1.push_back(ind[i]+1);
  //   for(int i=0;i<(int)ind.size()-2;i+=3)vi.push_back(ind[i]+ind[i+1]+ind[i+2]);
  //   if(BUT.at(vi).exists()){
  //     BUT.at(vi) = elt(T,vi1);
  //   }
  // }

  // rg = Range(2,1,2,2,1,2);
  // for(auto& ind : rg){
  //   vector<cytnx_uint64> vi,vi1;
  //   for(int i=0;i<(int)ind.size();i++)vi1.push_back(ind[i]+1);
  //   for(int i=0;i<(int)ind.size()-2;i+=3)vi.push_back(ind[i]+ind[i+1]+ind[i+2]);
  //   if(BUtrT.at(vi).exists()){
  //     BUtrT.at(vi) = elt(trT,vi1);
  //   }
  // }
  
  for(size_t i=1;i<=2;i++)for(size_t j=1;j<=1;j++)
      for(size_t k=1;k<=2;k++)for(size_t l=1;l<=4;l++)
          if(BUT.at({i-1,j-1,k-1,l-1}).exists()){
          BUT.at({i-1,j-1,k-1,l-1}) = elt(T,i,j,k,l);
        }

  for(size_t j=1;j<=11;j++)
      for(size_t k=1;k<=3;k++)
        if(BUtrT.at({j-1,k-1}).exists()){
          BUtrT.at({j-1,k-1}) = elt(trT,j,k);
        }

  BUT.Save("OriginalBUT");
  BUtrT.Save("BUtrT");

  return 0;
}
