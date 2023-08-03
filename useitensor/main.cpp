#include "main.h"

using namespace itensor;
using namespace cytnx;
using namespace std;

int main() {
  UniTensor ut1 = UniTensor(zeros(2 * 3 * 4 * 5)).to(Device.cuda);
  return 0;
  Index three = Index(3, "three");
  Index four = Index(4, "four");
  Index five = Index(5, "five");
  Index six = Index(6, "six");
  Index six1 = Index(6, "six1");
  Index six2 = Index(6, "six2");
  auto T = randomITensorC(three, four, five, six);
  // auto T = readFromFile<ITensor>("densetensor1.txt");
  writeToFile("densetensor1.txt", T);
  auto T2 = randomITensorC(three, four, five, six);
  // auto T2 = readFromFile<ITensor>("densetensor2.txt");
  writeToFile("densetensor2.txt", T2);
  // auto T3 = randomITensorC(four, six);
  // auto T3 = readFromFile<ITensor>("densetensor3.txt");
  // writeToFile("densetensor3.txt", T3);
  // auto T4 = randomITensorC(three, four, five, six);
  // auto T4 = readFromFile<ITensor>("densetensor4.txt");
  // writeToFile("densetensor4.txt", T4);

  // auto T = randomITensorC(six1, four, five, six2);
  // writeToFile("dense4trtensor.txt", T);

  UniTensor UT = UniTensor(zeros(3 * 4 * 5 * 6)).reshape({3, 4, 5, 6}).astype(Type.ComplexDouble);
  UniTensor UT2 = UniTensor(zeros(3 * 4 * 5 * 6)).reshape({3, 4, 5, 6}).astype(Type.ComplexDouble);
  // UniTensor UT3 = UniTensor(zeros(4 * 6)).reshape({4, 6}).astype(Type.ComplexDouble);
  // UniTensor UT4 = UniTensor(zeros(3 * 4 * 5 * 6)).reshape({3, 4, 5,
  // 6}).astype(Type.ComplexDouble); UniTensor UT = UniTensor(zeros(6 * 4 * 5 * 6)).reshape({6, 4,
  // 5, 6}).astype(Type.ComplexDouble);

  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++) UT.at({i - 1, j - 1, k - 1, l - 1}) = eltC(T, i, j, k, l);
  for (size_t i = 1; i <= 3; i++)
    for (size_t j = 1; j <= 4; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++) UT2.at({i - 1, j - 1, k - 1, l - 1}) = eltC(T2, i, j, k, l);
  // for(size_t j=1;j<=4;j++)for(size_t l=1;l<=6;l++)
  //   UT3.at({j-1,l-1}) = eltC(T3,j,l);
  // for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
  //   for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
  //     UT4.at({i-1,j-1,k-1,l-1}) = eltC(T4,i,j,k,l);

  // for(size_t i=1;i<=6;i++)for(size_t j=1;j<=4;j++)
  //   for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
  //     UT.at({i-1,j-1,k-1,l-1}) = eltC(T,i,j,k,l);

  // ITensor cont1 = T * prime(T2,IndexSet(four,five,six));
  // ITensor cont2 = T * prime(T2,IndexSet(five,six));
  // ITensor cont3 = T * prime(T2,IndexSet(six));

  ITensor cont1 = prime(T, IndexSet(four, five, six)) * T2;
  ITensor cont2 = prime(T, IndexSet(five, six)) * T2;
  ITensor cont3 = prime(T, IndexSet(six)) * T2;

  // ITensor trT = T * delta(six1, six2);
  // ITensor permu1 = permute(T4, IndexSet(four, three, six, five));
  // ITensor permu2 = permute(T3, IndexSet(six, four));

  UniTensor UTcont1 =
    UniTensor(zeros(4 * 5 * 6 * 4 * 5 * 6)).reshape({4, 5, 6, 4, 5, 6}).astype(Type.ComplexDouble);
  UniTensor UTcont2 =
    UniTensor(zeros(5 * 6 * 5 * 6)).reshape({5, 6, 5, 6}).astype(Type.ComplexDouble);
  UniTensor UTcont3 = UniTensor(zeros(6 * 6)).reshape({6, 6}).astype(Type.ComplexDouble);
  // UniTensor UTcont3 = UniTensor(zeros(6*6)).reshape({6,6}).astype(Type.ComplexDouble);
  // UniTensor UTpermu1 = UniTensor(zeros(3*4*5*6)).reshape({4,3,6,5}).astype(Type.ComplexDouble);
  // UniTensor UTpermu2 = UniTensor(zeros(4*6)).reshape({6,4}).astype(Type.ComplexDouble);

  // for(size_t i=1;i<=3;i++)for(size_t j=1;j<=4;j++)
  //   for(size_t k=1;k<=5;k++)for(size_t l=1;l<=6;l++)
  //     UTpermu1.at({j-1,i-1,l-1,k-1}) = eltC(permu1,j,i,l,k);
  // for(size_t j=1;j<=4;j++)for(size_t l=1;l<=6;l++)
  //   UTpermu2.at({l-1,j-1}) = eltC(permu2,l,j);

  // UniTensor UTtr = UniTensor(zeros(4*5)).reshape({4,5}).astype(Type.ComplexDouble);

  // PrintData(cont1);
  // PrintData(cont2);
  PrintData(cont3);

  // for(size_t i=1;i<=4;i++)for(size_t j=1;j<=5;j++)
  //   UTtr.at({i-1,j-1}) = eltC(trT,i,j);

  for (size_t i = 1; i <= 4; i++)
    for (size_t j = 1; j <= 5; j++)
      for (size_t k = 1; k <= 6; k++)
        for (size_t l = 1; l <= 4; l++)
          for (size_t m = 1; m <= 5; m++)
            for (size_t n = 1; n <= 6; n++)
              UTcont1.at({i - 1, j - 1, k - 1, l - 1, m - 1, n - 1}) =
                eltC(cont1, i, j, k, l, m, n);
  for (size_t i = 1; i <= 5; i++)
    for (size_t j = 1; j <= 6; j++)
      for (size_t k = 1; k <= 5; k++)
        for (size_t l = 1; l <= 6; l++)
          UTcont2.at({i - 1, j - 1, k - 1, l - 1}) = eltC(cont2, i, j, k, l);
  for (size_t k = 1; k <= 6; k++)
    for (size_t l = 1; l <= 6; l++) UTcont3.at({k - 1, l - 1}) = eltC(cont3, k, l);

  // UT.Save("dense4trtensor");
  // UTtr.Save("densetr");
  UT.Save("denseutensor1");
  UT2.Save("denseutensor2");
  // UT3.Save("denseutensor3");
  // UT4.Save("denseutensor4");
  UTcont1.Save("densecontres1");
  UTcont2.Save("densecontres2");
  UTcont3.Save("densecontres3");
  // UTpermu1.Save("densepermu1");
  // UTpermu2.Save("densepermu2");

  return 0;
  // auto I = Index(QN(-1), 2, QN(0), 1, QN(1), 2, In, "I");
  // auto J = Index(QN(-1), 4, QN(0), 3, QN(+1), 4, Out, "J");
  // auto K = Index(QN(-1), 1, QN(0), 1, QN(+2), 1, In, "K");
  // auto L = Index(QN(-1), 2, QN(0), 1, QN(+1), 2, Out, "L");
  // // auto T = randomITensorC(QN(0), I, J, K, L);
  // auto T = readFromFile<ITensor>("tensor.txt");

  // // auto T2 = randomITensorC(QN(0), I, J, K, L);
  // auto T2 = readFromFile<ITensor>("tensor2.txt");

  // // auto trT = readFromFile<ITensor>("trace.txt");
  // // PrintData(trT);
  // // PrintData(T);
  // // PrintData(delta(I, L));

  // // auto trT = T * delta(I.dag(), L.dag());

  // // PrintData(trT);
  // // PrintData(qn(inds(T)(1),1));

  // auto conjT = T.conj();

  // Real normT = norm(T);
  // cout<<setprecision(16)<<normT;

  // auto TpT2 = T + T2;
  // auto TsT2 = T - T2;
  // auto Tm9 = T * 9;
  // auto Td9 = T / 9;
  // auto TdT2 = T / T2;

  // // writeToFile("tensor.txt", T);
  // // writeToFile("tensor2.txt", T2);

  // Bond B1 = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {2, 1, 2});
  // Bond B2 = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {4, 3, 4});
  // Bond B3 = Bond(BD_IN, {Qs(-1), Qs(0), Qs(2)}, {1, 1, 1});
  // Bond B4 = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {2, 1, 2});
  // UniTensor BUT = UniTensor({B1, B2, B3, B4},{"0","1","2","3"},-1,Type.ComplexDouble);
  // UniTensor BUT2 = UniTensor({B1, B2, B3, B4},{"0","1","2","3"},-1,Type.ComplexDouble);
  // UniTensor BUconjT = UniTensor({B1, B2, B3, B4},{"0","1","2","3"},-1,Type.ComplexDouble);
  // UniTensor BUtrT = UniTensor({B2, B3},vector<string>{"0","1"},-1,Type.ComplexDouble);
  // UniTensor BUTpT2 = UniTensor({B1, B2, B3, B4},{"0","1","2","3"},-1,Type.ComplexDouble);
  // UniTensor BUTsT2 = UniTensor({B1, B2, B3, B4},{"0","1","2","3"},-1,Type.ComplexDouble);
  // UniTensor BUTm9 = UniTensor({B1, B2, B3, B4},{"0","1","2","3"},-1,Type.ComplexDouble);
  // UniTensor BUTd9 = UniTensor({B1, B2, B3, B4},{"0","1","2","3"},-1,Type.ComplexDouble);
  // UniTensor BUTdT2 = UniTensor({B1, B2, B3, B4},{"0","1","2","3"},-1,Type.ComplexDouble);

  // // Range rg = Range(2,1,2,4,3,4,1,1,1,2,1,2);
  // // for(auto& ind : rg){
  // //   vector<cytnx_uint64> vi,vi1;
  // //   for(int i=0;i<(int)ind.size();i++)vi1.push_back(ind[i]+1);
  // //   for(int i=0;i<(int)ind.size()-2;i+=3)vi.push_back(ind[i]+ind[i+1]+ind[i+2]);
  // //   if(BUT.at(vi).exists()){
  // //     BUT.at(vi) = elt(T,vi1);
  // //   }
  // // }

  // // rg = Range(2,1,2,2,1,2);
  // // for(auto& ind : rg){
  // //   vector<cytnx_uint64> vi,vi1;
  // //   for(int i=0;i<(int)ind.size();i++)vi1.push_back(ind[i]+1);
  // //   for(int i=0;i<(int)ind.size()-2;i+=3)vi.push_back(ind[i]+ind[i+1]+ind[i+2]);
  // //   if(BUtrT.at(vi).exists()){
  // //     BUtrT.at(vi) = elt(trT,vi1);
  // //   }
  // // }

  // for(size_t i=1;i<=5;i++)for(size_t j=1;j<=11;j++)
  //     for(size_t k=1;k<=3;k++)for(size_t l=1;l<=5;l++)
  //         // if(BUT.at({i-1,j-1,k-1,l-1}).exists()){
  //         if(BUT.elem_exists({i-1,j-1,k-1,l-1})){
  //           // vec_print(cout,vector<size_t>({i,j,k,l}));
  //           BUT.at({i-1,j-1,k-1,l-1}) = eltC(T,i,j,k,l);
  //         }

  // for(size_t i=1;i<=5;i++)for(size_t j=1;j<=11;j++)
  //     for(size_t k=1;k<=3;k++)for(size_t l=1;l<=5;l++)
  //         if(BUT2.at({i-1,j-1,k-1,l-1}).exists()){
  //         BUT2.at({i-1,j-1,k-1,l-1}) = eltC(T2,i,j,k,l);
  //       }

  // for(size_t i=1;i<=5;i++)for(size_t j=1;j<=11;j++)
  //     for(size_t k=1;k<=3;k++)for(size_t l=1;l<=5;l++)
  //         if(BUconjT.at({i-1,j-1,k-1,l-1}).exists()){
  //         BUconjT.at({i-1,j-1,k-1,l-1}) = eltC(conjT,i,j,k,l);
  //       }

  // for(size_t i=1;i<=5;i++)for(size_t j=1;j<=11;j++)
  //     for(size_t k=1;k<=3;k++)for(size_t l=1;l<=5;l++)
  //         if(BUTpT2.at({i-1,j-1,k-1,l-1}).exists()){
  //         BUTpT2.at({i-1,j-1,k-1,l-1}) = eltC(TpT2,i,j,k,l);
  //       }

  // for(size_t i=1;i<=5;i++)for(size_t j=1;j<=11;j++)
  //     for(size_t k=1;k<=3;k++)for(size_t l=1;l<=5;l++)
  //         if(BUTsT2.at({i-1,j-1,k-1,l-1}).exists()){
  //         BUTsT2.at({i-1,j-1,k-1,l-1}) = eltC(TsT2,i,j,k,l);
  //       }

  // for(size_t i=1;i<=5;i++)for(size_t j=1;j<=11;j++)
  //     for(size_t k=1;k<=3;k++)for(size_t l=1;l<=5;l++)
  //         if(BUTm9.at({i-1,j-1,k-1,l-1}).exists()){
  //         BUTm9.at({i-1,j-1,k-1,l-1}) = eltC(Tm9,i,j,k,l);
  //       }

  // for(size_t i=1;i<=5;i++)for(size_t j=1;j<=11;j++)
  //     for(size_t k=1;k<=3;k++)for(size_t l=1;l<=5;l++)
  //         if(BUTd9.at({i-1,j-1,k-1,l-1}).exists()){
  //         BUTd9.at({i-1,j-1,k-1,l-1}) = eltC(Td9,i,j,k,l);
  //       }

  // for(size_t i=1;i<=5;i++)for(size_t j=1;j<=11;j++)
  //     for(size_t k=1;k<=3;k++)for(size_t l=1;l<=5;l++)
  //         if(BUTdT2.at({i-1,j-1,k-1,l-1}).exists()){
  //         BUTdT2.at({i-1,j-1,k-1,l-1}) = eltC(TdT2,i,j,k,l);
  //       }

  // // for(size_t j=1;j<=11;j++)
  // //     for(size_t k=1;k<=3;k++)
  // //       if(BUtrT.at({j-1,k-1}).exists()){
  // //         BUtrT.at({j-1,k-1}) = eltC(trT,j,k);
  // //       }

  // BUT.Save("OriginalBUT");
  // BUT2.Save("OriginalBUT2");

  // // BUtrT.Save("BUtrT");
  // BUconjT.Save("BUconjT");
  // BUTpT2.Save("BUTpT2");
  // BUTsT2.Save("BUTsT2");
  // BUTm9.Save("BUTm9");
  // BUTd9.Save("BUTd9");
  // BUTdT2.Save("BUTdT2");

  // return 0;
}
