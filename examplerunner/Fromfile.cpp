#include "cytnx.hpp"
#include <iostream>
using namespace cytnx;
using namespace std;

int main(int argc, char* argv[]) {
  Network N;
  N.Fromfile("example.net");
  UniTensor ta = UniTensor(arange(24).reshape({2,3,4}));
  UniTensor tb = UniTensor(arange(24).reshape({2,3,4}));
  UniTensor tc = UniTensor(arange(24).reshape({2,3,4}));
  UniTensor td = UniTensor(arange(24).reshape({2,3,4}));

  N.PutUniTensors({"A","B","C","D"},{ta,tb,tc,td});
  
  N.Launch(true);

  cout << N << endl;
}
