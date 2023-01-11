#include "test.h"
#include "random.hpp"

using namespace cytnx;

struct timespec t_start, t_end;
clock_t start, end;

struct timespec diff(struct timespec start, struct timespec end) {
  struct timespec temp;

  if (end.tv_sec - start.tv_sec == 0) {
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  } else {
    temp.tv_nsec = ((end.tv_sec - start.tv_sec) * 1000000000) + end.tv_nsec - start.tv_nsec;
  }

  return temp;
}

int main(void) {
  const int _n = 3;
  std::vector<Bond> bds1(_n);
  for (int i = 0; i < _n; i++) {
    if (i & 1)
      bds1[i] = Bond(5, bondType::BD_KET, {{1}, {0}, {0}, {-1}, {-1}});
    else
      bds1[i] = Bond(5, bondType::BD_BRA, {{1}, {0}, {0}, {-1}, {-1}});
  }
  std::vector<Bond> bds2(_n);
  for (int i = 0; i < _n; i++) {
    if (i & 1)
      bds2[i] = Bond(5, bondType::BD_BRA, {{1}, {0}, {0}, {-1}, {-1}});
    else
      bds2[i] = Bond(5, bondType::BD_KET, {{1}, {0}, {0}, {-1}, {-1}});
  }
  std::vector<cytnx_int64> lbl;
  for (int i = 0; i < _n; i++) {
    lbl.push_back(i);
  }
  UniTensor ut1 = UniTensor(bds1, lbl);
  UniTensor ut2 = UniTensor(bds2, lbl);

  clock_gettime(CLOCK_MONOTONIC, &t_start);

  ut1.contract(ut2);

  clock_gettime(CLOCK_MONOTONIC, &t_end);
  printf("%lf\n", (double)(diff(t_start, t_end).tv_nsec / 1000000000.0));
}
