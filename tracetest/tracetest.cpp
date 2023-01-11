//#pragma GCC optimize(1)
//#pragma GCC optimize(2)
/*#pragma GCC optimize(3)
#pragma GCC optimize("Ofast")
#pragma GCC optimize("inline")
#pragma GCC optimize("-fgcse")
#pragma GCC optimize("-fgcse-lm")
#pragma GCC optimize("-fipa-sra")
#pragma GCC optimize("-ftree-pre")
#pragma GCC optimize("-ftree-vrp")
#pragma GCC optimize("-fpeephole2")
#pragma GCC optimize("-ffast-math")
#pragma GCC optimize("-fsched-spec")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("-falign-jumps")
#pragma GCC optimize("-falign-loops")
#pragma GCC optimize("-falign-labels")
#pragma GCC optimize("-fdevirtualize")
#pragma GCC optimize("-fcaller-saves")
#pragma GCC optimize("-fcrossjumping")
#pragma GCC optimize("-fthread-jumps")
#pragma GCC optimize("-funroll-loops")
#pragma GCC optimize("-freorder-blocks")
#pragma GCC optimize("-fschedule-insns")
#pragma GCC optimize("inline-functions")
#pragma GCC optimize("-ftree-tail-merge")
#pragma GCC optimize("-fschedule-insns2")
#pragma GCC optimize("-fstrict-aliasing")
#pragma GCC optimize("-falign-functions")
#pragma GCC optimize("-fcse-follow-jumps")
#pragma GCC optimize("-fsched-interblock")
#pragma GCC optimize("-fpartial-inlining")
#pragma GCC optimize("no-stack-protector")
#pragma GCC optimize("-freorder-functions")
#pragma GCC optimize("-findirect-inlining")
#pragma GCC optimize("-fhoist-adjacent-loads")
#pragma GCC optimize("-frerun-cse-after-loop")
#pragma GCC optimize("inline-small-functions")
#pragma GCC optimize("-finline-small-functions")
#pragma GCC optimize("-ftree-switch-conversion")
#pragma GCC optimize("-foptimize-sibling-calls")
#pragma GCC optimize("-fexpensive-optimizations")
#pragma GCC optimize("inline-functions-called-once")
#pragma GCC optimize("-fdelete-null-pointer-checks")
#pragma comment(linker, "/STACK:1024000000,1024000000")*/
#include "tracetest.h"
using namespace cytnx;
using namespace std;
//#define int long long
#define rep(i, a, n) for (int i = a; i < n; i++)
#define per(i, a, n) for (int i = n - 1; i >= a; i--)
#define pb push_back
//#define mp make_pair
#define all(x) (x).begin(), (x).end()
#define fi first
#define se second
#define SZ(x) ((int)(x).size())
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define abs(x) (((x) < 0) ? (-(x)) : (x))
typedef vector<int> VI;
typedef long long ll;
typedef pair<int, int> PII;
typedef double db;
mt19937 mrand(random_device{}());
const ll mod = 1000000007;
int rnd(int x) { return mrand() % x; }
ll powmod(ll a, ll b) {
  ll res = 1;
  a %= mod;
  assert(b >= 0);
  for (; b; b >>= 1) {
    if (b & 1) res = res * a % mod;
    a = a * a % mod;
  }
  return res;
}
ll gcd(ll a, ll b) { return b ? gcd(b, a % b) : a; }
inline int pmod(int x, int d) {
  int m = x % d;
  return m + ((m >> 31) & d);
}
#define y1 ojsapogjahg
#define prev ojaposjdas
#define rank oiajgpowsdjg
#define left aijhgpiaejhgp
//#define end aononcncnccc
// head
#include <omp.h>
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
  const int N = 2, M = 3000;
  Tensor T = random::uniform(N * M * M * N, 10, 100);
  T.reshape_(N, M, M, N);
  UniTensor UT = UniTensor(T, false, 2);
  UT.print_diagram();

  clock_gettime(CLOCK_MONOTONIC, &t_start);

  UniTensor UT2 = UT.Trace(0, 3);
  UT2.print_diagram();
  
  clock_gettime(CLOCK_MONOTONIC, &t_end);
  printf("%lf\n", (double)(diff(t_start, t_end).tv_nsec / 1000000000.0));

  clock_gettime(CLOCK_MONOTONIC, &t_start);

  UniTensor I = UniTensor(ones(2), true, -1);
  I.set_labels({1,2});
  UT.set_labels({1,100,200,2});
  I.print_diagram();
  UT.print_diagram();
  Tensor out = Contract(I, UT).get_block_();
  UniTensor(out).print_diagram();
  
  clock_gettime(CLOCK_MONOTONIC, &t_end);
  printf("%lf\n", (double)(diff(t_start, t_end).tv_nsec / 1000000000.0));
  return 0;
}
