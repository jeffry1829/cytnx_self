#include <immintrin.h>
#include <bits/stdc++.h>

double a0[4] = {1.12, 2.22, 3.33, 4.44};
double a1[4] = {5.12, 6.22, 7.33, 8.44};
double a2[4] = {9.12, 10.22, 11.33, 12.44};
double a3[4] = {13.12, 14.22, 15.33, 16.44};
int main() {
  __m256d rowA0 = _mm256_loadu_pd(a0);
  __m256d rowA1 = _mm256_loadu_pd(a1);
  __m256d rowA2 = _mm256_loadu_pd(a2);
  __m256d rowA3 = _mm256_loadu_pd(a3);
  __m256d r4, r34, r3, r33;
  r33 = _mm256_shuffle_pd(rowA2, rowA3, 0x3);
  r3 = _mm256_shuffle_pd(rowA0, rowA1, 0x3);
  r34 = _mm256_shuffle_pd(rowA2, rowA3, 0xc);
  r4 = _mm256_shuffle_pd(rowA0, rowA1, 0xc);
  rowA0 = _mm256_permute2f128_pd(r34, r4, 0x2);
  rowA1 = _mm256_permute2f128_pd(r33, r3, 0x2);
  rowA2 = _mm256_permute2f128_pd(r33, r3, 0x13);
  rowA3 = _mm256_permute2f128_pd(r34, r4, 0x13);

  _mm256_storeu_pd(a0, rowA0);
  _mm256_storeu_pd(a1, rowA1);
  _mm256_storeu_pd(a2, rowA2);
  _mm256_storeu_pd(a3, rowA3);
  //   __m256d a = _mm256_loadu_pd(&b[0]);
  //   __m256d aa = _mm256_loadu_pd(&c[0]);
  //   a = _mm256_shuffle_pd(a, aa, 0x3);
  //   _mm256_storeu_pd(&b[0], a);
  for (int i = 0; i < 4; i++) {
    std::cout << a0[i] << ' ';
  }
  std::cout << std::endl;

  for (int i = 0; i < 4; i++) {
    std::cout << a1[i] << ' ';
  }
  std::cout << std::endl;
  for (int i = 0; i < 4; i++) {
    std::cout << a2[i] << ' ';
  }
  std::cout << std::endl;
  for (int i = 0; i < 4; i++) {
    std::cout << a3[i] << ' ';
  }
}
