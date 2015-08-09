#include <fstream>
#include <iostream>

#include <xmmintrin.h>

#include <boost/simd/include/functions/load.hpp>
#include <boost/simd/include/functions/multiplies.hpp>
#include <boost/simd/include/functions/plus.hpp>
#include <boost/simd/include/functions/aligned_store.hpp>
#include <boost/simd/sdk/simd/pack.hpp>

#include "util.h"

// Assumes AOS memory layout
void mul_scalar(float* matrix, float* xs, float* ys, unsigned long n)
{
  for (decltype(n) k = 0; k < n; ++k)
  {
    float* x = xs + 4*k;
    float* y = ys + 4*k;
    for (unsigned i = 0; i < 4; ++i)
    {
      float a = 0;
      for (unsigned j = 0; j < 4; ++j)
      {
        a += matrix[4*i + j] * x[j];
      }
      y[i] = a;
    }
  }
}

// Assumes AOS memory layout
void mul_sse2_bad(float* matrix, float* xs, float* ys, unsigned long n)
{
  for (decltype(n) k = 0; k < n; ++k)
  {
    auto x = _mm_load_ps(xs + 4*k);
    auto y = ys + 4*k;
    for (unsigned i = 0; i < 4; ++i)
    {
      auto row = _mm_load_ps(matrix + 4*i);
      auto t0 = _mm_mul_ps(x, row); // = [ a, b, c, d ] = [ m[i,0]*x[1], m[i,1]*x[1], m[i,2]*x[2], m[i,3]*x[3] ]
      auto t1 = _mm_shuffle_ps(t0, t0, 0x44); // = [ a, b, a, b ], 0b01000100
      auto t2 = _mm_shuffle_ps(t0, t0, 0xbb); // = [ c, d, c, d ], 0b10111011
      auto t3 = _mm_add_ps(t1, t2);       // = [ a+c, b+d, a+c, b+d ]
      auto t4 = _mm_shuffle_ps(t3, t3, 0xb1); // = [ b+d, a+c, b+d, a+c ] // 0b10110001
      auto t5 = _mm_add_ps(t3, t4);       // = [ a+c+b+d, b+d+a+c, a+c+b+d, b+d+a+c ]
      y[i] = _mm_cvtss_f32(t5);
    }
  }
}

// Assumes AOS memory layout
void mul_sse2_notquite(float* matrix, float* xs, float* ys, unsigned long n)
{
  for (decltype(n) k = 0; k < n; k += 4)
  {
    for (unsigned i = 0; i < 4; ++i)
    {
      auto a = _mm_set_ps1(0);
      for (unsigned j = 0; j < 4; ++j)
      {
        auto tempM = _mm_set_ps1(matrix[4*i + j]);
        auto tempX = _mm_set_ps(xs[4*(k+0) + j], xs[4*(k+1) + j], xs[4*(k+2) + j], xs[4*(k+3) + j]);
        a = _mm_add_ps(a, _mm_mul_ps(tempM, tempX));
      }
      ys[4*(k+0) + i] = extract<3>(a);
      ys[4*(k+1) + i] = extract<2>(a);
      ys[4*(k+2) + i] = extract<1>(a);
      ys[4*(k+3) + i] = extract<0>(a);
    }
  }
}

// Assumes Hybrid SOA memory layout
void mul_sse2(float* matrix, float* xs, float* ys, unsigned long n)
{
  for (decltype(n) k = 0; k < n; k += 4)
  {
    for (unsigned i = 0; i < 4; ++i)
    {
      auto a = _mm_set_ps1(0);
      for (unsigned j = 0; j < 4; ++j)
      {
        auto tempM = _mm_set_ps1(matrix[4*i + j]);
        auto tempX = _mm_load_ps(xs + 4*k + 4*j);
        a = _mm_add_ps(a, _mm_mul_ps(tempM, tempX));
      }
      _mm_store_ps(ys + 4*k + 4*i, a);
    }
  }
}

// Assumes Hybrid SOA memory layout
template <unsigned vecWidth>
void mul_boost(float* matrix, float* xs, float* ys, unsigned long n)
{
  typedef boost::simd::pack<float, vecWidth> pack;

  for (decltype(n) k = 0; k < n; k += vecWidth)
  {
    for (unsigned i = 0; i < 4; ++i)
    {
      auto a = pack(0);
      for (unsigned j = 0; j < 4; ++j)
      {
        pack tempM = pack(matrix[4*i + j]);
        pack tempX = boost::simd::aligned_load<pack>(xs + 4*k + 4*j);
        a = a + tempM * tempX;
      }
      boost::simd::aligned_store<pack>(a, ys + 4*k + 4*i);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

// Converts indices between AOS and Hybrid SOA. Goes both ways.
// Example:
// * in:   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
// * out:  0  4  8 12  1  5  9 13  2  6 10 14  3  7 11 15
unsigned long switchLayoutIndex(unsigned long aosIndex)
{
  return (aosIndex & ~0xf) | ((aosIndex & 0xc) / 4) | ((aosIndex & 0x3) * 4);
}

////////////////////////////////////////////////////////////////////////////////

#define DUMP_FULL 0

void dump_aos(const char* path, float* ys, unsigned long n)
{
  std::ofstream out(path);
#if DUMP_FULL
  std::copy(ys, ys+4*n, std::ostream_iterator<float>(out, " "));
  out << std::endl;
#else
  out << std::accumulate(ys, ys+4*n, 0) << std::endl;
#endif
}

void dump_soa(const char* path, float* ys, unsigned long n)
{
  std::ofstream out(path);
#if DUMP_FULL
  for (unsigned long i = 0; i < 4*n; ++i)
  {
    out << ys[switchLayoutIndex(i)] << " ";
  }
  out << std::endl;
#else
  out << std::accumulate(ys, ys+4*n, 0) << std::endl;
#endif
}

////////////////////////////////////////////////////////////////////////////////

int main()
{
  // Number of vectors. Must be a multiple of 4 since we are too lazy to handle
  // the other case.
  //const unsigned long n = 8; // For testing
  const unsigned long n = 1 << 24; // For benchmarking

  //const unsigned nBenchmarkRuns = 1; // For testing
  const unsigned nBenchmarkRuns = 9; // For benchmarking

  // Prepare input data

  auto matrix = aligned_new<float>(16);
  for (unsigned int i = 0; i < 16; ++i)
  {
    matrix[i] = i;
  }

  auto in = aligned_new<float>(4*n);
  for (std::remove_const<decltype(n)>::type i = 1; i < n; ++i)
  {
    in[4*i + 0] = i;
    in[4*i + 1] = i*i;
    in[4*i + 2] = i*i*i;
    in[4*i + 3] = i*i*i*i;
  }

  auto inSoa = aligned_new<float>(4*n);
  for (unsigned long i = 0; i < 4*n; ++i)
  {
    inSoa[i] = in[switchLayoutIndex(i)];
  }

  // Benchmark for a couple of iterations
  std::cout << "type,t" << std::endl;
  for (int i = 0; i < nBenchmarkRuns; ++i)
  {
    {
      auto outScalar = aligned_new<float>(4*n);
      {
        Benchmark b("Scalar");
        mul_scalar(matrix.get(), in.get(), outScalar.get(), n);
      }
      dump_aos("matrix_scalar.txt", outScalar.get(), n);
    }

    {
      auto outSse2Bad = aligned_new<float>(4*n);
      {
        Benchmark b("Naive");
        mul_sse2_bad(matrix.get(), in.get(), outSse2Bad.get(), n);
      }
      dump_aos("matrix_sse2Bad.txt", outSse2Bad.get(), n);
    }

    {
      auto outSse2Notquite = aligned_new<float>(4*n);
      {
        Benchmark b("SPMD AOS");
        mul_sse2_notquite(matrix.get(), in.get(), outSse2Notquite.get(), n);
      }
      dump_aos("matrix_sse2NotQuite.txt", outSse2Notquite.get(), n);
    }

    {
      auto outSse2 = aligned_new<float>(4*n);
      {
        Benchmark b("SPMD SOA");
        mul_sse2(matrix.get(), inSoa.get(), outSse2.get(), n);
      }
      dump_soa("matrix_sse2.txt", outSse2.get(), n);
    }

    {
      auto outBoost4 = aligned_new<float>(4*n);
      {
        Benchmark b("Boost.SIMD");
        mul_boost<4>(matrix.get(), inSoa.get(), outBoost4.get(), n);
      }
      dump_soa("matrix_boost4.txt", outBoost4.get(), n);
    }

    {
      // As a baseline, hard to be fater than memcpy.
      auto outMemcpy = aligned_new<float>(4*n);
      {
        Benchmark b("memcpy");
        std::copy(
          inSoa.get(),
          inSoa.get() + 4*n,
          outMemcpy.get());
      }
      dump_soa("matrix_memcpy.txt", outMemcpy.get(), n);
    }

  }
}
