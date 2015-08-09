#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

#include <xmmintrin.h>

#include "util.h"

static const unsigned long n = 1 << 24;

// x y z w x y z w ...
void mul_scalar(float* matrix, float* xs, float* ys)
{
  for (std::remove_const<decltype(n)>::type k = 0; k < n; ++k)
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

// x y z w x y z w ...
void mul_sse2_bad(float* matrix, float* xs, float* ys)
{
  for (std::remove_const<decltype(n)>::type k = 0; k < n; ++k)
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

// From SSE 4.1
#define _mm_extract_ps(a, i) (_mm_cvtss_f32(_mm_shuffle_ps(a, a, i)))

// x y z w x y z w ...
void mul_sse2_notquite(float* matrix, float* xs, float* ys)
{
  for (std::remove_const<decltype(n)>::type k = 0; k < n; k += 4)
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
      ys[4*(k+0) + i] = (float)(_mm_extract_ps(a, 3));
      ys[4*(k+1) + i] = (float)(_mm_extract_ps(a, 2));
      ys[4*(k+2) + i] = (float)(_mm_extract_ps(a, 1));
      ys[4*(k+3) + i] = (float)(_mm_extract_ps(a, 0));
    }
  }
}

// x x x x y y y y z z z z w w w w x x x x ...
void mul_sse2(float* matrix, float* xs, float* ys)
{
  for (std::remove_const<decltype(n)>::type k = 0; k < n; k += 4)
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

#include <boost/simd/include/functions/load.hpp>
#include <boost/simd/include/functions/multiplies.hpp>
#include <boost/simd/include/functions/plus.hpp>
#include <boost/simd/include/functions/aligned_store.hpp>
#include <boost/simd/sdk/simd/pack.hpp>

template <unsigned vecWidth>
void mul_boost(float* matrix, float* xs, float* ys)
{
  typedef boost::simd::pack<float, vecWidth> pack;

  for (std::remove_const<decltype(n)>::type k = 0; k < n; k += vecWidth)
  {
    for (unsigned i = 0; i < 4; ++i)
    {
      auto a = pack(0);
      for (unsigned j = 0; j < 4; ++j)
      {
        auto tempM = pack(matrix[4*i + j]);
        auto tempX = boost::simd::aligned_load<pack>(xs + 4*k + 4*j);
        a = a + tempM * tempX;
      }
      boost::simd::aligned_store<pack>(a, ys + 4*k + 4*i);
    }
  }
}

#ifdef ISPC
#include "matrix_ispc.h"
#endif

////////////////////////////////////////////////////////////////////////////////

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
#else
  out << std::accumulate(ys, ys+4*n, 0) << std::endl;
#endif
}

////////////////////////////////////////////////////////////////////////////////

int main()
{
  std::cout << "type,t" << std::endl;
  for (int i = 0; i < 9; ++i)
  {
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

    {
      auto outScalar = aligned_new<float>(4*n);
      {
        Benchmark b("Scalar");
        mul_scalar(matrix, in, outScalar);
      }
      dump_aos("matrix_scalar.txt", outScalar, n);
      delete[] outScalar;
    }

    {
      auto outSse2Bad = aligned_new<float>(4*n);
      {
        Benchmark b("Naive");
        mul_sse2_bad(matrix, in, outSse2Bad);
      }
      dump_aos("matrix_sse2Bad.txt", outSse2Bad, n);
      delete[] outSse2Bad;
    }

    {
      auto outSse2Notquite = aligned_new<float>(4*n);
      {
        Benchmark b("SPMD AOS");
        mul_sse2_notquite(matrix, in, outSse2Notquite);
      }
      dump_aos("matrix_sse2NotQuite.txt", outSse2Notquite, n);
      delete[] outSse2Notquite;
    }

    {
      auto outSse2 = aligned_new<float>(4*n);
      {
        Benchmark b("SPMD SOA");
        mul_sse2(matrix, inSoa, outSse2);
      }
      dump_soa("matrix_sse2.txt", outSse2, n);
      delete[] outSse2;
    }

#if 0
    {
      auto outBoost2 = aligned_new<float>(4*n);
      {
        Benchmark b;
        mul_boost<2>(matrix, inSoa, outBoost2);
      }
      dump_soa("matrix_boost2.txt", outBoost2, n);
      delete[] outBoost2;
    }
#endif

    {
      auto outBoost4 = aligned_new<float>(4*n);
      {
        Benchmark b("Boost.SIMD");
        mul_boost<4>(matrix, inSoa, outBoost4);
      }
      dump_soa("matrix_boost4.txt", outBoost4, n);
      delete[] outBoost4;
    }

    {
      // As a baseline, hard to be fater than memcpy.
      auto outMemcpy = aligned_new<float>(4*n);
      {
        Benchmark b("memcpy");
        std::copy(
          inSoa,
          inSoa + 4*n,
          outMemcpy);
      }
      dump_soa("matrix_memcpy.txt", outMemcpy, n);
      delete[] outMemcpy;
    }

#ifdef AVX
    {
      auto outBoost8 = aligned_new<float>(8*n);
      {
        Benchmark b;
        mul_boost<8>(matrix, inSoa, outBoost8);
      }
      dump_soa("matrix_boost8.txt", outBoost8, n);
    }
#endif

#ifdef ISPC
    {
      auto outIspc = aligned_new<float>(4*n);
      {
        Benchmark b;
        ispc::mul_ispc(matrix, inSoa, outIspc, n);
      }
      dump_soa("matrix_ispc.txt", outIspc, n);
    }
#endif

  }
}
