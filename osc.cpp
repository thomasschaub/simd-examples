#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <thread>

#include <xmmintrin.h>

#include <sndfile.h>

#include "util.h"

float Q_rsqrt( float number )
{
  long i;
  float x2, y;
  const float threehalfs = 1.5F;

  x2 = number * 0.5F;
  y  = number;
  i  = * ( long * ) &y;                       // evil floating point bit level hacking
  i  = 0x5f3759df - ( i >> 1 );               // what the fuck? 
  y  = * ( float * ) &i;
  y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
  //  y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

  return y;
}


class Osc_scalar
{
public:
  void setFreq(float f);

  float value();
  void step();

private:
  float real = 1, imag = 0;
  float dReal = 1, dImag = 0;
};

void Osc_scalar::setFreq(float f)
{
  dReal = cos(f);
  dImag = sin(f);
}

float Osc_scalar::value()
{
  return imag;
}

void Osc_scalar::step()
{
  float r = real*dReal - imag*dImag;
  float i = real*dImag + imag*dReal;

  auto l = r*r + i*i;
  auto f = 1/sqrt(l);

  real = f*r;
  imag = f*i;
}

class Osc_scalar_q
{
public:
  void setFreq(float f);

  float value();
  void step();

private:
  float real = 1, imag = 0;
  float dReal = 1, dImag = 0;
};

void Osc_scalar_q::setFreq(float f)
{
  dReal = cos(f);
  dImag = sin(f);
}

float Osc_scalar_q::value()
{
  return imag;
}

void Osc_scalar_q::step()
{
  float r = real*dReal - imag*dImag;
  float i = real*dImag + imag*dReal;

  auto l = r*r + i*i;
  auto f = Q_rsqrt(l);

  real = f*r;
  imag = f*i;
}

class Osc_sse2
{
public:
  void setFreq(int i, float f);

  __m128 value();
  void step();

private:
  float real[4] = {1, 1, 1, 1}, imag[4] = {0, 0, 0, 0};
  float dReal[4] = {1, 1, 1, 1}, dImag[4] = {0, 0, 0, 0};
};

void Osc_sse2::setFreq(int i, float f)
{
  dReal[i] = cos(f);
  dImag[i] = sin(f);
}

__m128 Osc_sse2::value()
{
  return _mm_load_ps(imag);
}

void Osc_sse2::step()
{
  auto r0 = _mm_load_ps(real);
  auto dr0 = _mm_load_ps(dReal);
  auto i0 = _mm_load_ps(imag);
  auto di0 = _mm_load_ps(dImag);
  auto r = _mm_sub_ps(_mm_mul_ps(r0, dr0), _mm_mul_ps(i0, di0));
  auto i = _mm_add_ps(_mm_mul_ps(r0, di0), _mm_mul_ps(i0, dr0));

  auto l = _mm_mul_ps(r, r) + _mm_mul_ps(i, i);
  auto f = _mm_rsqrt_ps(l);

  _mm_store_ps(real, _mm_mul_ps(f, r));
  _mm_store_ps(imag, _mm_mul_ps(f, i));
}

#include <boost/simd/include/functions/load.hpp>
#include <boost/simd/include/functions/multiplies.hpp>
#include <boost/simd/include/functions/plus.hpp>
#include <boost/simd/include/functions/sum.hpp>
#include <boost/simd/include/functions/rsqrt.hpp>
#include <boost/simd/include/functions/fast_rsqrt.hpp>
#include <boost/simd/include/functions/aligned_store.hpp>
#include <boost/simd/sdk/simd/pack.hpp>

class Osc_boost
{
public:
  using pack = boost::simd::pack<float, 4>;

  void setFreq(int i, float f);

  pack value();
  void step();

private:
  float real[4] = {1, 1, 1, 1}, imag[4] = {0, 0, 0, 0};
  float dReal[4] = {1, 1, 1, 1}, dImag[4] = {0, 0, 0, 0};
};

void Osc_boost::setFreq(int i, float f)
{
  dReal[i] = cos(f);
  dImag[i] = sin(f);
}

Osc_boost::pack Osc_boost::value()
{
  return boost::simd::aligned_load<pack>((float*)imag);
}

void Osc_boost::step()
{
  pack r0 = boost::simd::aligned_load<pack>((float*)real);
  pack dr0 = boost::simd::aligned_load<pack>((float*)dReal);
  pack i0 = boost::simd::aligned_load<pack>((float*)imag);
  pack di0 = boost::simd::aligned_load<pack>((float*)dImag);
  pack r = r0*dr0 - i0*di0;
  pack i = r0*di0 + i0*dr0;

  pack l = r*r + i*i;
  //auto f = boost::simd::fast_rsqrt(l); // Who knows what that is...
  pack f = boost::simd::rsqrt(l);

  boost::simd::aligned_store<pack>(f*r, (float*)real);
  boost::simd::aligned_store<pack>(f*i, (float*)imag);
}

////////////////////////////////////////////////////////////////////////////////

int main()
{
  // Benchmark
  std::cout << "type,t" << std::endl;
  auto n = 1u << 26;
  for (auto i = 0; i < 9; ++i)
  {
    {
      auto oscs = new Osc_scalar_q[4];
      oscs[0].setFreq(0.1);
      oscs[1].setFreq(0.2);
      oscs[2].setFreq(0.3);
      oscs[3].setFreq(0.4);
      {
        Benchmark b("Scalar (fast)");
        for (decltype(n) i = 0; i < n; ++i)
        {
          oscs[0].step();
          oscs[1].step();
          oscs[2].step();
          oscs[3].step();
        }
      }
      std::cerr << oscs[0].value() << " "
                << oscs[1].value() << " "
                << oscs[2].value() << " "
                << oscs[3].value() << std::endl;
      delete[] oscs;
    }

    {
      auto oscs = new Osc_scalar[4];
      oscs[0].setFreq(0.1);
      oscs[1].setFreq(0.2);
      oscs[2].setFreq(0.3);
      oscs[3].setFreq(0.4);
      {
        Benchmark b("Scalar");
        for (decltype(n) i = 0; i < n; ++i)
        {
          oscs[0].step();
          oscs[1].step();
          oscs[2].step();
          oscs[3].step();
        }
      }
      std::cerr << oscs[0].value() << " "
                << oscs[1].value() << " "
                << oscs[2].value() << " "
                << oscs[3].value() << std::endl;
      delete[] oscs;
    }

    {
      auto osc = new Osc_sse2();
      osc->setFreq(0, 0.1);
      osc->setFreq(1, 0.2);
      osc->setFreq(2, 0.3);
      osc->setFreq(3, 0.4);
      {
        Benchmark b("Intrinsics");
        for (decltype(n) i = 0; i < n; ++i)
        {
          osc->step();
        }
      }
      auto v = osc->value();
      std::cerr << extract<0>(v) << " "
                << extract<1>(v) << " "
                << extract<2>(v) << " "
                << extract<3>(v) << std::endl;
      delete osc;
    }

#if 1
    {
      auto osc = new Osc_boost();
      osc->setFreq(0, 0.1);
      osc->setFreq(1, 0.2);
      osc->setFreq(2, 0.3);
      osc->setFreq(3, 0.4);
      {
        Benchmark b("Boost");
        for (decltype(n) i = 0; i < n; ++i)
        {
          osc->step();
        }
      }
      auto v = osc->value();
      std::cerr << v[0] << " "
                << v[1] << " "
                << v[2] << " "
                << v[3] << std::endl;
      delete osc;
    }
#endif
  }

#if 0
  // Dump
  {
#if 0
    auto osc = new Osc_sse2();
    osc->setFreq(0, 0.1);
    osc->setFreq(1, 0.2);
    osc->setFreq(2, 0.4);
    osc->setFreq(3, 0.39);
    auto samples = aligned_new<float>(44100);
    const char* path = "osc_sse2.wav";
    for (auto i = 0; i < 44100; i += 1)
    {
      auto t0 = _mm_mul_ps(_mm_set_ps1(0.2), osc->value());
      auto t1 = _mm_shuffle_ps(t0, t0, 0x44); // = [ a, b, a, b ], 0b01000100
      auto t2 = _mm_shuffle_ps(t0, t0, 0xbb); // = [ c, d, c, d ], 0b10111011
      auto t3 = _mm_add_ps(t1, t2);       // = [ a+c, b+d, a+c, b+d ]
      auto t4 = _mm_shuffle_ps(t3, t3, 0xb1); // = [ b+d, a+c, b+d, a+c ] // 0b10110001
      auto t5 = _mm_add_ps(t3, t4);       // = [ a+c+b+d, b+d+a+c, a+c+b+d, b+d+a+c ]
      auto value = _mm_cvtss_f32(t5);
      samples[i] = value;
      osc->step();
    }
#elif 0
    auto osc = new Osc_boost();
    osc->setFreq(0, 0.1);
    osc->setFreq(1, 0.2);
    osc->setFreq(2, 0.4);
    osc->setFreq(3, 0.39);
    auto samples = aligned_new<float>(44100);
    const char* path = "osc_boost.wav";
    for (auto i = 0; i < 44100; i += 1)
    {
      auto t0 = Osc_boost::pack(0.2) * osc->value();
      samples[i] = boost::simd::sum(t0);
      osc->step();
    }
#elif 0
    auto osc = new Osc_scalar[4];
    osc[0].setFreq(0.1);
    osc[1].setFreq(0.2);
    osc[2].setFreq(0.4);
    osc[3].setFreq(0.39);
    auto samples = aligned_new<float>(44100);
    const char* path = "osc_scalar.wav";
    for (auto i = 0; i < 44100; i += 1)
    {
      auto value = osc[0].value() + osc[1].value() + osc[2].value() + osc[3].value();
      samples[i] = .2 * value;
      osc[0].step();
      osc[1].step();
      osc[2].step();
      osc[3].step();
    }
#else
    auto osc = new Osc_scalar_q[4];
    osc[0].setFreq(0.1);
    osc[1].setFreq(0.2);
    osc[2].setFreq(0.4);
    osc[3].setFreq(0.39);
    auto samples = aligned_new<float>(44100);
    const char* path = "osc_scalar_q.wav";
    for (auto i = 0; i < 44100; i += 1)
    {
      auto value = osc[0].value() + osc[1].value() + osc[2].value() + osc[3].value();
      samples[i] = .2 * value;
      osc[0].step();
      osc[1].step();
      osc[2].step();
      osc[3].step();
    }
#endif

    SF_INFO info {};
    info.samplerate = 44100;
    info.channels = 1;
    info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    auto sf = sf_open(path, SFM_WRITE, &info);
    sf_writef_float(sf, samples, 44100);
    delete[] samples;
  }
#endif
}
