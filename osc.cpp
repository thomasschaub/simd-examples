#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <thread>

#include <xmmintrin.h>

#include <sndfile.h>

#include "util.h"

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

  real = r;
  imag = i;
}

class Osc_sse2
{
public:
  void setFreq(float f);

  float value();
  __m128 valueV();
  void step();
  void stepV();

private:
  float real = 1, imag = 0;
  float dReal = 1, dImag = 0;
  float dRealV = 1, dImagV = 0;
  __m128 fixRealV = _mm_set_ps1(0), fixImagV = _mm_set_ps1(0);
};

void Osc_sse2::setFreq(float f)
{
  dReal = cos(f);
  dImag = sin(f);

  float rs[5] = { 1, dReal };
  float is[5] = { 0, dImag };
  for (auto i = 2; i < 5; ++i)
  {
    rs[i] = rs[i-1]*dReal - is[i-1]*dImag;
    is[i] = rs[i-1]*dImag + is[i-1]*dReal;
  }

  fixRealV = _mm_set_ps(rs[3], rs[2], rs[1], rs[0]);
  fixImagV = _mm_set_ps(is[3], is[2], is[1], is[0]);

  dRealV = rs[4];
  dImagV = is[4];
}

float Osc_sse2::value()
{
  return imag;
}

__m128 Osc_sse2::valueV()
{
  return _mm_add_ps(
    _mm_mul_ps(_mm_set_ps1(real), fixImagV),
    _mm_mul_ps(_mm_set_ps1(imag), fixRealV));
}

void Osc_sse2::step()
{
  float r = real*dReal - imag*dImag;
  float i = real*dImag + imag*dReal;

  real = r;
  imag = i;
}

void Osc_sse2::stepV()
{
  float r = real*dRealV - imag*dImagV;
  float i = real*dImagV + imag*dRealV;

  real = r;
  imag = i;
}

////////////////////////////////////////////////////////////////////////////////

int main()
{
  auto n = 1u << 26;
  // Benchmark
  for (auto i = 0; i < 9; ++i)
  {
    {
      Osc_scalar osc;
      osc.setFreq(0.1);
      {
        Benchmark b("Scalar");
        for (decltype(n) i = 0; i < n; ++i)
        {
          osc.step();
        }
      }
      std::cerr << osc.value() << std::endl;
    }

    {
      Osc_sse2 osc;
      osc.setFreq(0.1);
      {
        Benchmark b("Intrinsics");
        for (decltype(n) i = 0; i < n; i += 4)
        {
          osc.stepV();
        }
        std::cerr << osc.value() << std::endl;
      }
    }
  }

#if 0
  // Dump
  {
    Osc_sse2 osc;
    osc.setFreq(0.1);
    auto samples = aligned_new<float>(44100);
#if 1
    const char* path = "osc_sse2.wav";
    for (auto i = 0; i < 44100; i += 4)
    {
      auto value = _mm_mul_ps(_mm_set_ps1(0.9), osc.valueV());
      _mm_store_ps(samples + i, value);
      osc.stepV();
    }
#else
    const char* path = "osc_scalar.wav";
    for (auto i = 0; i < 44100; ++i)
    {
      samples[i] = 0.9 * osc.value();
      osc.step();
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
