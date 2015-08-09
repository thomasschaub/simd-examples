#include <iostream>

#include <xmmintrin.h>

#include <boost/simd/arithmetic/arithmetic.hpp>
#include <boost/simd/sdk/simd/pack.hpp>

#include <sndfile.h>

#include "util.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image_write.h"

////////////////////////////////////////////////////////////////////////////////

void waveform_scalar(float* xs, unsigned n, unsigned char* ys, unsigned w, unsigned h)
{
  // A sample here is a sample in the sense of super sampling for anti aliasing
  // the output image. "Audio samples" are called frames.
  const unsigned samples = 16;

  // The input is processed in chunks, where one chunk corresponds to one pixel
  // in the output image. We potentially ignore some frames if the division has
  // a remainder, but no one will notice.
  unsigned k = n / w; // Number of frames per chunk.

  // Compute the waveform's output columns one by one.
  for (unsigned i = 0; i < w; ++i)
  {
    // For each output column, determine the min and max value of the
    // corresponding chunk of audio frames. Again, division might ignore frames,
    // no one cares.
    float mins[samples];
    float maxs[samples];
    for (unsigned s = 0; s < samples; ++s)
    {
      auto min = std::numeric_limits<float>::max();
      auto max = std::numeric_limits<float>::min();
      for (unsigned j = s * k / samples, end = (s+1) * k / samples; j < end; ++j)
      {
        float x = xs[k*i + j];
        min = std::min(min, x);
        max = std::max(max, x);
      }
      mins[s] = min;
      maxs[s] = max;
    }

    // Draw the output column by iterating its rows (levels). The ouput color is
    // determined by the fraction of samples surpassing the current level.
    unsigned hMin[samples];
    unsigned hMax[samples];
    for (unsigned j = 0; j < samples; ++j)
    {
      hMin[j] = h*(mins[j] + 1) / 2;
      hMax[j] = h*(maxs[j] + 1) / 2;
    }
    for (unsigned j = 0; j < h; ++j)
    {
      unsigned char y = 255;
      for (unsigned s = 0; s < samples; ++s)
      {
        if (j < hMin[s] || hMax[s] < j)
        {
          y -= 255 / samples;
        }
      }
      ys[w*j + i] = y;
    }
  }
}

void waveform_sse2(float* xs, unsigned n, unsigned char* ys, unsigned w, unsigned h)
{
  const unsigned samples = 16;
  unsigned k = n / w;
  for (unsigned i = 0, end = w & ~4; i < end; i += 4)
  {
    __m128 mins[samples];
    __m128 maxs[samples];
    for (unsigned s = 0; s < samples; ++s)
    {
      auto min = _mm_set_ps1(std::numeric_limits<float>::max());
      auto max = _mm_set_ps1(std::numeric_limits<float>::min());
      for (unsigned j = s * k / samples, end = (s+1) * k / samples; j < end; ++j)
      {
        auto x = _mm_set_ps(
          xs[k*(i+3) + j],
          xs[k*(i+2) + j],
          xs[k*(i+1) + j],
          xs[k*i     + j]);
        min = _mm_min_ps(min, x);
        max = _mm_max_ps(max, x);
      }
      mins[s] = min;
      maxs[s] = max;
    }

    __m128i hMins[samples];
    __m128i hMaxs[samples];
    for (unsigned j = 0; j < samples; ++j)
    {
      hMins[j] = _mm_cvttps_epi32(_mm_mul_ps(_mm_set_ps1(0.5*h), _mm_add_ps(mins[j], _mm_set_ps1(1))));
      hMaxs[j] = _mm_cvttps_epi32(_mm_mul_ps(_mm_set_ps1(0.5*h), _mm_add_ps(maxs[j], _mm_set_ps1(1))));
    }
    for (unsigned j = 0; j < h; ++j)
    {
      auto y = _mm_set_epi32(255, 255, 255, 255);
      for (unsigned s = 0; s < samples; ++s)
      {
        auto jV = _mm_set_epi32(j, j, j, j);
        auto cond0 = _mm_cmpgt_epi32(jV, hMaxs[s]);
        auto cond1 = _mm_cmplt_epi32(jV, hMins[s]);
        auto cond2 = _mm_or_si128(cond0, cond1);
        auto dY = 255 / samples;
        y = _mm_sub_epi32(y, _mm_and_si128(_mm_set_epi32(dY, dY, dY, dY), cond2));
      }
      ys[w*j + i+0] = _mm_cvtsi128_si32(_mm_shuffle_epi32(y, 0));
      ys[w*j + i+1] = _mm_cvtsi128_si32(_mm_shuffle_epi32(y, 1));
      ys[w*j + i+2] = _mm_cvtsi128_si32(_mm_shuffle_epi32(y, 2));
      ys[w*j + i+3] = _mm_cvtsi128_si32(_mm_shuffle_epi32(y, 3));
    }
  }
}

// `x = pack_from_f<4>()(f)` is equivalent to (but faster than)
// `for (unsigned l = 0; l < 4; ++l) x[l] = f(l);`
template <unsigned vecWidth>
struct pack_from_f
{
  template <typename Fn>
  boost::simd::pack<float, vecWidth> operator()(const Fn& fn);
};

template <>
struct pack_from_f<4>
{
  template <typename Fn>
  boost::simd::pack<float, 4> operator()(const Fn& fn)
  {
    return boost::simd::pack<float, 4>(fn(0), fn(1), fn(2), fn(3));
  }
};

template <unsigned vecWidth>
void waveform_boost(float* xs, unsigned n, unsigned char* ys, unsigned w, unsigned h)
{
  using packF = boost::simd::pack<float, vecWidth>;
  using packI = boost::simd::pack<int, vecWidth>;

  const unsigned samples = 16;
  unsigned k = n / w;
  for (unsigned i = 0, end = w & ~vecWidth; i < end; i += vecWidth)
  {
    packF mins[samples];
    packF maxs[samples];
    for (unsigned s = 0; s < samples; ++s)
    {
      packF min = packF(std::numeric_limits<float>::max());
      packF max = packF(std::numeric_limits<float>::min());
      for (unsigned j = s * k / samples, end = (s+1) * k / samples; j < end; ++j)
      {
        packF x = pack_from_f<4>()([=] (unsigned l) { return xs[k*(i+l) + j]; });
        min = boost::simd::min(min, x);
        max = boost::simd::max(max, x);
      }
      mins[s] = min;
      maxs[s] = max;
    }

    packI hMin[samples];
    packI hMax[samples];
    for (unsigned j = 0; j < samples; ++j)
    {
      hMin[j] = boost::simd::toints(packF(h)*(mins[j] + packF(1)) / packF(2));
      hMax[j] = boost::simd::toints(packF(h)*(maxs[j] + 1.0f) / 2.0f);
    }
    for (unsigned j = 0; j < h; ++j)
    {
      packI y = packI(255);
      for (unsigned s = 0; s < samples; ++s)
      {
        packI jV = packI(j);
        y = y - boost::simd::if_else_zero(
          jV < hMin[s] || hMax[s] < jV,
          packI(255 / samples));
      }

      for (unsigned l = 0; l < vecWidth; ++l)
      {
        ys[w*j + i + l] = y[l];
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
  // Parse arguments
  if (argc != 2)
  {
    std::cerr << "Usage: " << argv[0] << " <in.wav>" << std::endl;
    return 1;
  }
  auto path = argv[1];

  SF_INFO info;
  auto sf = sf_open(path, SFM_READ, &info);
  if (sf == nullptr)
  {
    abort();
  }

  unsigned n = info.frames * info.channels;
  unsigned w = 400; // Must be a multiple of 4.
  unsigned h = 100;
  float* xs = new float[n];
  {
    auto frames = sf_readf_float(sf, xs, info.frames);
    assert(frames == info.frames);
  }

  std::cout << "type,t" << std::endl;

  for (int i = 0; i < 9; ++i)
  {
    {
      auto ys_scalar = aligned_new<unsigned char>(w*h);
      {
        Benchmark b("Scalar");
        waveform_scalar(xs, n, ys_scalar.get(), w, h);
      }
      stbi_write_png("waveform_scalar.png", w, h, 1, ys_scalar.get(), w);
    }

    {
      auto ys_sse2 = aligned_new<unsigned char>(w*h);
      {
        Benchmark b("Intrinsics");
        waveform_sse2(xs, n, ys_sse2.get(), w, h);
      }
      stbi_write_png("waveform_sse2.png", w, h, 1, ys_sse2.get(), w);
    }

    {
      auto ys_boost = aligned_new<unsigned char>(w*h);
      {
        Benchmark b("Boost.SIMD");
        waveform_boost<4>(xs, n, ys_boost.get(), w, h);
      }
      stbi_write_png("waveform_boost.png", w, h, 1, ys_boost.get(), w);
    }
  }

}
