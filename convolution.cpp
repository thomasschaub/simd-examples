#include <algorithm>

#include <sndfile.h>

#include <xmmintrin.h>

#include "util.h"

// We assume 2 channels everywhere, so every "inexplicable" factor of 2 (or 1/2)
// is most likely because of that

//        0  1  2  3  4  5
// f   = [a, b, c, d, e, f]
// g   = [g, h]
// f*g = [ag, bg+ah, cg+bh, dg+ch, eg+dh, fg+eh, fh]
// (f*g)[n] = sum(f[n-m]*g[m]) = f[n]*g[0] + f[n-1]*g[1]
void convolve_scalar(const float *xs, int n, const float* kernel, int m, float* ys)
{
  for (decltype(n) i = 0; i < n; ++i)
  {
    float a[] = {0, 0};
    auto start = std::max<int>(0, i - m + 1);
    for (decltype(n) j = start; j <= i; ++j)
    {
      a[0] += xs[2*j+0]*kernel[2*(i-j)+0];
      a[1] += xs[2*j+1]*kernel[2*(i-j)+1];
    }
    ys[2*i+0] = a[0];
    ys[2*i+1] = a[1];
  }
}

// Closer to SIMD, but still scalar.
void convolve_scalar_alt(const float *xs, int n, const float* kernel, int m, float* ys)
{
  // C.f. SIMD version
  auto split = m + 1;
  convolve_scalar(xs, split, kernel, m, ys);

  for (decltype(n) i = split, end = n & ~1; i < end; i += 2)
  {
    float a[] = {0, 0, 0, 0};
    for (decltype(n) j = i - m + 1; j <= i; j += 1)
    {
      auto kL = kernel[2*(i-j)+0];
      auto kR = kernel[2*(i-j)+1];
      a[0] += xs[2*j+0]*kL;
      a[1] += xs[2*j+1]*kR;
      a[2] += xs[2*j+2]*kL;
      a[3] += xs[2*j+3]*kR;
    }
    ys[2*i+0] = a[0];
    ys[2*i+1] = a[1];
    ys[2*i+2] = a[2];
    ys[2*i+3] = a[3];
  }
}

void convolve_sse2(const float *xs, int n, const float* kernel, int m, float* ys)
{
  // Peel off some iterations so we have the same number of iterations for all
  // instances in the inner loop. We need to make sure that the addresses we
  // load from below are sill  properly aligned. We will load from
  // xs + 2*(i - m + 1). so 2*(i - m + 1) must be a multiple of 4. We will chose
  // i = m + 1 => 2*(m + 1 - m + 1) = 4.
  auto split = m + 1;
  convolve_scalar(xs, split, kernel, m, ys);

  // Increase by steps of 2 because we have two channels.
  for (decltype(n) i = split, end = n & ~3; i < end; i += 2)
  {
    auto a = _mm_set_ps1(0);
    for (decltype(n) j = i - m + 1; j <= i; j += 1)
    {
      auto x = _mm_loadu_ps(xs + 2*j);
      auto kL = kernel[2*(i - j)];
      auto kR = kernel[2*(i - j) + 1];
      auto k = _mm_set_ps(kR, kL, kR, kL);
      a = _mm_add_ps(a, _mm_mul_ps(x, k));
    }
    _mm_storeu_ps(ys + 2*i, a);
  }
}

////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<float[]> boxKernel(unsigned n)
{
  auto xs = aligned_new<float>(2*n);
  auto x = 1.0f / n;
  for (unsigned i = 0; i < 2*n; ++i)
  {
    xs[i] = x;
  }
  return xs;
}

#ifndef NOMAIN
std::unique_ptr<float[]> readAudio(const char* path, unsigned& n)
{
  SF_INFO info;
  auto sf = sf_open(path, SFM_READ, &info);
  if (sf == nullptr)
  {
    abort();
  }
  if (info.samplerate != 44100 || info.channels != 2)
  {
    std::cout << "Warning: Unexpected samplerate/#channels" << std::endl;
  }
  n = info.frames * info.channels;
  auto xs = aligned_new<float>(n);
  {
    auto frames = sf_readf_float(sf, xs.get(), info.frames);
    assert(frames == info.frames);
  }
  sf_close(sf);
  return xs;
}

void writeAudio(const char* path, float* xs, unsigned n)
{
  SF_INFO info {};
  info.samplerate = 44100;
  info.channels = 2;
  info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
  auto sf = sf_open(path, SFM_WRITE, &info);
  auto frames = sf_writef_float(sf, xs, n / 2);
  assert(frames == n/2);
  sf_close(sf);
}

int main(int argc, char* argv[])
{
  // Parse arguments
  if (argc != 3)
  {
    std::cerr << "Usage: " << argv[0] << " <in.wav> <kernel witdth>" << std::endl;
    return 1;
  }

  // Read input
  auto inPath = argv[1];
  unsigned n;
  auto xs = readAudio(inPath, n);
  unsigned nChannels = 2;
  auto m = 2*atoi(argv[2]);
  auto kernel = boxKernel(m);

  std::cout << "type,t" << std::endl;

  for (auto i = 0; i < 9; ++i)
  {
    {
      auto ys = aligned_new<float>(n);
      {
        Benchmark b("Scalar");
        convolve_scalar(xs.get(), n/nChannels, kernel.get(), m/nChannels, ys.get());
      }
      writeAudio("convolution_scalar.wav", ys.get(), n);
    }

    {
      auto ys = aligned_new<float>(n);
      {
        Benchmark b("Scalar (alternative)");
        convolve_scalar_alt(xs.get(), n/nChannels, kernel.get(), m/nChannels, ys.get());
      }
      writeAudio("convolution_scalar_alt.wav", ys.get(), n);
    }

    {
      auto ys = aligned_new<float>(n);
      {
        Benchmark b("Intrinsics");
        convolve_sse2(xs.get(), n/nChannels, kernel.get(), m/nChannels, ys.get());
      }
      writeAudio("convolution_sse2.wav", ys.get(), n);
    }
  }
}
#endif
