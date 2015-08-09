#pragma once

#include <cassert>
#include <chrono>
#include <iostream>

struct Benchmark
{
  Benchmark(const char* label)
    : label(label)
  {
  }


  ~Benchmark()
  {
    auto t1 = std::chrono::system_clock::now();
    std::cout << label << "," << (t1 - t0).count() << std::endl;
  }

private:
  const char* label;
  std::chrono::system_clock::time_point t0 = std::chrono::system_clock::now();
};

template<typename T>
T* aligned_new(unsigned long n)
{
  auto p = new T[n];
  auto ui = reinterpret_cast<uintptr_t>(p);
  assert((ui & 0xF) == 0);
  return p;
}

template <int i>
float extract(__m128 v)
{
  auto t = _mm_shuffle_ps(v, v, i);
  return _mm_cvtss_f32(t);
};
