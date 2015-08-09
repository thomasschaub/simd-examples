#pragma once

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>

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

// Does not really align anything, but fails if not aligned to a 16 byte
// boundary.
template<typename T>
std::unique_ptr<T[]> aligned_new(unsigned long n)
{
  auto p = new T[n];
  auto ui = reinterpret_cast<uintptr_t>(p);
  assert((ui & 0xF) == 0);
  return std::unique_ptr<T[]>(p);
}

template <int i>
float extract(__m128 v)
{
  auto t = _mm_shuffle_ps(v, v, i);
  return _mm_cvtss_f32(t);
};
