#include <gtest/gtest.h>

#define NOMAIN
#include "convolution.cpp"

// f*g[n] = sum(f[m]*g[n-m])
//        = sum(f[n-m]*g[m])

// f[1]*g[-1] + f[0]*g[0] + f[1]*g[-1]

using ConvolveFn = void (*)(const float *, int, const float*, int, float*);

class ConvolutionTest: public ::testing::TestWithParam<ConvolveFn>
{
};

TEST_P(ConvolutionTest, Simple)
{
  unsigned n = 8;
  auto xs = aligned_new<float>(2*n);
  for (auto i = 0; i < 2*n; ++i)
  {
    xs[i] = i;
  }
  unsigned m = 4;
  auto kernel = aligned_new<float>(2*m);
  for (auto i = 0; i < 2*m; ++i)
  {
    kernel[i] = 2*i;
  }
  auto ys = aligned_new<float>(2*n);

  GetParam()(xs.get(), n, kernel.get(), m, ys.get());

  EXPECT_FLOAT_EQ( 0*0, ys[ 0]);
  EXPECT_FLOAT_EQ( 1*2, ys[ 1]);
  EXPECT_FLOAT_EQ( 2*0+ 0*4, ys[ 2]);
  EXPECT_FLOAT_EQ( 3*2+ 1*6, ys[ 3]);
  EXPECT_FLOAT_EQ( 4*0+ 2*4+ 0* 8, ys[ 4]);
  EXPECT_FLOAT_EQ( 5*2+ 3*6+ 1*10, ys[ 5]);
  EXPECT_FLOAT_EQ( 6*0+ 4*4+ 2* 8+0*12, ys[ 6]);
  EXPECT_FLOAT_EQ( 7*2+ 5*6+ 3*10+1*14, ys[ 7]);
  EXPECT_FLOAT_EQ( 8*0+ 6*4+ 4* 8+2*12, ys[ 8]);
  EXPECT_FLOAT_EQ( 9*2+ 7*6+ 5*10+3*14, ys[ 9]);
  EXPECT_FLOAT_EQ(10*0+ 8*4+ 6* 8+4*12, ys[10]);
  EXPECT_FLOAT_EQ(11*2+ 9*6+ 7*10+5*14, ys[11]);
  EXPECT_FLOAT_EQ(12*0+10*4+ 8* 8+6*12, ys[12]);
  EXPECT_FLOAT_EQ(13*2+11*6+ 9*10+7*14, ys[13]);
  EXPECT_FLOAT_EQ(14*0+12*4+10* 8+8*12, ys[14]);
  EXPECT_FLOAT_EQ(15*2+13*6+11*10+9*14, ys[15]);
}

INSTANTIATE_TEST_CASE_P(MyConvolutionTest, ConvolutionTest, ::testing::Values(&convolve_scalar, &convolve_scalar_alt, &convolve_sse2));
