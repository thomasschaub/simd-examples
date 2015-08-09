require('ggplot2')

df <- read.csv('waveform.csv')

tScalar <- min(df[df$type == 'Scalar',]$t)
tSimd <- min(df[df$type == 'Intrinsics',]$t)
tScalar / tSimd

p <- ggplot(df, aes(type, t))
ggsave(
       'waveform.pdf',
       width = 6,
       height = 5,
       p +
       geom_boxplot() +
       theme_bw() +
       expand_limits(y=0) +
       scale_x_discrete(
                        name="Implementation",
                        limits=c("Scalar", "Intrinsics", "Boost.SIMD")
                        ))
