require('ggplot2')

df <- read.csv('waveform.csv')
p <- ggplot(df, aes(type, t))
ggsave(
       'waveform.svg',
       p +
       geom_boxplot() +
       theme_bw() +
       expand_limits(y=0) +
       scale_x_discrete(
                        name="Implementation",
                        limits=c("Scalar", "Intrinsics", "Boost.SIMD")
                        ))
