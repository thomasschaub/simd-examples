require('ggplot2')

df <- read.csv('matrix.csv')

tScalar <- min(df[df$type == 'Scalar',]$t)
tSimd <- min(df[df$type == 'SPMD SOA',]$t)
tScalar / tSimd

p <- ggplot(df, aes(type, t))
ggsave(
       'matrix.pdf',
       width = 9,
       height = 5,
       p +
       geom_boxplot() +
       theme_bw() +
       expand_limits(y=0) +
       scale_x_discrete(
                        name="Implementation",
                        limits=c("Scalar", "Naive", "SPMD AOS", "SPMD SOA", "Boost.SIMD", "memcpy")
                        ))
