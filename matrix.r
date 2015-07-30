require('ggplot2')

df <- read.csv('matrix.csv')
p <- ggplot(df, aes(type, t))
ggsave(
       'matrix.svg',
       p +
       geom_boxplot() +
       theme_bw() +
       expand_limits(y=0) +
       scale_x_discrete(
                        name="Implementation",
                        limits=c("Scalar", "Naive", "SPMD AOS", "SPMD SOA", "Boost.SIMD")
                        ))
