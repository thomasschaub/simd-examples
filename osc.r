require('ggplot2')

df <- read.csv('osc.csv')
p <- ggplot(df, aes(type, t))
ggsave(
       'osc.svg',
       p +
       geom_boxplot() +
       theme_bw() +
       expand_limits(y=0) +
       scale_x_discrete(
                        name="Implementation",
                        limits=c("Scalar", "Intrinsics")
                        ))
