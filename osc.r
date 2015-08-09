require('ggplot2')

df <- read.csv('osc.csv')

tScalar <- min(df[df$type == 'Scalar',]$t)
tScalar
tScalarFast <- min(df[df$type == 'Scalar (fast)',]$t)
tScalarFast
tSimd <- min(df[df$type == 'Intrinsics',]$t)
tSimd
tScalar / tSimd
tScalarFast / tSimd

p <- ggplot(df, aes(type, t))
ggsave(
       'osc.pdf',
       width = 9,
       height = 5,
       p +
       geom_boxplot() +
       theme_bw() +
       expand_limits(y=0) +
       scale_x_discrete(
                        name="Implementation",
                        limits=c("Scalar", "Scalar (fast)", "Intrinsics", "Boost")
                        ))
