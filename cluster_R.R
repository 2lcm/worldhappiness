#library(readr)
#library(mclust)
#library(rgl)

data <- read_csv("dataset/all_data.csv")
dt <- data[,-11]
dt_group <- data[,11]
pca_dt <- prcomp(dt, center=T, scale. = T)
#print(pca_dt)

#plot(pca_dt, type = "l")
#summary(pca_dt)


data_pca <- predict(pca_dt, newdata = dt)
X = dt
BIC = mclustBIC(X)
#plot(BIC)
mod1 = Mclust(X, x = BIC)
plot3d(data_pca, col = mod1$classification)
