#-----------------------------------------------------#
#             GRINDING WHEEL PREDICT                  #
#                                                     #
#                     FUNCTIONs                       #
#                                                     #
#              VERSION 1.0 - 2019-24-09               #
#-----------------------------------------------------#

stand <- function(s)
{
  value <- (s - min(s))/(max(s)-min(s))
  return(value)
}

standm <- function(s,mn,mx)
{
  value <- (s - mn)/(mx-mn)
  return(value)
}

destand <- function(x,mn,mx)
{
  value <- (x*(mx-mn))+mn
  return(value)
}

calcrmseMult <- function(y,yhat)
{
  error <- rep(NA, dim(y)[1])
  
  for (i in 1:dim(y)[1])
  {
    error[i] <- sum(((y[i,] - yhat[i,])^2))
  }
  
  mse <- sum(error)/dim(y)[1]
  rmse <- sqrt(mse)
  
  return(rmse)
}

calcAllErrors <- function(y, yhat, rw, cl)
{
  
  errors <- rep(NA, 4)
  errors[1] <- RMSE(y,yhat)
  errors[2] <- MAPE(y,yhat)
  errors[3] <- MedianAPE(y,yhat)
  errors[4] <- R2_Score(y,yhat)
  
  print(paste("RMSE: ",error_table[rw[1],cl] <- round(errors[1],4)))
  print(paste("MAPE: ",error_table[rw[2],cl] <- round(errors[2],4)))
  print(paste("MedianAPE: ",error_table[rw[3],cl] <- round(errors[3],4)))
  print(paste("R2 Score: ",error_table[rw[4],cl] <- round(errors[4],4)))
  
  return (error_table)
}

testSigCorr <- function(mat, ...) 
{
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat<- matrix(NA, n, n)
  diag(p.mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j], ...)
      p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
    }
  }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}