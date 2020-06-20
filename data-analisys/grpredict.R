#-----------------------------------------------------#
#             GRINDING WHEEL PREDICT                  #
#                                                     #
#                 DATA ANALISYS                       #
#                                                     #
#              VERSION 1 - 2019-16-10                 #
#-----------------------------------------------------#
 
#init workspace
rm(list=ls())
cat('\014')
set.seed(13)
options(warn=-1)

#LOAD FUNCTIONS-------------------------------------------------
source("functions-grpredict.R")

#LOAD LIBRARIES-------------------------------------------------
library(RSNNS)
library(rgl)
library(corrplot)
library(corrgram)
library(agricolae)
library(MLmetrics)

#LOAD DATASET---------------------------------------------------
filename <- "dataset.csv"
#carrega a base de dados
dt <- read.table(filename,
                    header=TRUE,
                    sep=";",
                    colClasses=rep("numeric",9))
colnames(dt) <- c("gr","mta","mtb","speed","rate","vol","wear","rough","cost")

#PREPARE DATASET------------------------------------------------
#minimum and maximum limits for speed, rate and volume
speedinf <- 45
speedsup <- 80

rateinf <- 50
ratesup <- 150

volinf <- 5000
volsup <- 12000

#configure dolar/real
coin <- 3.79

#labs of axis
wearlab <- expression(paste("Radial wear (",Delta,"rs) [",mu,"m]",sep=""))
roughlab <- expression(paste("Roughness (Ra) [",mu,"m]",sep=""))
costlab <- "Total cost/part (Ct) [US$]"
speedlab <- "Wheel speed (vs) [m/s]"
ratelab <- "Specifc material removal rate (Q'w) [mm³/mm.s]"
volumelab <- "Metal removed (Vw) [10³ mm³]"

wg <- 900
hg <- 650
rg <- 90
wp <- 700
hp <- 350
rp <- 70 

#standardize dataset by column
dtstand <- cbind(dt[,1:3],                                #gr and material
            round(standm(dt[,4],speedinf,speedsup),4),    #speed
            round(standm(dt[,5],rateinf,ratesup),4),      #rate
            round(standm(dt[,6],volinf,volsup),4),        #volume
            round(stand(dt[,7]),4),                       #wear
            round(stand(dt[,8]),4),                       #rough
            round(stand(dt[,9]),4))                       #cost

colnames(dtstand) <- c("gr","mta","mtb","speed","rate","vol","wear","rough","cost")

#DIVIDE DATASET-------------------------------------------------
lentrain <- 0.6

bound <- floor(length(dtstand$gr)*lentrain)
indextrain <- sample(length(dtstand$gr),bound,FALSE)
dttrain <- dtstand[indextrain,]
dttest <- dtstand[-indextrain,]

#TRAIN MODEL----------------------------------------------------
newmodel <- FALSE

#1:gr
#2:mta
#3:mtb
#4:speed
#5:rate
#6:vol
#7:wear
#8:rough
#9:cost

#configure inputs
inputs <- c(1,2,3,4,5,6)

#configure outputs
outputs <- c(7,8,9)

#configure model
nneuros = c(14, 14, 10)
maxep <- 40000
lrrate <- 0.079
arch <- "std"  #lm, res, mom, std

#generate model
modelmlp <- NULL

if (newmodel==FALSE)
{
  #load model
  load('grmodel.Rdata')
} else {
  #train model
  switch(arch,
         "lm" = {  #Levenberg-Marquardt
           metodo <- "Levenberg-Marquardt"
           modelmlp <- mlp(dttrain[,inputs], dttrain[,outputs], size=nneuros, maxit=maxep, initFunc="Randomize_Weights",
                       initFuncParams=c(-0.3, 0.3), learnFunc="SCG",
                       learnFuncParams=c(10e-4,10e-6,0,10e-16), updateFunc="Topological_Order",
                       updateFuncParams=c(0), hiddenActFunc="Act_Logistic",
                       shufflePatterns=F, linOut=TRUE)
         },
         "res" = {  #Backpropagation resiliente
           metodo <- "Resiliente"
           modelmlp <- mlp(dttrain[,inputs], dttrain[,outputs], size=nneuros, maxit=maxep, initFunc="Randomize_Weights",
                       initFuncParams=c(-0.3, 0.3), learnFunc="Rprop",
                       learnFuncParams=c(0.1,10,4), updateFunc="Topological_Order",
                       updateFuncParams=c(0), hiddenActFunc="Act_Logistic",
                       shufflePatterns=F, linOut=TRUE)
         },
         "mom" = {  #Backpropagation Momentum
           metodo <- "Termo momentum"
           modelmlp <- mlp(dttrain[,inputs], dttrain[,outputs], size=nneuros, maxit=maxep, initFunc="Randomize_Weights",
                       initFuncParams=c(-0.3, 0.3), learnFunc="BackpropMomentum",
                       learnFuncParams=c(0.07,0.3,0.1,0.03), updateFunc="Topological_Order",
                       updateFuncParams=c(0), hiddenActFunc="Act_Logistic",
                       shufflePatterns=F, linOut=TRUE)
         },
         "std" = {  #Backpropagation PadrÃo
           metodo <- "Padrão"
           modelmlp <- mlp(dttrain[,inputs], dttrain[,outputs], size=nneuros, maxit=maxep, initFunc="Randomize_Weights",
                       initFuncParams=c(-0.3, 0.3), learnFunc="Std_Backpropagation",
                       learnFuncParams=c(lrrate), updateFunc="Topological_Order",
                       updateFuncParams=c(0), hiddenActFunc="Act_Logistic",
                       shufflePatterns=F, linOut=TRUE)
         })
  
  #save model
  save(modelmlp, file='grmodel.Rdata')
  save(modelmlp, file='../spec-optimization/grmodel.Rdata')
}

stop()

#show convergence of model
png(filename=file.path(paste("plots/1_mlp_convergence_error.png",sep="")), width=wg, height=hg, res=rg)
plot(modelmlp$IterativeFitError,type="l",main="Error convergence of ANN MLP", xlab="Epoch", ylab="Mean Square Error", panel.first=grid())
graphics.off()

png(filename=file.path(paste("plots/small/1_mlp_convergence_error.png",sep="")), width=wp, height=hp, res=rp)
plot(modelmlp$IterativeFitError,type="l",main="Error convergence of ANN MLP", xlab="Epoch", ylab="Mean Square Error", panel.first=grid())
graphics.off()

#EVALUATE GENERAL ERROR OF MODEL----------------------------------------------
# Save table of errors
error_table <- matrix(NA, nrow = 16, ncol = 5)
header_error_table <- c("metric","general error", "gr1 without outlier", "gr1 with outlier","gr2 without outlier", "gr2 with outlier")
metrics_error_table <- c("total rmse","real scale wear error","real scale rough error","real scale cost error","wear rmse","wear mape","wear medianape","wear r2 score","rough rmse","rough mape","rough medianape","rough r2 score","cost rmse","cost mape","cost medianape","cost r2 score")
error_table <- cbind(metrics_error_table, error_table)
colnames(error_table) <- header_error_table

print("GENERAL ERROR")
realdata <- dttest[,outputs]
predictdata <- as.data.frame(round(predict(modelmlp, dttest[,inputs]),4))
colnames(predictdata) <- c("wear","roughness","cost")

#calculate errors
rwearerror <- RMSE(realdata$wear, predictdata$wear)
wearerror <- round(destand(rwearerror, min(dt$wear), max(dt$wear)),4) - min(dt$wear)

rrougherror <- RMSE(realdata$rough, predictdata$rough)
rougherror <- round(destand(rrougherror, min(dt$rough), max(dt$rough)),4) - min(dt$rough)

rcosterror <- RMSE(realdata$cost, predictdata$cost)
costerror <- round(destand(rcosterror, min(dt$cost), max(dt$cost)),4) - min(dt$cost)

totalerror <- round(sum(rwearerror,rrougherror,rcosterror)/3,4)

#show errors
print("REAL SCALE----:")
print(paste("total rmse:",error_table[1,2] <- totalerror))
print(paste("wear rmse converted to real scale:",error_table[2,2] <- wearerror))
print(paste("rough rmse converted to real scale:",error_table[3,2] <- rougherror))
print(paste("cost rmse converted to real scale:",error_table[4,2] <- costerror))

print("WEAR----:")
error_table <- calcAllErrors(realdata$wear, predictdata$wear, c(5,6,7,8), 2)
print("ROUGH----:")
error_table <- calcAllErrors(realdata$rough, predictdata$rough, c(9,10,11,12), 2)
print("COST----:")
error_table <- calcAllErrors(realdata$cost, predictdata$cost, c(13,14,15,16), 2)

print("------------------------------")
#CORRELATION REAL VS. PREDICT

#calculate the correlation matrix
correaldata <- cor(realdata)
print("Correlation of real values")
print(correaldata)

#calculate the correlation matrix
corpredictdata <- cor(predictdata)
print("Correlation of predicted values")
print(corpredictdata)

#corrplots
# png(filename=file.path(paste("plots/2_correlogram_real.png",sep="")), width=wg, height=hg, res=rg)
# corrplot(correaldata, type="lower", order="hclust", title="Correlogram of real values",mar = c(0, 0, 3, 0))
# graphics.off()
# 
# png(filename=file.path(paste("plots/3_correlogram_predicted.png",sep="")), width=wg, height=hg, res=rg)
# corrplot(corpredictdata, type="lower", order="hclust", title="Correlogram of predicted values",mar = c(0, 0, 3, 0))
# graphics.off()

print("------------------------------")
#STATISTICS METRICS
print("STATISTICS METRICS")
# Save table of statistics
stat_table <- matrix(NA, nrow = 3, ncol = 6)
header_stat_table <- c("metric","real wear","predict wear","real rough","predict rough","real cost", "predict cost")
metrics_stat_table <- c("mean","median","sd")
stat_table <- cbind(metrics_stat_table, stat_table)
colnames(stat_table) <- header_stat_table

bp <- cbind(realdata$wear,predictdata$wear,realdata$rough,predictdata$rough,realdata$cost,predictdata$cost)
colnames(bp) <- c("Real wear","Predicted wear","Real Ra", "Predicted Ra", "Real cost", "Predicted cost")

png(filename=file.path(paste("plots/4_boxplot_real_vs_predicted.png",sep="")), width =wg, height=hg, res=80)
boxplot(bp, main="Normalized [0-1] comparison between real and predicted values", col=(c("lightgray","white")))
graphics.off()

png(filename=file.path(paste("plots/small/4_boxplot_real_vs_predicted.png",sep="")), width=wp, height=450, res=70)
boxplot(bp, col=(c("lightgray","white")), cex.axis=1.3)
graphics.off()

#means
print(paste("Mean of real wear:", stat_table[1,2] <- round(mean(realdata$wear),3)))
print(paste("Mean of predict wear:", stat_table[1,3] <- round(mean(predictdata$wear),3)))
print(paste("Mean of real rough:", stat_table[1,4] <- round(mean(realdata$rough),3)))
print(paste("Mean of predict rough:", stat_table[1,5] <- round(mean(predictdata$rough),3)))
print(paste("Mean of real cost:", stat_table[1,6] <- round(mean(realdata$cost),3)))
print(paste("Mean of predict cost:", stat_table[1,7] <- round(mean(predictdata$cost),3)))
stat_table[2,3] <- 
#medians
print(paste("Median of real wear:", stat_table[2,2] <-round(median(realdata$wear),3)))
print(paste("Median of predict wear:", stat_table[2,3] <-round(median(predictdata$wear),3)))
print(paste("Median of real rough:", stat_table[2,4] <-round(median(realdata$rough),3)))
print(paste("Median of predict rough:", stat_table[2,5] <-round(median(predictdata$rough),3)))
print(paste("Median of real cost:", stat_table[2,6] <-round(median(realdata$cost),3)))
print(paste("Median of predict cost:", stat_table[2,7] <-round(median(predictdata$cost),3)))

#standard deviation
print(paste("standard deviation of real wear:", stat_table[3,2] <-round(sd(realdata$wear),3)))
print(paste("standard deviation of predict wear:", stat_table[3,3] <-round(sd(predictdata$wear),3)))
print(paste("standard deviation of real rough:", stat_table[3,4] <-round(sd(realdata$rough),3)))
print(paste("standard deviation of predict rough:", stat_table[3,5] <-round(sd(predictdata$rough),3)))
print(paste("standard deviation of real cost:", stat_table[3,6] <-round(sd(realdata$cost),3)))
print(paste("standard deviation of predict cost:", stat_table[3,7] <-round(sd(predictdata$cost),3)))

#WILCOX TEST
wxwear <- wilcox.test(realdata$wear,predictdata$wear,alternative = "two.sided")
wxrough <- wilcox.test(realdata$rough,predictdata$rough,alternative = "two.sided")
wxcost <- wilcox.test(realdata$cost,predictdata$cost,alternative = "two.sided")
print(paste("Wilcox p-value for wear", wxwear$p.value))
print(paste("Wilcox p-value for rough", wxrough$p.value))
print(paste("Wilcox p-value for cost", wxcost$p.value))

#save wilcox test
wx_table <- cbind(c("wear","rough","cost"),round(c(wxwear$p.value,wxrough$p.value,wxcost$p.value),4))
colnames(wx_table) <- c("feature","wilcox test")

#EVALUATE ERROR BY GRINDING WHEEL---------------------------------------------
#identify index of grinding wheel in dataframe
startgr1 <- 0
endgr1 <- dim(subset(dttest[,inputs],gr==0))[1]
startgr2 <- endgr1 + 1
endgr2 <- dim(dttest[,inputs])[1]

print("------------------------------")
print("ERROR GRINDING WHEEL 1 WITHOUT OUTLIER")
gr1realdata <- dttest[startgr1:endgr1,outputs]
gr1predictdata <- as.data.frame(round(predict(modelmlp, dttest[startgr1:endgr1,inputs]),4))
colnames(gr1predictdata) <- c("wear","rough","cost")

#calculate errors
rgr1wearerror <- RMSE(gr1realdata$wear[-1], gr1predictdata$wear[-1])
gr1wearerror <- round(destand(rgr1wearerror, min(dt$wear), max(dt$wear)),4) - min(dt$wear)

rgr1rougherror <- RMSE(gr1realdata$rough[-1], gr1predictdata$rough[-1])
gr1rougherror <- round(destand(rgr1rougherror, min(dt$rough), max(dt$rough)),4) - min(dt$rough)

rgr1costerror <- RMSE(gr1realdata$cost[-1], gr1predictdata$cost[-1])
gr1costerror <- round(destand(rgr1costerror, min(dt$cost), max(dt$cost)),4) - min(dt$cost)

gr1totalerror <- round(sum(rgr1wearerror,rgr1rougherror,rgr1costerror)/3,4)

#show errors
print("REAL SCALE----:")
print(paste("total rmse:",error_table[1,3] <- gr1totalerror))
print(paste("wear rmse converted to real scale:",error_table[2,3] <- gr1wearerror))
print(paste("rough rmse converted to real scale:",error_table[3,3] <- gr1rougherror))
print(paste("cost rmse converted to real scale:",error_table[4,3] <- round(gr1costerror,4)))

print("WEAR----:")
error_table <- calcAllErrors(gr1realdata$wear[-1], gr1predictdata$wear[-1], c(5,6,7,8), 3)
print("ROUGH----:")
error_table <- calcAllErrors(gr1realdata$rough[-1], gr1predictdata$rough[-1], c(9,10,11,12), 3)
print("COST----:")
error_table <- calcAllErrors(gr1realdata$cost[-1], gr1predictdata$cost[-1], c(13,14,15,16), 3)

#show graphical errors
png(filename=file.path(paste("plots/5_gr1_wear_error_without_outlier.png",sep="")), width=wg, height=hg, res=rg)
plot(round(destand(gr1realdata$wear[-1], min(dt$wear), max(dt$wear)),4), type="b", col="blue", main="3NQ: Wear RMSE converted to real scale without outlier", ylim= c(0,160), xlab="Configuration", ylab=wearlab, xaxt='n')
par(new=T)
plot(round(destand(gr1predictdata$wear[-1], min(dt$wear), max(dt$wear)),4), type="b", col="red", ylim= c(0,160), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real"),
       col=c("red", "blue"), lty=1, cex=0.8)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
graphics.off()

png(filename=file.path(paste("plots/small/5_gr1_wear_error_without_outlier.png",sep="")), width=wp, height=hp, res=rp)
plot(round(destand(gr1realdata$wear[-1], min(dt$wear), max(dt$wear)),4), type="b", col="blue", main="3NQ: Wear RMSE converted to real scale without outlier", ylim= c(0,160), xlab="Configuration", ylab=wearlab, xaxt='n')
par(new=T)
plot(round(destand(gr1predictdata$wear[-1], min(dt$wear), max(dt$wear)),4), type="b", col="red", ylim= c(0,160), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real"),
       col=c("red", "blue"), lty=1, cex=0.8)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
graphics.off()

png(filename=file.path(paste("plots/6_gr1_rough_error_without_outlier.png",sep="")), width =wg, height=hg, res=rg)
plot(round(destand(gr1realdata$rough[-1], min(dt$rough), max(dt$rough)),4), type="b", col="blue", main="3NQ: Roughness RMSE converted to real scale without outlier", ylim= c(0,1.5), xlab="Configuration", ylab=roughlab, xaxt='n')
par(new=T)
plot(round(destand(gr1predictdata$rough[-1], min(dt$rough), max(dt$rough)),4), type="b", col="red", ylim= c(0,1.5), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real"),
       col=c("red", "blue"), lty=1, cex=0.8)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
graphics.off()

png(filename=file.path(paste("plots/small/6_gr1_rough_error_without_outlier.png",sep="")), width =wp, height=hp, res=rp)
plot(round(destand(gr1realdata$rough[-1], min(dt$rough), max(dt$rough)),4), type="b", col="blue", main="3NQ: Roughness RMSE converted to real scale without outlier", ylim= c(0,1.5), xlab="Configuration", ylab=roughlab, xaxt='n')
par(new=T)
plot(round(destand(gr1predictdata$rough[-1], min(dt$rough), max(dt$rough)),4), type="b", col="red", ylim= c(0,1.5), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real"),
       col=c("red", "blue"), lty=1, cex=0.8)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
graphics.off()

png(filename=file.path(paste("plots/7_gr1_cost_error_without_outlier.png",sep="")), width =wg, height=hg, res=rg)
plot(round(destand(gr1realdata$cost[-1], min(dt$cost), max(dt$cost)),4)/coin, type="b", col="blue", main="3NQ: Cost RMSE converted to real scale without outlier", ylim = c(0,25), xlab="Configuration", ylab=costlab, xaxt='n')
par(new=T)
plot(round(destand(gr1predictdata$cost[-1], min(dt$cost), max(dt$cost)),4)/coin, type="b", col="red", ylim = c(0,25), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real"),
       col=c("red", "blue"), lty=1, cex=0.8)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
graphics.off()

png(filename=file.path(paste("plots/small/7_gr1_cost_error_without_outlier.png",sep="")), width =wp, height=hp, res=rp)
plot(round(destand(gr1realdata$cost[-1], min(dt$cost), max(dt$cost)),4)/coin, type="b", col="blue", main="3NQ: Cost RMSE converted to real scale without outlier", ylim = c(0,25), xlab="Configuration", ylab=costlab, xaxt='n')
par(new=T)
plot(round(destand(gr1predictdata$cost[-1], min(dt$cost), max(dt$cost)),4)/coin, type="b", col="red", ylim = c(0,25), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real"),
       col=c("red", "blue"), lty=1, cex=0.8)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
graphics.off()

print("------------------------------")
print("ERROR GRINDING WHEEL 1")

#calculate errors
rgr1wearerror <- RMSE(gr1realdata$wear, gr1predictdata$wear)
gr1wearerror <- round(destand(rgr1wearerror, min(dt$wear), max(dt$wear)),4) - min(dt$wear)

rgr1rougherror <- RMSE(gr1realdata$rough, gr1predictdata$rough)
gr1rougherror <- round(destand(rgr1rougherror, min(dt$rough), max(dt$rough)),4) - min(dt$rough)

rgr1costerror <- RMSE(gr1realdata$cost, gr1predictdata$cost)
gr1costerror <- round(destand(rgr1costerror, min(dt$cost), max(dt$cost)),4) - min(dt$cost)

gr1totalerror <- round(sum(rgr1wearerror,rgr1rougherror,rgr1costerror)/3,4)

#show errors
print("REAL SCALE----:")
print(paste("total rmse:",error_table[1,4] <- gr1totalerror))
print(paste("wear rmse converted to real scale:",error_table[2,4] <- gr1wearerror))
print(paste("rough rmse converted to real scale:",error_table[3,4] <- gr1rougherror))
print(paste("cost rmse converted to real scale:",error_table[4,4] <- gr1costerror))

print("WEAR----:")
error_table <- calcAllErrors(gr1realdata$wear, gr1predictdata$wear, c(5,6,7,8), 4)
print("ROUGH----:")
error_table <- calcAllErrors(gr1realdata$rough, gr1predictdata$rough, c(9,10,11,12), 4)
print("COST----:")
error_table <- calcAllErrors(gr1realdata$cost, gr1predictdata$cost, c(13,14,15,16), 4)

#show graphical errors
png(filename=file.path(paste("plots/8_gr1_wear_error_with_outlier.png",sep="")), width =wg, height=hg, res=rg)
plot(round(destand(gr1realdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="blue", main="3NQ: Wear RMSE converted to real scale", ylim= c(0,160), xlab="Configuration", ylab=wearlab, xaxt='n')
par(new=T)
plot(round(destand(gr1predictdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="red", ylim= c(0,160), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=0.8)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
abline(v=1, col="black", lty=3)
graphics.off()

png(filename=file.path(paste("plots/small/8_gr1_wear_error_with_outlier.png",sep="")), width =wp, height=hp, res=rp)
plot(round(destand(gr1realdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="blue", main="3NQ: Wear RMSE converted to real scale", ylim= c(0,160), xlab="Configuration", ylab=wearlab, xaxt='n')
par(new=T)
plot(round(destand(gr1predictdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="red", ylim= c(0,160), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=0.8)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
abline(v=1, col="black", lty=3)
graphics.off()

png(filename=file.path(paste("plots/9_gr1_rough_error_with_outlier.png",sep="")), width =wg, height=hg, res=rg)
plot(round(destand(gr1realdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="blue", main="3NQ: Roughness RMSE converted to real scale", ylim= c(0,1.5), xlab="Configuration", ylab=roughlab, xaxt='n')
par(new=T)
plot(round(destand(gr1predictdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="red", ylim= c(0,1.5), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=0.8)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
abline(v=1, col="black", lty=3)
graphics.off()

png(filename=file.path(paste("plots/small/9_gr1_rough_error_with_outlier.png",sep="")), width =wp, height=hp, res=rp)
plot(round(destand(gr1realdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="blue", main="3NQ: Roughness RMSE converted to real scale", ylim= c(0,1.5), xlab="Configuration", ylab=roughlab, xaxt='n')
par(new=T)
plot(round(destand(gr1predictdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="red", ylim= c(0,1.5), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=0.8)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
abline(v=1, col="black", lty=3)
graphics.off()

png(filename=file.path(paste("plots/10_gr1_cost_error_with_outlier.png",sep="")), width =wg, height=hg, res=rg)
plot(round(destand(gr1realdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="blue", main="3NQ: Cost RMSE converted to real scale", ylim = c(0,25), xlab="Configuration", ylab=costlab, xaxt='n')
par(new=T)
plot(round(destand(gr1predictdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="red", ylim = c(0,25), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=0.8)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
abline(v=1, col="black", lty=3)
graphics.off()

png(filename=file.path(paste("plots/small/10_gr1_cost_error_with_outlier.png",sep="")), width =wp, height=hp, res=rp)
plot(round(destand(gr1realdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="blue", main="3NQ: Cost RMSE converted to real scale", ylim = c(0,25), xlab="Configuration", ylab=costlab, xaxt='n')
par(new=T)
plot(round(destand(gr1predictdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="red", ylim = c(0,25), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=0.8)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
abline(v=1, col="black", lty=3)
graphics.off()

print("------------------------------")
print("ERROR GRINDING WHEEL 2 WITHOUT OUTLIER")
gr2realdata <- dttest[startgr2:endgr2,outputs]
gr2predictdata <- as.data.frame(round(predict(modelmlp, dttest[startgr2:endgr2,inputs]),4))
colnames(gr2predictdata) <- c("wear","rough","cost")

#calculate error without outlier
rgr2wearerror <- RMSE(gr2realdata$wear[-4], gr2predictdata$wear[-4])
gr2wearerror <- round(destand(rgr2wearerror[-4], min(dt$wear), max(dt$wear)),4) - min(dt$wear)

rgr2rougherror <- RMSE(gr2realdata$rough[-4], gr2predictdata$rough[-4])
gr2rougherror <- round(destand(rgr2rougherror[-4], min(dt$rough), max(dt$rough)),4) - min(dt$rough)

rgr2costerror <- RMSE(gr2realdata$cost[-4], gr2predictdata$cost[-4])
gr2costerror <- round(destand(rgr2costerror[-4], min(dt$cost), max(dt$cost)),4) - min(dt$cost)

gr2totalerror <- round(sum(rgr2wearerror,rgr2rougherror,rgr2costerror)/3,4)

#show errors
print("REAL SCALE----:")
print(paste("total rmse:",error_table[1,5] <- gr2totalerror))
print(paste("wear rmse converted to real scale:",error_table[2,5] <- gr2wearerror))
print(paste("rough mean error converted to real scale:",error_table[3,5] <- gr2rougherror))
print(paste("cost rmse: converted to real scale",error_table[4,5] <- gr2costerror))

print("WEAR----:")
error_table <- calcAllErrors(gr2realdata$wear[-4], gr2predictdata$wear[-4], c(5,6,7,8), 5)
print("ROUGH----:")
error_table <- calcAllErrors(gr2realdata$rough[-4], gr2predictdata$rough[-4], c(9,10,11,12), 5)
print("COST----:")
error_table <- calcAllErrors(gr2realdata$cost[-4], gr2predictdata$cost[-4], c(13,14,15,16), 5)

#show graphical errors
png(filename=file.path(paste("plots/11_gr2_wear_error_without_outlier.png",sep="")), width =wg, height=hg, res=rg)
plot(round(destand(gr2realdata$wear[-4], min(dt$wear), max(dt$wear)),4), type="b", col="blue", main="cBN: Wear RMSE converted to real scale without outlier", ylim= c(0,8), xlab="Configuration", ylab=wearlab, xaxt='n')
par(new=T)
plot(round(destand(gr2predictdata$wear[-4], min(dt$wear), max(dt$wear)),4), type="b", col="red", ylim= c(0,8), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real"),
       col=c("red", "blue"), lty=1, cex=0.8)
axis(side=1, at=c(0:dim(gr2realdata[-4,])[1]))
graphics.off()

png(filename=file.path(paste("plots/small/11_gr2_wear_error_without_outlier.png",sep="")), width =wp, height=hp, res=rp)
plot(round(destand(gr2realdata$wear[-4], min(dt$wear), max(dt$wear)),4), type="b", col="blue", main="cBN: Wear RMSE converted to real scale without outlier", ylim= c(0,8), xlab="Configuration", ylab=wearlab, xaxt='n')
par(new=T)
plot(round(destand(gr2predictdata$wear[-4], min(dt$wear), max(dt$wear)),4), type="b", col="red", ylim= c(0,8), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real"),
       col=c("red", "blue"), lty=1, cex=0.8)
axis(side=1, at=c(0:dim(gr2realdata[-4,])[1]))
graphics.off()

png(filename=file.path(paste("plots/12_gr2_rough_error_without_outlier.png",sep="")), width =wg, height=hg, res=rg)
plot(round(destand(gr2realdata$rough[-4], min(dt$rough), max(dt$rough)),4), type="b", col="blue", main="cBN: Roughness RMSE converted to real scale without outlier", ylim= c(0,0.7), xlab="Configuration", ylab=roughlab, xaxt='n')
par(new=T)
plot(round(destand(gr2predictdata$rough[-4], min(dt$rough), max(dt$rough)),4), type="b", col="red", ylim= c(0,0.7), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real"),
       col=c("red", "blue"), lty=1, cex=0.8)
axis(side=1, at=c(0:dim(gr2realdata[-4,])[1]))
graphics.off()

png(filename=file.path(paste("plots/small/12_gr2_rough_error_without_outlier.png",sep="")), width =wp, height=hp, res=rp)
plot(round(destand(gr2realdata$rough[-4], min(dt$rough), max(dt$rough)),4), type="b", col="blue", main="cBN: Roughness RMSE converted to real scale without outlier", ylim= c(0,0.7), xlab="Configuration", ylab=roughlab, xaxt='n')
par(new=T)
plot(round(destand(gr2predictdata$rough[-4], min(dt$rough), max(dt$rough)),4), type="b", col="red", ylim= c(0,0.7), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real"),
       col=c("red", "blue"), lty=1, cex=0.8)
axis(side=1, at=c(0:dim(gr2realdata[-4,])[1]))
graphics.off()

png(filename=file.path(paste("plots/13_gr2_cost_error_without_outlier.png",sep="")), width =wg, height=hg, res=rg)
plot(round(destand(gr2realdata$cost[-4], min(dt$cost), max(dt$cost)),4)/coin, type="b", col="blue", main="cBN: Cost RMSE converted to real scale without outlier", ylim = c(0,50), xlab="Configuration", ylab=costlab, xaxt='n')
par(new=T)
plot(round(destand(gr2predictdata$cost[-4], min(dt$cost), max(dt$cost)),4)/coin, type="b", col="red", ylim = c(0,50), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real"),
       col=c("red", "blue"), lty=1, cex=0.8)
axis(side=1, at=c(0:dim(gr2realdata[-4,])[1]))
graphics.off()

png(filename=file.path(paste("plots/small/13_gr2_cost_error_without_outlier.png",sep="")), width =wp, height=hp, res=rp)
plot(round(destand(gr2realdata$cost[-4], min(dt$cost), max(dt$cost)),4)/coin, type="b", col="blue", main="cBN: Cost RMSE converted to real scale without outlier", ylim = c(0,50), xlab="Configuration", ylab=costlab, xaxt='n')
par(new=T)
plot(round(destand(gr2predictdata$cost[-4], min(dt$cost), max(dt$cost)),4)/coin, type="b", col="red", ylim = c(0,50), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real"),
       col=c("red", "blue"), lty=1, cex=0.8)
axis(side=1, at=c(0:dim(gr2realdata[-4,])[1]))
graphics.off()

print("------------------------------")
print("ERROR GRINDING WHEEL 2")

#calculate errors
rgr2wearerror <- RMSE(gr2realdata$wear, gr2predictdata$wear)
gr2wearerror <- round(destand(rgr2wearerror, min(dt$wear), max(dt$wear)),4) - min(dt$wear)

rgr2rougherror <- RMSE(gr2realdata$rough, gr2predictdata$rough)
gr2rougherror <- round(destand(rgr2rougherror, min(dt$rough), max(dt$rough)),4) - min(dt$rough)

rgr2costerror <- RMSE(gr2realdata$cost, gr2predictdata$cost)
gr2costerror <- round(destand(rgr2costerror, min(dt$cost), max(dt$cost)),4) - min(dt$cost)

gr2totalerror <- round(sum(rgr2wearerror,rgr2rougherror,rgr2costerror)/3,4)

#show errors
print("REAL SCALE----:")
print(paste("total rmse:",error_table[1,6] <- gr2totalerror))
print(paste("wear rmse converted to real scale:",error_table[2,6] <- gr2wearerror))
print(paste("rough rmse converted to real scale:",error_table[3,6] <- gr2rougherror))
print(paste("cost rmse converted to real scale:",error_table[4,6] <- gr2costerror))

print("WEAR----:")
error_table <- calcAllErrors(gr2realdata$wear, gr2predictdata$wear, c(5,6,7,8), 6)
print("ROUGH----:")
error_table <- calcAllErrors(gr2realdata$rough, gr2predictdata$rough, c(9,10,11,12), 6)
print("COST----:")
error_table <- calcAllErrors(gr2realdata$cost, gr2predictdata$cost, c(13,14,15,16), 6)

#show graphical errors
png(filename=file.path(paste("plots/14_gr2_wear_error_with_outlier.png",sep="")), width =wg, height=hg, res=rg)
plot(round(destand(gr2realdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="blue", main="cBN: Wear RMSE converted to real scale", ylim= c(0,25), xlab="Configuration", ylab=wearlab, xaxt='n')
par(new=T)
plot(round(destand(gr2predictdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="red", ylim= c(0,25), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=0.8)
axis(side=1, at=c(0:dim(gr2realdata)[1]))
abline(v=4, col="black", lty=3)
graphics.off()

png(filename=file.path(paste("plots/small/14_gr2_wear_error_with_outlier.png",sep="")), width =wp, height=hp, res=rp)
plot(round(destand(gr2realdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="blue", main="cBN: Wear RMSE converted to real scale", ylim= c(0,25), xlab="Configuration", ylab=wearlab, xaxt='n')
par(new=T)
plot(round(destand(gr2predictdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="red", ylim= c(0,25), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=0.8)
axis(side=1, at=c(0:dim(gr2realdata)[1]))
abline(v=4, col="black", lty=3)
graphics.off()

png(filename=file.path(paste("plots/15_gr2_rough_error_with_outlier.png",sep="")), width =wg, height=hg, res=rg)
plot(round(destand(gr2realdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="blue", main="cBN: Roughness RMSE converted to real scale", ylim= c(0,1), xlab="Configuration", ylab=roughlab, xaxt='n')
par(new=T)
plot(round(destand(gr2predictdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="red", ylim= c(0,1), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=0.8)
axis(side=1, at=c(0:dim(gr2realdata)[1]))
abline(v=4, col="black", lty=3)
graphics.off()

png(filename=file.path(paste("plots/small/15_gr2_rough_error_with_outlier.png",sep="")), width =wp, height=hp, res=rp)
plot(round(destand(gr2realdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="blue", main="cBN: Roughness RMSE converted to real scale", ylim= c(0,1), xlab="Configuration", ylab=roughlab, xaxt='n')
par(new=T)
plot(round(destand(gr2predictdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="red", ylim= c(0,1), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=0.8)
axis(side=1, at=c(0:dim(gr2realdata)[1]))
abline(v=4, col="black", lty=3)
graphics.off()

png(filename=file.path(paste("plots/16_gr2_cost_error_with_outlier.png",sep="")), width =wg, height=hg, res=rg)
plot(round(destand(gr2realdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="blue", main="cBN: Cost RMSE converted to real scale", ylim = c(0,50), xlab="Configuration", ylab=costlab, xaxt='n')
par(new=T)
plot(round(destand(gr2predictdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="red", ylim = c(0,50), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=0.8)
axis(side=1, at=c(0:dim(gr2realdata)[1]))
abline(v=4, col="black", lty=3)
graphics.off()

png(filename=file.path(paste("plots/small/16_gr2_cost_error_with_outlier.png",sep="")), width =wp, height=hp, res=rp)
plot(round(destand(gr2realdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="blue", main="cBN: Cost RMSE converted to real scale", ylim = c(0,50), xlab="Configuration", ylab=costlab, xaxt='n')
par(new=T)
plot(round(destand(gr2predictdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="red", ylim = c(0,50), xlab="", ylab="", xaxt='n',panel.first=grid())
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=0.8)
axis(side=1, at=c(0:dim(gr2realdata)[1]))
abline(v=4, col="black", lty=3)
graphics.off()

#show elements real vs. predict
#plot3d(rbind(gr1realdata,gr1predictdata), size=8, col=c(rep(1,dim(gr1realdata)[1]),rep(2,dim(gr1realdata)[1])))
#plot3d(rbind(gr2realdata,gr2predictdata), size=8, col=c(rep(1,dim(gr2realdata)[1]),rep(2,dim(gr2realdata)[1])))

#EXPORT CONFIG FILE------------------------------------
config <- cbind(speedinf,
                speedsup,
                rateinf,
                ratesup,
                volinf,
                volsup,
                min(dt$wear),
                max(dt$wear),
                min(dt$rough),
                max(dt$rough),
                min(dt$cost),
                max(dt$cost))

colnames(config) <- c("speedinf","speedsup","rateinf","ratesup","volinf","volsup´","wearmin","wearmax","roughmin","roughmax","costmin","costmax")

write.table(config,file=paste("../spec-optimization/config.csv",sep=""),na="",row.names = FALSE,sep=";",quote=FALSE)

#EXPORT RESULTS TABLES
error_table[4,2:6] <- round(as.numeric(error_table[4,2:6])/coin,4)
write.table(error_table,file=paste("tables/table_errors.csv",sep=""),na="",row.names = FALSE,sep=";",quote=FALSE)
write.table(stat_table,file=paste("tables/table_stat.csv",sep=""),na="",row.names = FALSE,sep=";",quote=FALSE)
write.table(wx_table,file=paste("tables/table_wx.csv",sep=""),na="",row.names = FALSE,sep=";",quote=FALSE)
write.table(round(correaldata,4),file=paste("tables/table_cor_realdata.csv",sep=""),na="",row.names = FALSE,sep=";",quote=FALSE)
write.table(round(corpredictdata,4),file=paste("tables/table_cor_predictdata.csv",sep=""),na="",row.names = FALSE,sep=";",quote=FALSE)

#EXPORTE JOIN PLOT
png(filename=paste('plots/0_errorjoin23.png',sep=""), width=1500, height=730, res=100)

plot.new()
frame()
par(mfrow=c(2,3))

plot(round(destand(gr1realdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="blue", ylim= c(0,180), xlab="Configuration", ylab=wearlab, xaxt='n', cex.lab=1.2, lwd=2)
par(new=T)
plot(round(destand(gr1predictdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="red", ylim= c(0,180), xlab="", ylab="", xaxt='n',panel.first=grid(), lwd=2)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
mtext("3NQ: Wear", side = 3, cex = 1.2)
abline(v=1, col="black", lty=3, lwd=2)
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=1.2, horiz = TRUE, lwd=2)

plot(round(destand(gr1realdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="blue", ylim= c(0,1.5), xlab="Configuration", ylab=roughlab, xaxt='n', cex.lab=1.2, lwd=2)
par(new=T)
plot(round(destand(gr1predictdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="red", ylim= c(0,1.5), xlab="", ylab="", xaxt='n',panel.first=grid(), lwd=2)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
mtext("3NQ: Roughness", side = 3, cex = 1.2)
abline(v=1, col="black", lty=3, lwd=2)
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=1.2, horiz = TRUE, lwd=2)

plot(round(destand(gr1realdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="blue", ylim = c(0,30), xlab="Configuration", ylab=costlab, xaxt='n', cex.lab=1.2, lwd=2)
par(new=T)
plot(round(destand(gr1predictdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="red", ylim = c(0,30), xlab="", ylab="", xaxt='n',panel.first=grid(), lwd=2)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
mtext("3NQ: Cost", side = 3, cex = 1.2)
abline(v=1, col="black", lty=3, lwd=2)
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=1.2, horiz = TRUE, lwd=2)

plot(round(destand(gr2realdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="blue", ylim= c(0,25), xlab="Configuration", ylab=wearlab, xaxt='n', cex.lab=1.2, lwd=2)
par(new=T)
plot(round(destand(gr2predictdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="red", ylim= c(0,25), xlab="", ylab="", xaxt='n',panel.first=grid(), lwd=2)
axis(side=1, at=c(0:dim(gr2realdata)[1]))
mtext("cBN: Wear", side = 3, cex = 1.2)
abline(v=4, col="black", lty=3, lwd=2)
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=1.2, horiz = TRUE, lwd=2)

plot(round(destand(gr2realdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="blue", ylim= c(0,1), xlab="Configuration", ylab=roughlab, xaxt='n', cex.lab=1.2, lwd=2)
par(new=T)
plot(round(destand(gr2predictdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="red", ylim= c(0,1), xlab="", ylab="", xaxt='n',panel.first=grid(), lwd=2)
axis(side=1, at=c(0:dim(gr2realdata)[1]))
mtext("cBN: Roughness", side = 3, cex = 1.2)
abline(v=4, col="black", lty=3, lwd=2)
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=1.2, horiz = TRUE, lwd=2)

plot(round(destand(gr2realdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="blue", ylim = c(0,50), xlab="Configuration", ylab=costlab, xaxt='n', cex.lab=1.2, lwd=2)
par(new=T)
plot(round(destand(gr2predictdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="red", ylim = c(0,50), xlab="", ylab="", xaxt='n',panel.first=grid(), lwd=2)
axis(side=1, at=c(0:dim(gr2realdata)[1]))
mtext("cBN: Cost", side = 3, cex = 1.2)
abline(v=4, col="black", lty=3, lwd=2)
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=1.2, horiz = TRUE, lwd=2)

dev.off()



png(filename=paste('plots/0_errorjoin32.png',sep=""), width=1500, height=1290, res=160)
plot.new()
frame()
par(mfrow=c(3,2))
cextitle = 0.8

plot(round(destand(gr1realdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="blue", ylim= c(0,180), xlab="Configuration", ylab=wearlab, xaxt='n', cex.lab=1.2, lwd=2)
par(new=T)
plot(round(destand(gr1predictdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="red", ylim= c(0,180), xlab="", ylab="", xaxt='n',panel.first=grid(), lwd=2)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
mtext("3NQ: Wear", side = 3, cex = cextitle)
abline(v=1, col="black", lty=3, lwd=2)
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=1, horiz = TRUE, lwd=2)

plot(round(destand(gr1realdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="blue", ylim= c(0,1.5), xlab="Configuration", ylab=roughlab, xaxt='n', cex.lab=1.2, lwd=2)
par(new=T)
plot(round(destand(gr1predictdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="red", ylim= c(0,1.5), xlab="", ylab="", xaxt='n',panel.first=grid(), lwd=2)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
mtext("3NQ: Roughness", side = 3, cex = cextitle)
abline(v=4, col="black", lty=3, lwd=2)
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=1, horiz = TRUE, lwd=2)

plot(round(destand(gr1realdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="blue", ylim = c(0,30), xlab="Configuration", ylab=costlab, xaxt='n', cex.lab=1.2, lwd=2)
par(new=T)
plot(round(destand(gr1predictdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="red", ylim = c(0,30), xlab="", ylab="", xaxt='n',panel.first=grid(), lwd=2)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
mtext("3NQ: Cost", side = 3, cex = cextitle)
abline(v=12, col="black", lty=3, lwd=2)
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=1, horiz = TRUE, lwd=2)

plot(round(destand(gr2realdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="blue", ylim= c(0,25), xlab="Configuration", ylab=wearlab, xaxt='n', cex.lab=1.2, lwd=2)
par(new=T)
plot(round(destand(gr2predictdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="red", ylim= c(0,25), xlab="", ylab="", xaxt='n',panel.first=grid(), lwd=2)
axis(side=1, at=c(0:dim(gr2realdata)[1]))
mtext("cBN: Wear", side = 3, cex = cextitle)
abline(v=4, col="black", lty=3, lwd=2)
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=1, horiz = TRUE, lwd=2)

plot(round(destand(gr2realdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="blue", ylim= c(0,1), xlab="Configuration", ylab=roughlab, xaxt='n', cex.lab=1.2, lwd=2)
par(new=T)
plot(round(destand(gr2predictdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="red", ylim= c(0,1), xlab="", ylab="", xaxt='n',panel.first=grid(), lwd=2)
axis(side=1, at=c(0:dim(gr2realdata)[1]))
mtext("cBN: Roughness", side = 3, cex = cextitle)
abline(v=4, col="black", lty=3, lwd=2)
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=1, horiz = TRUE, lwd=2)

plot(round(destand(gr2realdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="blue", ylim = c(0,50), xlab="Configuration", ylab=costlab, xaxt='n', cex.lab=1.2, lwd=2)
par(new=T)
plot(round(destand(gr2predictdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="red", ylim = c(0,50), xlab="", ylab="", xaxt='n',panel.first=grid(), lwd=2)
axis(side=1, at=c(0:dim(gr2realdata)[1]))
mtext("cBN: Cost", side = 3, cex = cextitle)
abline(v=4, col="black", lty=3, lwd=2)
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=1, horiz = TRUE, lwd=2)

dev.off()



#PLOTS USED IN PAPER

#FIGURE boxplot.pdf
#boxplot(bp, col=(c("lightgray","white")))

#FIGURE predictedvsreal.pdf
plot.new()
frame()
par(mfrow=c(3,2))
cextitle = 0.8

plot(round(destand(gr1realdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="blue", 
     ylim= c(0,220), xlab="Configuration", ylab=wearlab, xaxt='n', cex.lab=1.4, lwd=2)
par(new=T)
plot(round(destand(gr1predictdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="red", 
     ylim= c(0,220), xlab="", ylab="", xaxt='n',panel.first=grid(), lwd=2)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
mtext("3NQ: Wear", side = 3, cex = cextitle)
abline(v=1, col="black", lty=3, lwd=2)
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=1, horiz = TRUE, lwd=2, pt.cex = 0.5)

plot(round(destand(gr1realdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="blue", ylim= c(0,1.5), xlab="Configuration", ylab=roughlab, xaxt='n', cex.lab=1.4, lwd=2)
par(new=T)
plot(round(destand(gr1predictdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="red", ylim= c(0,1.5), xlab="", ylab="", xaxt='n',panel.first=grid(), lwd=2)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
mtext("3NQ: Roughness", side = 3, cex = cextitle)
abline(v=4, col="black", lty=3, lwd=2)
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=1, horiz = TRUE, lwd=2, pt.cex = 0.5)

plot(round(destand(gr1realdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="blue", ylim = c(0,30), xlab="Configuration", ylab=costlab, xaxt='n', cex.lab=1.4, lwd=2)
par(new=T)
plot(round(destand(gr1predictdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="red", ylim = c(0,30), xlab="", ylab="", xaxt='n',panel.first=grid(), lwd=2)
axis(side=1, at=c(0:dim(gr1realdata)[1]))
mtext("3NQ: Cost", side = 3, cex = cextitle)
abline(v=12, col="black", lty=3, lwd=2)
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=1, horiz = TRUE, lwd=2, pt.cex = 0.5)

plot(round(destand(gr2realdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="blue", ylim= c(0,25), xlab="Configuration", ylab=wearlab, xaxt='n', cex.lab=1.4, lwd=2)
par(new=T)
plot(round(destand(gr2predictdata$wear, min(dt$wear), max(dt$wear)),4), type="b", col="red", ylim= c(0,25), xlab="", ylab="", xaxt='n',panel.first=grid(), lwd=2)
axis(side=1, at=c(0:dim(gr2realdata)[1]))
mtext("cBN: Wear", side = 3, cex = cextitle)
abline(v=4, col="black", lty=3, lwd=2)
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=1, horiz = TRUE, lwd=2, pt.cex = 0.5)

plot(round(destand(gr2realdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="blue", ylim= c(0,1), xlab="Configuration", ylab=roughlab, xaxt='n', cex.lab=1.4, lwd=2)
par(new=T)
plot(round(destand(gr2predictdata$rough, min(dt$rough), max(dt$rough)),4), type="b", col="red", ylim= c(0,1), xlab="", ylab="", xaxt='n',panel.first=grid(), lwd=2)
axis(side=1, at=c(0:dim(gr2realdata)[1]))
mtext("cBN: Roughness", side = 3, cex = cextitle)
abline(v=4, col="black", lty=3, lwd=2)
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=1, horiz = TRUE, lwd=2, pt.cex = 0.5)

plot(round(destand(gr2realdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="blue", ylim = c(0,60), xlab="Configuration", ylab=costlab, xaxt='n', cex.lab=1.4, lwd=2)
par(new=T)
plot(round(destand(gr2predictdata$cost, min(dt$cost), max(dt$cost)),4)/coin, type="b", col="red", ylim = c(0,60), xlab="", ylab="", xaxt='n',panel.first=grid(), lwd=2)
axis(side=1, at=c(0:dim(gr2realdata)[1]))
mtext("cBN: Cost", side = 3, cex = cextitle)
abline(v=4, col="black", lty=3, lwd=2)
legend("topright", legend=c("Predicted", "Real", "Outlier"),
       col=c("red", "blue", "black"), lty=c(1,1,2), cex=1, horiz = TRUE, lwd=2, pt.cex = 0.5)

