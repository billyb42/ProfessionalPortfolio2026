#Below here will be the code for the questions for Assignment 7
#Here will be the code for question 6
#predict age using age through polynomial regression, CV is use to select optimal degree of d, ANOVA is done for comparison, 
library(ISLR) 
data(Wage)
attach(Wage)
library(ISLR)
library(splines)
library(boot)
Wage
set.seed(1)
cv.errors <- rep(0, 10)#this allows for the storing of errors for degrees 1 to 10
for (d in 1:10) {
  model <- glm(wage ~ poly(age, d), data = Wage)#this fits the polynomial regression and prepares for the varying degrees and cross validation
  cv.errors[d] <- cv.glm(Wage, model, K = 10)$delta[1]#10 fold cv error
}
wage_optimal_degree <- which.min(cv.errors)#this finds the optimal degree from above
print(paste("Optimal Polynomial Degree:", wage_optimal_degree))
best_pr_model <- lm(wage ~ poly(age, wage_optimal_degree), data = Wage)
#below will plot the data
age_grid <- seq(min(age), max(age), length.out = 100)
preds <- predict(best_pr_model, newdata = data.frame(age = age_grid))
plot(age, wage, col = "gray", main = "Polynomial Regression Fit", xlab = "Age", ylab = "Wage")
lines(age_grid, preds, col = "blue", lwd = 2)

#then the comparison with ANOVA testing
anova_results <- list()
for (d in 1:10) {
  anova_results[[d]] <- lm(wage ~ poly(age, d), data = Wage)#this causes a return to the anova_results variable
}

anova(anova_results[[1]], anova_results[[2]], anova_results[[3]], anova_results[[4]], anova_results[[5]], anova_results[[6]], anova_results[[7]], 
      anova_results[[8]], anova_results[[9]], anova_results[[10]])#different polynomial degress will be compared against the ANOVA test
#which is really interesting as we can see the differences between increasing or decreasing the polynomial degrees to the comparison to the ANOVA test
#then a Step-Function Regression is done with cross-validation
set.seed(1)#I dont think this same seed number would cause issues below but I would be curious of a different opinion
cv.errors.step <- rep(NA, 10)
for(i in 1:9) {
  cuts <- i + 1
  Wage$age.cut <- cut(Wage$age, cuts)#this creates step function bins
  model <- glm(wage ~ age.cut, data = Wage)
  cv.errors.step[i] <- cv.glm(Wage, model, K = 10)$delta[1]
}
optimal_cuts <- which.min(cv.errors.step) + 1
print(paste("Optimal number of cuts:", optimal_cuts))
#then the step function will be fit with the optimal cuts
Wage$age.cut <- cut(Wage$age, optimal_cuts)
step_model <- lm(wage ~ age.cut, data = Wage)
age_grid <- seq(min(age), max(age), length.out = 100)
preds_step <- predict(step_model, newdata = data.frame(age.cut = cut(age_grid, optimal_cuts)))
plot(age, wage, col = "gray", main = "With Step Function", xlab = "Age", ylab = "Wage")
lines(age_grid, preds_step, col = "red", lwd = 2, type = "s")

#Below here will be the code for question 7
library(ISLR)
library(ggplot2)
library(dplyr)
library(splines)
library(gam)
data(Wage)
head(Wage)
summary(Wage)
summary(Wage$maritl)
summary(Wage$jobclass)
summary(Wage$education)
ggplot(Wage, aes(x = maritl, y = wage)) + 
  geom_boxplot(fill = "lightblue") + 
  labs(title = "Wage by Marital Status", x = "Marital Status", y = "Wage") + 
  theme_minimal()
ggplot(Wage, aes(x = jobclass, y = wage)) + 
  geom_boxplot(fill = "lightgreen") + 
  labs(title = "Wage by Job Class", x = "Job Class", y = "Wage") + 
  theme_minimal()
ggplot(Wage, aes(x = education, y = wage)) + 
  geom_boxplot(fill = "lightcoral") + 
  labs(title = "Wage by Education Level", x = "Education", y = "Wage") + 
  theme_minimal()
#after the above of exploring some of these variables against the wage measurements
#trying to now fit in a natural spline model
ns_fit <- lm(wage ~ ns(age, df = 4) + education + jobclass, data = Wage)
age.grid <- seq(from = min(Wage$age), to = max(Wage$age), length.out = 100)
preds <- predict(ns_fit, newdata = data.frame(
  age = age.grid, 
  education = "1. < HS Grad", 
  jobclass = "1. Industrial"
))
plot(Wage$age, Wage$wage, col = "gray", main = "Splines with Education and Jobclass", 
     xlab = "Age", ylab = "Wage")
lines(age.grid, preds, col = "blue", lwd = 2)
#then below is the use of the Generalized Additive Model which extends linear models to allow for non-linear relationships between predictor variables and the response variable using smooth functions
gam_model <- gam(wage ~ s(age, df = 5) + education + jobclass, data = Wage)
par(mfrow = c(1, 1))
plot(gam_model, se = TRUE, col = "blue")
summary(gam_model)
#then the GAM with interaction between age and education below
gam_age_education <- gam(wage ~ s(age, df = 4) * education, data = Wage)
plot(gam_age_education, se = TRUE, col = "darkred")
anova(ns_fit, gam_model, gam_age_education, test = "F")#takes and compares all of the models through an ANOVA

#Below here will be the code for question 10
library(ISLR)
data("College")
str(College)
college_data <- College
#For the first part of the question here is to split the data into a training set and a test set, with out-of-state
#tuition is the response and the other variables as the predictors, then a forward stepwise selection on the training set in order
#to identify a satisfactory model that uses just a subset of the predictors
set.seed(389)
train_set <- sample(1:nrow(college_data), 0.7 * nrow(college_data))
train_data <- college_data[train_set, ]#70% of the data is used for the train subset
test_data <- college_data[-train_set, ]#using the train_set, the data is made into two sets of data, with this set being the remaining 30%
#then a forward stepwise selection using Outstate as the response variable and gradually adding predictors
initial_model <- lm(Outstate ~ 1, data = train_data)#starting off with just the intercept in the model
stepwise_model <- step(initial_model, 
                       scope = formula(lm(Outstate ~ ., data = train_data)), 
                       direction = "forward", 
                       trace = 0)#with trace being 0 it suppresses the output
#the "scope" allows the specification of which variables to consider for the predictor, which for this it specifies all of the other variables to be used as predictors
#the forward stepwise direction allows for the start at intercept and then adding predictors one by one
summary(stepwise_model)
#now for a check of the final model to ensure the AIC is optimized
#starting with the test set of data
predictions <- predict(stepwise_model, newdata = test_data)
rmse <- sqrt(mean((predictions - test_data$Outstate)^2))#this evaluates the final model
print(paste("RMSE on Test Set:", round(rmse, 2)))
#now the next step for question 10
#we'll start with doing an spline for the variables that are numeric that might have non-linear effects
library(gam)
gam_model <- gam(Outstate ~ 
                   s(Room.Board) +
                   s(perc.alumni) + 
                   s(Expend) + 
                   Private + 
                   s(PhD) +
                   s(Grad.Rate) + 
                   s(S.F.Ratio) + 
                   s(Personal) + 
                   s(Accept) +
                   s(Apps) + 
                   s(F.Undergrad) + 
                   s(Top10perc) + 
                   s(Terminal), 
                 data = train_data)
par(mfrow = c(4, 4))#this arranges the plots in a grid
quartz()
plot(gam_model, se = TRUE, col = "blue")
#Now the part C of question 10
predictions_gam <- predict(gam_model, newdata = test_data)
rmse_gam <- sqrt(mean((predictions_gam - test_data$Outstate)^2))
print(paste("GAM RMSE on Test Set:", round(rmse_gam, 2)))
#now for part D of question 10
summary(gam_model)

#################################################################################################################
#Below here will be the practice code for Module 7
# Chapter 7 Lab: Non-linear Modeling
library(ISLR) 
attach(Wage)
dim(Wage)
names(Wage)
range(age)

#Fit a Polynomial Regression model
#Poly(_,_) will return a _ by _ matrix
tmp = poly(age, 3)
dim(tmp)
colMeans(tmp)
round(t(tmp) %*% tmp, dig = 4)
fit = lm(wage ~ age + I(age^2) + I(age^3), data = Wage)
round(summary(fit)$coef, dig = 3)
#this below alternative uses the default design matrix where the jth column corresponds to age^j
fit2 = lm(wage ~ age + I(age^2) + I(age^3), data = Wage)
round(summary(fit2)$coef, dig = 3)
#The default design matrix can also be generated by poly
#with option raw = TRUE. 
#fit3 should return the same set of coefficients as fit2
#fit3 = lm(wage ~ poly(age, 3, raw = T), data = Wage)
#round(summary(fit2)$coef, dig = 3)
#the results from fit and fit2 have different coefficients but t-value and p-value for the last predictor are always the same
#The fitted curves between the two are the same
#below is using a predict of wage at age - 82 to show/see if fit and fit2 give the same answer
predict(fit, newdata = list(age = 82))
predict(fit2, newdata = list(age = 82))
#The fitted curve from the all three models should be the same
agelims = range(age)
age.grid = seq(from = agelims[1], to = agelims[2])
preds = predict(fit, newdata = list(age = age.grid), se=TRUE)
plot(age, wage, xlim = agelims, pch = '.', cex = 2, col="darkgrey")
title("Degree -3 Polynomial ")
lines(age.grid, preds$fit, lwd=2, col="blue")
#Forward Selection for the polynomial order d based on the significance of the coefficient of the highest order starting with quadratic polynomial function (d = 3) below
summayr(lm(wage ~ poly(age, 2), data = Wage))$coef
summary(lm(wage ~ poly(age, 3), data = Wage))$coef
summary(lm(wage ~ poly(age, 4), data=Wage))$coef
#Then Back Selection for the polynomial order d based on the significance of the coefficient of the highest order starting with
#d = 6 with d = 3 finally being picked.
summary(lm(wage ~ poly(age, 6), data=Wage))$coef
summary(lm(wage ~ poly(age, 5), data=Wage))$coef
summary(lm(wage ~ poly(age, 4), data=Wage))$coef
summary(lm(wage ~ poly(age, 3), data=Wage))$coef

library(ISLR)
attach(Wage)

# Polynomial Regression and Step Functions

fit=lm(wage~poly(age,4),data=Wage)
coef(summary(fit))
fit2=lm(wage~poly(age,4,raw=T),data=Wage)
coef(summary(fit2))
fit2a=lm(wage~age+I(age^2)+I(age^3)+I(age^4),data=Wage)
coef(fit2a)
fit2b=lm(wage~cbind(age,age^2,age^3,age^4),data=Wage)
agelims=range(age)
age.grid=seq(from=agelims[1],to=agelims[2])
preds=predict(fit,newdata=list(age=age.grid),se=TRUE)
se.bands=cbind(preds$fit+2*preds$se.fit,preds$fit-2*preds$se.fit)
par(mfrow=c(1,2),mar=c(4.5,4.5,1,1),oma=c(0,0,4,0))
plot(age,wage,xlim=agelims,cex=.5,col="darkgrey")
title("Degree-4 Polynomial",outer=T)
lines(age.grid,preds$fit,lwd=2,col="blue")
matlines(age.grid,se.bands,lwd=1,col="blue",lty=3)
preds2=predict(fit2,newdata=list(age=age.grid),se=TRUE)
max(abs(preds$fit-preds2$fit))
fit.1=lm(wage~age,data=Wage)
fit.2=lm(wage~poly(age,2),data=Wage)
fit.3=lm(wage~poly(age,3),data=Wage)
fit.4=lm(wage~poly(age,4),data=Wage)
fit.5=lm(wage~poly(age,5),data=Wage)
anova(fit.1,fit.2,fit.3,fit.4,fit.5)
coef(summary(fit.5))
(-11.983)^2
fit.1=lm(wage~education+age,data=Wage)
fit.2=lm(wage~education+poly(age,2),data=Wage)
fit.3=lm(wage~education+poly(age,3),data=Wage)
anova(fit.1,fit.2,fit.3)
fit=glm(I(wage>250)~poly(age,4),data=Wage,family=binomial)
preds=predict(fit,newdata=list(age=age.grid),se=T)
pfit=exp(preds$fit)/(1+exp(preds$fit))
se.bands.logit = cbind(preds$fit+2*preds$se.fit, preds$fit-2*preds$se.fit)
se.bands = exp(se.bands.logit)/(1+exp(se.bands.logit))
preds=predict(fit,newdata=list(age=age.grid),type="response",se=T)
plot(age,I(wage>250),xlim=agelims,type="n",ylim=c(0,.2))
points(jitter(age), I((wage>250)/5),cex=.5,pch="|",col="darkgrey")
lines(age.grid,pfit,lwd=2, col="blue")
matlines(age.grid,se.bands,lwd=1,col="blue",lty=3)
table(cut(age,4))
fit=lm(wage~cut(age,4),data=Wage)
coef(summary(fit))

# Splines

library(splines)
fit=lm(wage~bs(age,knots=c(25,40,60)),data=Wage)
pred=predict(fit,newdata=list(age=age.grid),se=T)
plot(age,wage,col="gray")
lines(age.grid,pred$fit,lwd=2)
lines(age.grid,pred$fit+2*pred$se,lty="dashed")
lines(age.grid,pred$fit-2*pred$se,lty="dashed")
dim(bs(age,knots=c(25,40,60)))
dim(bs(age,df=6))
attr(bs(age,df=6),"knots")
fit2=lm(wage~ns(age,df=4),data=Wage)
pred2=predict(fit2,newdata=list(age=age.grid),se=T)
lines(age.grid, pred2$fit,col="red",lwd=2)
plot(age,wage,xlim=agelims,cex=.5,col="darkgrey")
title("Smoothing Spline")
fit=smooth.spline(age,wage,df=16)
fit2=smooth.spline(age,wage,cv=TRUE)
fit2$df
lines(fit,col="red",lwd=2)
lines(fit2,col="blue",lwd=2)
legend("topright",legend=c("16 DF","6.8 DF"),col=c("red","blue"),lty=1,lwd=2,cex=.8)
plot(age,wage,xlim=agelims,cex=.5,col="darkgrey")
title("Local Regression")
fit=loess(wage~age,span=.2,data=Wage)
fit2=loess(wage~age,span=.5,data=Wage)
lines(age.grid,predict(fit,data.frame(age=age.grid)),col="red",lwd=2)
lines(age.grid,predict(fit2,data.frame(age=age.grid)),col="blue",lwd=2)
legend("topright",legend=c("Span=0.2","Span=0.5"),col=c("red","blue"),lty=1,lwd=2,cex=.8)

# GAMs

gam1=lm(wage~ns(year,4)+ns(age,5)+education,data=Wage)
library(gam)
gam.m3=gam(wage~s(year,4)+s(age,5)+education,data=Wage)
par(mfrow=c(1,3))
plot(gam.m3, se=TRUE,col="blue")
plot.Gam(gam1, se=TRUE, col="red")
gam.m1=gam(wage~s(age,5)+education,data=Wage)
gam.m2=gam(wage~year+s(age,5)+education,data=Wage)
anova(gam.m1,gam.m2,gam.m3,test="F")
summary(gam.m3)
preds=predict(gam.m2,newdata=Wage)
gam.lo=gam(wage~s(year,df=4)+lo(age,span=0.7)+education,data=Wage)
plot.Gam(gam.lo, se=TRUE, col="green")
gam.lo.i=gam(wage~lo(year,age,span=0.5)+education,data=Wage)
library(akima)
plot(gam.lo.i)
gam.lr=gam(I(wage>250)~year+s(age,df=5)+education,family=binomial,data=Wage)
par(mfrow=c(1,3))
plot(gam.lr,se=T,col="green")
table(education,I(wage>250))
gam.lr.s=gam(I(wage>250)~year+s(age,df=5)+education,family=binomial,data=Wage,subset=(education!="1. < HS Grad"))
plot(gam.lr.s,se=T,col="green")

#Smoothing Splines
library(splines)
help(smooth.spline)
options(digits = 4)
set.seed(1234)
n = 30 
err = 1
x = sort(runif(n))
y = sin(12*(x+0.2))/(x+0.2) + rnorm(n, 0, err);
plot(x, y, col="red");

fx = 1:50/50;
fy = sin(12*(fx+0.2))/(fx+0.2)
lines(fx, fy, col=8, lwd=2);
#fit smoothing spline models with various dfs
par(mfrow=c(2,2));      # 2x2 (totally 4) subplots
plot(x,y, xlab='x', ylab='y');
lines(fx, fy, col=8, lwd=1.5);
lines(predict(smooth.spline(x, y, df=5),fx),  lty=2, col='blue', lwd=1.5);
title('df=5');

plot(x,y,xlab='x', ylab='y');
lines(fx, fy, col=8, lwd=1.5);
lines(predict(smooth.spline(x, y, df=9),fx),  lty=2, col='blue', lwd=1.5);
title('df=9');

plot(x,y,xlab='x', ylab='y');
lines(fx, fy, col=8, lwd=1.5);
lines(predict(smooth.spline(x, y, df=15),fx),  lty=2, col='blue', lwd=1.5);
title('df=15');

plot(x,y,xlab='x', ylab='y');
lines(fx, fy, col=8, lwd=1.5);
lines(predict(smooth.spline(x, y, df=20),fx),  lty=2, col='blue', lwd=1.5);
title('df=20')

#Then how to obtain the Demmler & Reinsch Basis basis where we first obtain the smoother matrix S and then the eigen-vectors of S are basically the DR basis functions
smooth.matrix = function(x, df){
  # return the smoother matrix with knots x and degree of freedom = df
  # this function is for x having unique values
  n = length(x);
  A = matrix(0, n, n);
  for(i in 1:n){
    y = rep(0, n); y[i]=1;
    yi = smooth.spline(x, y, df=df)$y;
    A[,i]= yi;
  }
  return((A+t(A))/2)
}
fx = 1:50/50
S4 = smooth.matrix(fx, df=4);
S9 = smooth.matrix(fx, df=9);
tmp = ns(fx, df=9, intercept=TRUE)
H9 = tmp%*%solve(t(tmp)%*%tmp)%*%t(tmp);
#Obtain the eigen value and eigen vector of the smoother/projection matrices. The eigen vectors of S4 and S9 should be the same, up to a sign flip
eigen.S4 = eigen(S4);
eigen.S9 = eigen(S9);
eigen.H9 = eigen(H9);

v4 = eigen.S4$ve;
v9=  eigen.S9$ve;

par(mfrow=c(3,5));
for(i in 1:15) {
  plot(c(min(x), max(x)),c(min(v4, v9), max(v4, v9)), xlab="x", ylab="", 
       type="n");
  lines(fx, v4[,i], col=2,lty=1, lwd=2.5);
  lines(fx, v9[,i], col=3, lty=2, lwd=2.5);}
#plot the eigen values note the first two eigen values are always 1
plot(eigen.H9$va, pch=5, col="black", cex=1);
points(eigen.S4$va, , col="red", xlab='', ylab='eigen values',
       cex=1.5);
points(eigen.S9$va, pch=4, col="blue", cex=1);
lines(c(0,n), c(1, 1), col=8, lty=2, lwd=1);
legend("topright", pch=c(1,4,5), col=c("red", "blue", "black"),
       legend=c("SS with df=4", "SS with df=9", "NCS with df=9"))
#Check for the effective degree of freedom
sum(diag(S4))
sum(diag(S9))
sum(diag(H9))
#LOO-CV and GCV
fit = smooth.spline(x, y, df=9);
fit$df
sum(fit$lev)
fit$lev  # leveage = diagnal entries of the smoother matrix
diag(smooth.matrix(x, df =9))
fit$cv  # default: GCV
sum((y-fit$y)^2)/(1-fit$df/n)^2/n
fit=smooth.spline(x, y, df=9, cv=T) # set 'cv=T' to return CV 
fit$cv
sum(((y-fit$y)/(1-fit$lev))^2)/n
#Use LOO-CV and GCV to select df, note that same as Ridge regression, smoothing splines could have fractional dfs
df = 2+(1:40)/2
m = length(df)
mycv = rep(0,m)
mygcv = rep(0,m)
for (i in 1:m){
  fit = smooth.spline(x, y, df=df[i]);
  mygcv[i] = fit$cv;
  fit = smooth.spline(x, y, df=df[i], cv=T);
  mycv[i] = fit$cv
}

plot(c(1,20), range(mycv, mygcv)+c(-0.5,0.5), xlab='df', 
     ylab='CV scores', type='n')
points(df, mycv, col="red", cex=2);
points(df, mygcv, col="blue", cex=2);
optdf = df[mygcv==min(mygcv)]
optdf
fitopt = smooth.spline(x, y, df=optdf);
plot(x, y, xlab="x", ylab="y")
lines(predict(fitopt, (1:100)/100),col="red", lwd=2)
lines(fx, fy, col="gray", lwd=2)
#Lecture Practice codes for the module, on Local Smoother
library(faraway)
par(mfrow=c(1,3))
data(exa)
#?exa # info for this simulated dataset
plot (y ~ x, exa, main="Example A")
lines(m ~ x, exa)
data(exb)
#?exb
plot(y ~ x, exb, main="Example B")
lines(m ~ x, exb)
data(faithful)
#?faithful
plot(waiting ~ eruptions, faithful,main="old Faithful")
#Kernel smoother with different bandwidths for the Old Faithful Data
?ksmooth

par(mfrow=c(1,3))
plot(waiting ~ eruptions,
     faithful,main="bandwidth=0.1", col="gray", cex=0.5)
lines(ksmooth(faithful$eruptions, faithful$waiting, "normal", 0.1), lwd=2)

plot(waiting ~ eruptions, faithful, main="bandwidth=0.5", col="gray", cex=0.5)
lines(ksmooth(faithful$eruptions, faithful$waiting,"normal", 0.5), lwd=2)

plot(waiting ~ eruptions, faithful, main="bandwidth=2", col="gray", cex=0.5)
lines(ksmooth(faithful$eruptions, faithful$waiting, "normal", 2), lwd=2)
#Use CV to select bandwidth
library(sm)
?hcv
?sm.options  # check the options
?sm.regression

par(mfrow=c(1,2))
hm = hcv(faithful$eruptions,faithful$waiting,display="lines")
sm.regression(faithful$eruptions, faithful$waiting, h=hm,
              xlab="duration", ylab="waiting")
hm
par(mfrow=c(1,2))
hm=hcv(exa$x, exa$y, display="lines")
sm.regression(exa$x, exa$y, h=hm, xlab="x", ylab="y")
hm
#error message for exb data since search boundary search is reached
#try Loess
?loess
# default: span = 0.75, degree = 2
par(mfrow=c(1,3))
plot(waiting ~ eruptions, faithful,col="gray", cex=0.5)
f=loess(waiting ~ eruptions, faithful)
i = order(faithful$eruptions)
lines(f$x[i],f$fitted[i], lwd=1.5, col="red")

plot(y ~ x, exa, col="gray", cex=0.5)
lines(exa$x,exa$m,lty=1)
f = loess(y ~ x,exa)
lines(f$x,f$fitted,lty=2, lwd=1.5, col="red")
f = loess(y ~ x, exa, span=0.22)
lines(f$x,f$fitted,lty=5, lwd=1.5, col="blue")

plot(y ~ x, exb, col="gray", cex=0.5)
lines(exb$x,exb$m, lty=1)
f =loess(y ~ x, exb)
lines(f$x,f$fitted,lty=2,lwd=1.5, col="red")
f = loess(y ~ x, exb,span=1)
lines(f$x,f$fitted,lty=5, lwd=1.5, col="blue")

lo.lev <- function(x1, sp){
  
  ## YOUR CODE: compute the diagonal entries of 
  ##            the smoother matrix S
  
}

onestep_CV <- function(x1, y1, sp){
  
  ## YOUR CODE: 
  ## 1) fit a loess model y1 ~ x1 with span = sp, and extract 
  ##    the corresponding residual vector
  ## 2) call lo.lev to obtain the diagonal entries of S
  ## 3) compute LOO-CV and GCV
  
  return(list(cv = cv, gcv = gcv))
}

myCV <- function(x1, y1, span){
  ## x1, y1: two vectors
  ## span: a sequence of values for "span"
  
  m = length(span)
  cv = rep(0, m)
  gcv = rep(0, m)
  for(i in 1:m){
    tmp = onestep_CV(x1, y1, span[i])
    cv[i] = tmp$cv
    gcv[i] = tmp$gcv
  }
  return(list(cv = cv, gcv = gcv))
}
