library(dplyr)
library(tidyverse)

dev.off()

##Setting directory in order to load data file.
setwd("C:/Users/dawso/Desktop")

##Loading data file.
a_raw <- read.csv("Absent.csv")

View(a_raw)


#Clean Data
a <- (a_raw %>% 
            select(Social.drinker, Reason.for.absence, Month.of.absence, Day.of.the.week, 
                   Seasons, Disciplinary.failure))

attach(a)

names(a)

##View data structure
str(a)

##General view of NA
is.na(a)

##View of how many NAs in each column
colSums(is.na(a))

library(ggplot2)

##Data Exploration
Seasons_count <- a %>%
  count(Seasons)

Seasons_count %>%
  mutate(Seasons = reorder(Seasons, n)) %>%
  ggplot(aes(x = Seasons, y = n)) +
  geom_col()+
  coord_flip()

Month.of.absense_count <- a %>%
  count(Month.of.absence)

Month.of.absense_count %>%
  mutate(Month.of.absence = reorder(Month.of.absence, n)) %>%
  ggplot(aes(x = Month.of.absence, y = n)) +
  geom_col()+
  coord_flip()

Day.of.the.week_count <- a %>%
  count(Day.of.the.week)

Day.of.the.week_count %>%
  mutate(Day.of.the.week = reorder(Day.of.the.week, n)) %>%
  ggplot(aes(x = Day.of.the.week, y = n)) +
  geom_col()+
  coord_flip()

Reason.for.absence_count <- a %>%
  count(Reason.for.absence)

Reason.for.absence_count %>%
  mutate(Reason.for.absence = reorder(Reason.for.absence, n)) %>%
  ggplot(aes(x = Reason.for.absence, y = n)) +
  geom_col()+
  coord_flip()

Social.drinker_count <- a %>%
  count(Social.drinker)

Social.drinker_count %>%
  mutate(Social.drinker = reorder(Social.drinker, n)) %>%
  ggplot(aes(x = Social.drinker, y = n)) +
  geom_col()+
  coord_flip()

summary(a)

##Regression 
a3 <- glm(formula = Social.drinker ~., data = a, family = binomial)

a5 <- glm(formula = Social.drinker ~ Day.of.the.week, data = a, family = binomial)

a7 <- glm(formula = Social.drinker ~ Reason.for.absence + Month.of.absence + Seasons +
            Disciplinary.failure, family = binomial, )

summary(a3)
summary(a5)

r3 <-a3$deviance;r3
sum(residuals(a3,type="pearson")^2)
n3 <-a3$null.deviance;n3
df3 <-a3$df.residual;df3

n5 <-a5$null.deviance;n5
r5 <-a5$deviance;r5
df5 <-a5$df.residual;df5
g2 <-r5-r3;g2
df <- df5-df3;df

qchisq(.95, df=4)

##Reject null hypothesis. There is enough evidence at alpha = 0.05 to conclude that Days.of.the.week 
##is not needed for this model.



a <- (a_raw %>% 
        select(Disciplinary.failure, Reason.for.absence, Month.of.absence,
               Seasons, Social.drinker))


##Export dataset to txt file.
write.table(a, file = "Absent1", sep=",", row.names = FALSE) 

##Machine learning

##KNN

#Normalization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

an <- as.data.frame(lapply(a[,2:5], normalize))

set.seed(12345)

##Creating a function
index <- function(data, size = 0.8, train = TRUE) {
  n_row = nrow(data)
  total_row = size * n_row
  train_sample <- 1: total_row
  if (train == TRUE) {
    return (data[train_sample, ])
  } else {
    return (data[-train_sample, ])
  }
}

##Create train and test sets
train <- index(a, 0.8, train = TRUE)
test <- index(a, 0.8, train = FALSE)

ad <- sample(1:nrow(an),size=nrow(an)*0.8,replace = FALSE) #random selection of 80% data.

#Creating seperate dataframe for 'Creditability' feature which is our target.
train_labels <- a[ad,1]
test_labels <- a[-ad,1]

install.packages('class')
library(class)

NROW(train_labels)
sqrt(592)
##One 'k' value will be 24 and the other model will be 25
knn.24 <- knn(train=train, test=test, cl=train_labels, k=24)
knn.25 <- knn(train=train, test=test, cl=train_labels, k=25)

#Calculate the proportion of correct classification for k = 24, 25
ACC.24 <- 100 * sum(test_labels == knn.24)/NROW(test_labels);ACC.24
ACC.25 <- 100 * sum(test_labels == knn.25)/NROW(test_labels);ACC.25

# Check prediction against actual value in tabular form for k=24
table(knn.24 ,test_labels)
table(knn.25 ,test_labels)

install.packages('caret')
install.packages('e1071')
library(caret)
library(e1071)

#Confusion matrix
confusionMatrix(table(knn.24, test_labels))

#Optimization
i=1
k.optm=1
for (i in 1:28){
  knn.mod <- knn(train=train, test=test, cl=train_labels, k=i)
  k.optm[i] <- 100 * sum(test_labels == knn.mod)/NROW(test_labels)
  k=i
  cat(k,'=',k.optm[i],'
')
}
#k = 12 (an even integer) is the model with the best accuracy, so I'll be using 13 (an odd integer) instead.
knn.13 <- knn(train=train, test=test, cl=train_labels, k=13);knn.13
ACC.13 <- 100 * sum(test_labels == knn.13)/NROW(test_labels);ACC.13
table(knn.13,test_labels)
confusionMatrix(table(knn.13, test_labels))

install.packages('gmodels')
library(gmodels)

CrossTable(x=test_labels, y=knn.13, prop.chisq = FALSE)

##Decision Tree
library(rpart) ##Used for decision tree formula
library(rpart.plot) ##Used to plot decision tree

set.seed(12345)

##Creating a function
index <- function(data, size = 0.8, train = TRUE) {
  n_row = nrow(data)
  total_row = size * n_row
  train_sample <- 1: total_row
  if (train == TRUE) {
    return (data[train_sample, ])
  } else {
    return (data[-train_sample, ])
  }
}

##Create train and test sets
train <- index(a, 0.8, train = TRUE)
test <- index(a, 0.8, train = FALSE)

##Review the split of the train and test data
dim(train)
dim(test)

##Marginal tables for train and test
prop.table(table(train$Social.drinker))
prop.table(table(test$Social.drinker))

##Create decision tree model and plot
fit <- rpart(Social.drinker ~ ., data = train, method ='class')
rpart.plot(fit, extra = 106)

pred<-predict(fit, test, type ='class')

##Create confusion table
tab <- table(test$Social.drinker, pred)
tab

##Obtain accuracy for model
accuracy <- sum(diag(tab))/sum(tab)

print(accuracy)

