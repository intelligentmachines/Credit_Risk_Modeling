##############loading required libraries#############

library(dplyr)
library(gmodels)
library(ggplot2)
library(corrplot)
library(oddsratio)
library(caret)
library(rpart)
library(rpart.plot)
library(pROC)
library(randomForest)
library(caret)
library(neuralnet)

#Starting with loading the data


#######EDA with Loan Data##############

summary(master)

#from the summary it was very clear that there is an outlier in the age of the lender - the max to 144. lets check that

master %>% filter(age > 100) 

master[which.max(master$age), 8] <- as.numeric(master %>% filter(age < 100) %>% summarise(max(age)))

#in order to keep our data closest to the pattern, we took the next highest value of age and replaced 144 with that 

CrossTable(master$loan_status)

#This shows us how many laons defaulted

#Exploring the relationship between loan grade and loan decision

CrossTable(y = master$loan_status, x = master$grade, prop.r = TRUE, prop.c = FALSE, prop.t = FALSE, prop.chisq = FALSE)

#Similarly looking at other categorical variables and how the impact loan decision

CrossTable(y = master$loan_status, x = master$home_ownership, prop.r = TRUE, prop.c = FALSE, prop.t = FALSE, prop.chisq = FALSE)
chisq.test(table(master$loan_status, master$home_ownership))
CrossTable(y = master$loan_status, x = master$emp_cat, prop.r = TRUE, prop.c = FALSE, prop.t = FALSE, prop.chisq = FALSE)
CrossTable(y = master$loan_status, x = master$ir_cat, prop.r = TRUE, prop.c = FALSE, prop.t = FALSE, prop.chisq = FALSE)

#Checking the distribution of the loan amount

hist(master$loan_amnt)

#distribution of loan amount, interest rate, employment lenght, annual income and age for all those who defaulted

master %>% filter(loan_status == 1) %>% ggplot(aes(loan_amnt)) + geom_histogram()
master %>% filter(loan_status == 1) %>% ggplot(aes(emp_length)) + geom_histogram()
master %>% filter(loan_status == 1, annual_inc < 80000) %>% ggplot(aes(annual_inc)) + geom_histogram()
master %>% filter(loan_status == 1) %>% ggplot(aes(int_rate)) + geom_histogram()

#this is an example where a single visualization might not be very enlightening, so might be comparing multiple graphs 
#help

master %>% ggplot(aes(loan_amnt)) + geom_histogram() + facet_wrap(~loan_status)
master %>% ggplot(aes(emp_length)) + geom_histogram() + facet_wrap(~loan_status)
master %>% ggplot(aes(int_rate)) + geom_histogram() + facet_wrap(~loan_status)

#also visualizing how the loan amount varied based on grade, home ownership and ir category

master %>% ggplot(aes(y = loan_amnt, x = grade, fill = grade)) + geom_boxplot() + facet_wrap(~loan_status)
master %>% ggplot(aes(y = loan_amnt, x = home_ownership, fill = home_ownership)) + geom_boxplot() + facet_wrap(~loan_status)
master %>% filter(ir_cat != "Missing") %>% ggplot(aes(y = loan_amnt, x = ir_cat, fill = ir_cat)) + geom_boxplot() + facet_wrap(~loan_status)

#Cleaning out outliers

par(mfrow = c(1,1))

boxplot(master$annual_inc)

master %>% filter(annual_inc < 1000000) %>% ggplot(aes(y = annual_inc, x = 1)) + geom_boxplot()

#Based on this we decide to cut off the dataset for those whose annual incomes are less than 1000000

master <- master %>% filter(annual_inc < 1000000)

summary(master)

#Imputing missing values

tokeep <- which(sapply(master,is.numeric))

corrplot(cor(na.omit(master[, tokeep])), method = "shade")

#This shows that we dont need any relational model to inpute the missing interest rate data

missing_rows <- which(is.na(master$int_rate))

master$int_rate[missing_rows] <- mean(master$int_rate, na.rm = TRUE)

summary(master)

#Similartly we can take care of the missing values in emp_length var

missing_rows <- which(is.na(master$emp_length))

master$emp_length[missing_rows] <- mean(master$emp_length, na.rm = TRUE)

summary(master)

#Finally we will replace the missing data columns with binned variables

master <- master %>% select(-ir_cat, -emp_cat)

######## Credit Risk Prediction Modeling ###############
 
#First creating train/test split (70:30 split)

random_rows <- sample(nrow(master), round(nrow(master)*0.7))
train <- master[random_rows, ]
test <- master[-random_rows, ]

#The first model will be a base model. We can argue that age, interest rate, loan amount and annual income 
# are the factors that would determine the probability of default

log_base_model <- glm(loan_status ~ age + int_rate + grade + loan_amnt + annual_inc, family = "binomial", data = train)
summary(log_base_model)

# To understand the model, the best way is to look into the odds ratio
or_base <- or_glm(train, model = log_base_model, incr = list(age = 30, loan_amnt = mean(train$loan_amnt), annual_inc = mean(train$annual_inc), int_rate = mean(train$int_rate)))

#Now we have to check if the model is accurate in predictions

pred_base <- predict(log_base_model, test, type = "response")
head(pred_base)

#what we get is the probablity of default. Now in order to be conservative, we chose 10% as the cutoff

pred_base_def <- ifelse(pred_base <= 0.1, 0, 1)

confusionMatrix(as.factor(pred_base_def), as.factor(test$loan_status))

#for us, sensitivity is the target. We need to accurately predict the default rate. lets decide on which cutoff to choose

grid <- seq(0.01, 0.3, length.out = 50)

sens_base <- rep(NA, length(grid))
acc_base <- rep(NA, length(grid))

for(i in 1:length(grid)){
  def <- ifelse(pred_base <= grid[i], 0, 1)
  cm <- confusionMatrix(as.factor(def), as.factor(test$loan_status))
  sens_base[i] <- cm[["byClass"]][["Sensitivity"]]
  acc_base[i] <- cm[["overall"]][["Accuracy"]]
}

par(mfrow = c(1,2))
plot(x = grid, y = sens_base)
plot(x = grid, y = acc_base)

#Given these details, seems like at 15% cutoff, both the sensitivity and accuracy stops marginally increasing significantly
#lets try a full model

log_full_model <- glm(loan_status ~ ., family = "binomial", train)
pred_full <- predict(log_full_model, test, type = "response")
pred_full_def <- ifelse(pred_full <= 0.15, 0, 1)

confusionMatrix(as.factor(pred_full_def), as.factor(test$loan_status))

#repeating the grid search for this mdoel

grid <- seq(0.01, 0.3, length.out = 50)

sens_full <- rep(NA, length(grid))
acc_full <- rep(NA, length(grid))

for(i in 1:length(grid)){
  def <- ifelse(pred_full <= grid[i], 0, 1)
  cm <- confusionMatrix(as.factor(def), as.factor(test$loan_status))
  sens_full[i] <- cm[["byClass"]][["Sensitivity"]]
  acc_full[i] <- cm[["overall"]][["Accuracy"]]
}

par(mfrow = c(1,2))
plot(x = grid, y = acc_base)
plot(x = grid, y = acc_full)

#Focusing on sensitivity, if we want to see which model is better

par(mfrow = c(1,2))
plot(x = grid, y = sens_base)
plot(x = grid, y = sens_full)

#Both the models are very similar and as such we will stick to the parsimonius model for our case

#Now we can try out different specification of models

model_perf <- function(x,y, pred){
  grid <- seq(x, y, length.out = 50)
  sens <- rep(NA, length(grid))
  acc <- rep(NA, length(grid))
  for(i in 1:length(grid)){
    def <- ifelse(pred <= grid[i], 0, 1)
    cm <- confusionMatrix(as.factor(def), as.factor(test$loan_status))
    sens[i] <- cm[["byClass"]][["Sensitivity"]]
    acc[i] <- cm[["overall"]][["Accuracy"]]
  }
  par(mfrow = c(1,2))
  plot(x = grid, y = sens)
  plot(x = grid, y = acc)
}

logit_model <- glm(loan_status ~ ., family = binomial(link = "logit"), data = train)
probit_model <- glm(loan_status ~ ., family = binomial(link = "probit"), data = train)
cloglog_model <- glm(loan_status ~ ., family = binomial(link = "cloglog"), data = train)

pred_logit <- predict(logit_model, test, type = "response")
pred_probit <- predict(probit_model, test, type = "response")
pred_cloglog <- predict(cloglog_model, test, type = "response")

model_perf(0.01, 0.3, pred_logit)
model_perf(0.01, 0.3, pred_probit)
model_perf(0.01, 0.3, pred_cloglog)

#We can see that almost no change in performance or accuracy

###################Tree based modeling ###############

#first we need to see how unbalanced our training set is

table(train$loan_status)

train_defs <- train %>% filter(loan_status == 1)
train_nondefs <- train %>% filter(loan_status == 0)

sample_rows <- sample(nrow(train_defs), nrow(train_nondefs)/2, replace = TRUE)
train_defs <- train_defs[sample_rows,]

under_train <- rbind(train_defs, train_nondefs)

table(under_train$loan_status)

#shuffling the whole data set for avoiding patters

under_train <- under_train[sample(nrow(under_train)), ]

#fitting the rpart model

under_tree <- rpart(loan_status ~ ., method = "class", under_train)

par(mfrow = c(1,1))

rpart.plot(under_tree)

#another way to fix the unbalancing problem is through setting prior probs - telling rpart to create equal representation

prior_tree <- rpart(loan_status ~ ., method = "class", train, parms = list(prior = c(0.5,0.5)))

rpart.plot(prior_tree)

#another way is to create a loss matrix. For a bank, marking a defaulter to be non-defaulter is more disparaging
#so we tell the rpart that to penalize a missclassified defaulter to be 10 times more

loss_tree <- rpart(loan_status ~ ., method = "class", train, parms = list(loss = matrix(c(0, 10, 1, 0), ncol = 2)))

rpart.plot(loss_tree)

#finally another way to handling unbalanced data is to introduce weighs

wts <- ifelse(train$loan_status == 0, 1, 10)

wts_tree <- rpart(loan_status ~ ., method = "class", train, weights = wts)

rpart.plot(wts_tree)

#to chose the tree which will be best at predicting defaults

pred_under_tree <- predict(under_tree, test, type = "class")
pred_prior_tree <- predict(prior_tree, test, type = "class")
pred_loss_tree <- predict(loss_tree, test, type = "class")
pred_wts_tree <- predict(wts_tree, test, type = "class")

confusionMatrix(pred_under_tree, as.factor(test$loan_status))
confusionMatrix(pred_prior_tree, as.factor(test$loan_status))
confusionMatrix(pred_loss_tree, as.factor(test$loan_status))
confusionMatrix(pred_wts_tree, as.factor(test$loan_status))



################# Choosing the final Model #####################

#first lets predict the pd based on logit model
pd_logit <- predict(logit_model, test, type = "response")
head(pd_logit)

# suppose the bank wants to reject 20% of the applications based on pd. THe cutoff point will be
(cutoff <- quantile(pd_logit, 0.8, na.rm = TRUE))


#getting the def preds

def_logit <- ifelse(pd_logit <= cutoff, 0, 1)

#calculating the proportion of people who were missclassified as not going to default

outcome_table <- data.frame(actual = test$loan_status, pred = def_logit)

false_positives <- outcome_table %>% filter(pred == 0)

sum(false_positives$actual)/nrow(false_positives)


#creating a function of getting the bad rate

bad_rate <- function(pd){
 
  acc_rate <- seq(1,0,length.out = 50)
  br <- rep(NA, length(acc_rate))
  cutoff <- rep(NA, length(acc_rate))
  
  for(i in 1:length(acc_rate)){
    cutoff[i] <- quantile(pd, acc_rate[i], na.rm = TRUE)
    def <- ifelse(pd <= cutoff[i], 0, 1)
    out <- as.numeric(test$loan_status[def == 0])
    br[i] <- mean(out, na.rm = TRUE)
  }
  
  result_df <- data.frame(acc_rate, cutoff, br)
  return(result_df)
}

br_logit <- bad_rate(pd_logit)

par(mfrow = c(1,2))

plot(x = br_logit$acc_rate, y = br_logit$br, type = "l")

#comparing the bad rate for two different models

pd_tree <- predict(prior_tree, test, type = "prob")[,2]

br_tree <- bad_rate(pd_tree)

plot(x = br_tree$acc_rate, y = br_tree$br, type = "l")

#another mean of comparing models is through ROC Curve and the calculation of AUC

par(mfrow = c(1,1))

roc_logit <- roc(test$loan_status, pd_logit)
roc_tree <- roc(test$loan_status, pd_tree)

plot(roc_logit, col = "blue")
lines(roc_tree, col = "red")

auc(roc_logit)
auc(roc_tree)

#Based on the auc of the models, the logit model is best at predicting the cred default

########### Black box methods - random forests and neural networks###########

#random forest

under_train$loan_status <- as.factor(under_train$loan_status)
test$loan_status <- as.factor(test$loan_status)

rf <- randomForest(loan_status ~ ., data = under_train)

plot(rf)

pred_rf <- predict(rf, test, type = "prob")[,2]

model_perf(0.05,0.5,pred_rf)

confusionMatrix(predict(rf, test), test$loan_status)

br_rf <- bad_rate(pred_rf)

plot(x = br_rf$acc_rate, y = br_rf$br, type = "l")

roc_rf <- roc(test$loan_status, pred_rf)

plot(roc_rf)

auc(roc_rf)

#neural networks

#in order to propery train a neural network, we need to one hot encode the categorical data, and standardize the numeric
# data

train_numeric <- train %>% select(loan_amnt, annual_inc, age, int_rate, emp_length)

preproc <- preProcess(train_numeric, method = c("center", "scale"))

train_norm <- predict(preproc, train_numeric)

# now we create onehot encoding for the categorical variable

train_categ <- train %>% select(grade, home_ownership)

dmy <- dummyVars(" ~ .", train_categ)

train_dummy <- data.frame(predict(dmy, train_categ))

#finally creating the train dataset for nn training

train_nn <- cbind(train$loan_status, train_dummy, train_norm)
names(train_nn)[1] <- "loan_status"

#repeating the same step for test

test_numeric <- test %>% select(loan_amnt, annual_inc, age, int_rate, emp_length)
preproc <- preProcess(test_numeric, method = c("center", "scale"))
test_norm <- predict(preproc, test_numeric)

test_categ <- test %>% select(grade, home_ownership)
dmy <- dummyVars(" ~ .", test_categ)
test_dummy <- data.frame(predict(dmy, test_categ))

test_nn <- cbind(test$loan_status, test_dummy, test_norm)
names(test_nn)[1] <- "loan_status"

#now we are ready to carry out the neural network modelling

nn <- neuralnet(loan_status ~ ., train_nn, hidden = 3, act.fct = "logistic", linear.output = FALSE)

plot(nn)

pred_nn <- as.numeric(compute(nn, test_nn)[[2]])

model_perf(0.05, 0.3, pred_nn)

br_nn <- bad_rate(pred_nn)

plot(x = br_nn$acc_rate, y = br_nn$br, type = "l")

pred_nn_binary <- ifelse(pred_nn < 0.15, 0, 1)

confusionMatrix(test$loan_status, as.factor(pred_nn_binary))

roc_nn <- roc(test$loan_status, pred_nn_binary)

plot(roc_nn)

auc(roc_nn)
