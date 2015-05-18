library(Boruta)
require(xgboost)
require(methods)
library(ggplot2)
library(readr)
library(Rtsne)

print("Preparing Data")

train <- read.csv('../Input/train.csv',header=TRUE,stringsAsFactors = F)
test <- read.csv('../Input/test.csv',header=TRUE,stringsAsFactors = F)

train$ID <- NULL

train = train[,-1]
test = test[,-1]

#shuffle
train <- train[sample(nrow(train)),]

y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
x = log(1 + x)
x = scale(x)
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

print("Running Cross Validation")

# Set necessary parameter
#param <- list("objective" = "multi:softprob",
#              "eval_metric" = "mlogloss",
#              "num_class" = 9,
#              "gamma" = 1,
#              "nthread" = 6,
#              "eta" = .4,
#              "max_depth" = 25,
#              "min_child_weight" = 3,
#              "subsample" = .5,
#              "colsample_bytree" = .9,
#              "max_delta_step" = 3)
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 16,
              "bst:eta" = .3,
              "bst:max_depth" = 30,
              "lambda" = 1,
              "lambda_bias" = 0,
              "gamma" = 1,
              "alpha" = .8,
              "min_child_weight" = 4,
              "subsample" = .9,
              "colsample_bytree" = .9)

cv.nround = 91
bst.cv = xgb.cv(param=param, data = x[trind,], label = y, nfold = 5, nrounds=cv.nround)

print("Training the model")
# Train the model
nround = 91
bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround, verbose = 2)

# Get the feature real names
#names <- dimnames(myData[1:n.train, ])[[2]]

# Compute feature importance matrix
#importance_matrix <- xgb.importance(names, model = bst)

# Nice graph
#xgb.plot.importance(importance_matrix)
#print("Plotting feature tree")
#xgb.plot.tree(feature_names = names, model = bst)

print("Making prediction")
# Make prediction
pred = predict(bst,x[teind,])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

print("Storing Output")
# Output submission
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='../Output/otto-classifier.csv', quote=FALSE,row.names=FALSE)
