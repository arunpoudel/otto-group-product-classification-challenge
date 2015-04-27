require(xgboost)
require(methods)

train = read.csv('../Input/train.csv',header=TRUE,stringsAsFactors = F)
test = read.csv('../Input/test.csv',header=TRUE,stringsAsFactors = F)
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
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

# Set necessary parameter
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
              "min_child_weight" = 3,
              "subsample" = .9,
              "colsample_bytree" = .9)

# Run Cross Valication
cv.nround = 91
bst.cv = xgb.cv(param=param, data = x[trind,], label = y,
                nfold = 8, nrounds=cv.nround)

# Train the model
nround = 91
bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround, verbose = 2)

# Make prediction
pred = predict(bst,x[teind,], verbose = TRUE)
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

# Output submission
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='../Output/otto-group-product-classification-challenge.csv', quote=FALSE,row.names=FALSE)