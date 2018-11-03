##########################################
#   PRATICAL MACHINE LEARNING COURSE 
#   "Prediction Assignment Writeup"
##########################################

#cleaning workspace
rm(list=ls())

# useful libraries
library(caret)
library(ggplot2)


## loading data
TrainDataSet <- read.csv("pml-training.csv")
TestDataSet <- read.csv("pml-testing.csv")

## Cleaning data to select predictors
# Non predictors data names are set to NA. Non NA datas only will be selected

Varselect <- names(TrainDataSet) 

#Identification variables with NA
Varselect[grepl("NA",TrainDataSet[1,])]<-"NA"

#removing empty colums and facors variables except "classe"
useless_data=c("X","raw_timestamp","cvtd","new_window","num_window",
               "kurtosis","skewness",
               "max_yaw_forearm","min_yaw_forearm","amplitude_yaw_forearm","total_accel_forearm",
               "amplitude_yaw_belt", "max_yaw_belt","min_yaw_belt",
               "amplitude_yaw_dumbbell","max_yaw_dumbbell","min_yaw_dumbbell" )

for (useless in useless_data)
{
  ids <- grep(useless, names(TrainDataSet))
  Varselect[ids] <- "NA"
}

# Selecting predictors
TrainFilt<- TrainDataSet[,!grepl("NA",Varselect)]
TrainFilt$user_name<- as.factor(TrainDataSet$user_name)

set.seed(33854) #random seed

## Splitting into test an Train set
inData <- createDataPartition(y=TrainFilt$classe,p=0.6, list=FALSE)
training <- TrainFilt[inData,]
testing <- TrainFilt[-inData,]

## Machine Learning Processing 

# Preprocessing of train and test data set
PreProcessTrain <-preProcess(training, method=c("center","scale"))
trainPP <- predict(PreProcessTrain,training)

PreProcessTest<-preProcess(testing, method=c("center","scale"))
testPP <- predict(PreProcessTest,testing)

# Optimisation of ML : using parallel core processing
library(parallel)
library(doSNOW)
Clust <- makeCluster(detectCores()-1)
registerDoParallel(Clust)

#Fitting model
modelFit <- train(classe~., model="rf", data=trainPP, trainControl=trainControl(allowParallel=TRUE, classProbs=TRUE) )

# Model Info
varImp(modelFit)

stopCluster(Clust)

# Prediction on training datas (preprocessed) 
Prediction <- predict(modelFit,trainPP)
confusionMatrix(Prediction, training$classe)$overall[1]

# Prediction on testing datas  (preprocessed) 
PredictionTest <- predict(modelFit,testPP)
confusionMatrix(PredictionTest,testing$classe)$overall[1]


## Prediction final values
# selection predictors
FinalTestFilt<- TestDataSet[,!grepl("NA",Varselect)]
PreProcessFinalTest<-preProcess(FinalTestFilt, method=c("center","scale"))
FinalTestPP <- predict(PreProcessFinalTest,FinalTestFilt)

PredictionFinal <- predict(modelFit,FinalTestPP)
PredictionFinal