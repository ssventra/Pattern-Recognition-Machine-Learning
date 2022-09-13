%% Lab report-1 
% Name: Ventrapragada Sai Shravani
% PRN:17070123120
% Batch: G3(17-21)
clc;
clear all;
close all;
%%
% A naïve Bayes classifier assumes the independence of the predictors
% within each class. A probabilty distribution within each class. 
% We can determine a new observation at a given location.
img= imread("C:\Users\acer\Documents\4th year\Machine Learning\naive_bayer.png");
imshow(img)
%% Loads and formats the data.
heartData = readtable("dataset_exp2_heartDiseaseData.csv");
heartData.HeartDisease = categorical(heartData.HeartDisease);
% Partitions the data into training and test sets.
pt = cvpartition(heartData.HeartDisease,"HoldOut",0.3);
hdTrain = heartData(training(pt),:);
hdTest = heartData(test(pt),:);
%% Naive Bayes function
% You can set the "DistributionNames" property using an array of distribution names.
% Repmat function to create a string array with repeated values.
dists = [repmat("kernel",1,11) repmat("mvmn",1,10)];
mdl = fitcnb(hdTrain,"HeartDisease","DistributionNames",dists)
% mdl = fitcnb(hdTrain,"HeartDisease","DistributionNames","kernel")
isLabels1 = resubPredict(mdl);
ConfusionMat1 = confusionchart(hdTest,isLabels1);
%% Calculating the loss for the training and testsets
% Shows us the error in the prediction of the data in the training and the 
% testing set. Also shows the loss in data for training and testing test.
errTrain = resubLoss(mdl);
errTest = resubLoss(mdl);
accuracy_train= (1-errTrain)*100;
accuracy_test= (1-errTest)*100;
disp("Training accuracy: " + accuracy_train)
disp("Test accuracy: " + accuracy_test)
