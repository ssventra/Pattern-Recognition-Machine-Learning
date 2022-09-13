%% Support Vector Machine for Alphabet recognition
% Name: Ventrapragada Sai Shravani
%
% PRN:17070123120
%
% Batch: Entc(2017-21) G-3

%% Aim:
% Implementation of Support Vector Machine for Alphabet recognition.
%
%% Theory:
% The worth of a classifier is not in how well it seperates the training
% data.
%
% SVMS try to find all such data that correctly classify the training data
% and among all such lines pick the one that has the greatest distance to
% the points closest to it.
%
% For data that isn't linearly sepereable we can project data to a space
% where it is almost linealy seperable.
%
% An important aspect of SVMs is that all the mathematical machinery that
% it uses, the exact projection of the number of dimensions doesn't show
% up. You can write all of it in terms of dot products between various data
% points represented as vectors.
%%
clc;
clear all;
close all;
%% SVM for Alphabet recognition
X = readmatrix('dataset_Trainingcsv2.csv'); %Input 24 features of A
Y = readmatrix('dataset_Character.csv'); %Alphabet a and b character as target
svmModel = fitcsvm(X,Y,'Standardize',true,'KernelFunction','RBF','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'))
err= resubLoss(svmModel);
accuracy= (1-errTrain)*100;
disp("Training accuracy: " + accuracy)

%% Conclusion: 
% In the above experiment we implement a MATLAB code for Support vector
% machine for clustering alphabet. I learnt Kernel function of Radial basis function Because support 
% vector machines employing the kernel trick do not scale well to large numbers 
% of training samples or large numbers of features in the input space, several 
% approximations to the RBF kernel (and similar kernels) have been
% introduced.
%
% We used fitcsvm to perform the SVM function, and also plotted the scatter
% plot using gscatter function. We used legend and axis and a few marker
% effects for better presentation.
