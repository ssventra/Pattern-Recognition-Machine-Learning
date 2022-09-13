%% Experiment 5: Backpropogation for Alphabet 
% Name: Ventrapragada Sai Shravani
%
% PRN:17070123120
%
% Batch:Entc(2017-21) G-3
%% Aim:
% Implementation of Alphabet gate using back propagation
%
%% Theory:
% Back propagation comes under a multiple layer feed forward networks.Back propogation is a supervisedlearning network, it consists of input, hidden and output layers.A feedback is provided to the hidden and output layer and input layer does not contain bias since thereis no feedback present.This  algorithm  can  be  explained  in  2  phases:  Feed-forward  phase,  back-propogation  phase  of  learningwhere signals are sent in reverse direction.Gradient descent: Is an optimisation algorithm used to find the value of parameter/ coefficient of a function,that minimises a cost function.
%% Back Propogation for Alphabet
clc;
clear all;
close all;
INPUT = transpose(readmatrix('dataset_Trainingcsv2.csv')); %Input 24 features of A
TARGET = transpose(readmatrix('dataset_Character.csv')); %Alphabet a and b character as target
net = feedforwardnet(2,'traingd'); % 2 defines the number of hidden layers
% Modifying the defined parameters
net.trainParam.show =50;  % The result is shown at every 50th iteration (epoch)
net.trainParam.lr = 0.05; % Learning rate used in some gradient schemes
net.trainParam.epochs = 300; % Max number of iterations
net.trainParam.goal = 1e-5; % Error tolerance; stopping criterion
% The goal of "training" a neural network is to find the right weights that
%correctly predict what the output of an given input is. There are two fundamental
%processes that go into training a neural network.
[net,tr] = train(net,INPUT,TARGET); %Training the network
%tr will contain all the details about the training
view(net)
y = net(INPUT);
testX = INPUT(:,tr.testInd);
testT = TARGET(:,tr.testInd);
testY = net(testX);
YPredicted = net(INPUT);
YPredicted(:,1:10)
figure(1)
figure(2)
%i=imread('C:\Users\acer\Documents\4th year\Machine Learning\MATLAB ML\Capture.png');
%imshow(i);
figure(3)
plotconfusion(TARGET,YPredicted)% plot the network training data
figure(4)
plotperform(tr) %performance plot
figure(5)
plottrainstate(tr) %trainstate plot
%% Conclusion
% In this code we observed graphs of performance, calculated and updated biases, also found final weightsafter upgradation. Also ploted gradient descent graph in figure 5. And simulated a test data and obtaineddesired output.We plotted confusion matrix which shows 4 A samples getting misclassified and B samples getting 100%correctly classified. We also took input as transpose because it is not a feedforward network.