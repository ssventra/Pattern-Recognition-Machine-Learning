%% Perceptron learning rule
% Name: Ventrapragada Sai Shravani
%
% PRN:17070123120
%
% Batch:Entc(2017-21) G-3
clc;
clear all;
close all;
%% Aim:
% Implementation of AND gate using perceptron
%
%% Theory:
% Perceptron comes under a signle layer feed forward networks
% Network consists of 3 units: 1) Sensory Unit (input) 2)Associate unit (hidden) 3)Response
% unit (output) The binary activation function is used in sensory and
% associator unit. Response unit has an activation of 1, 0 or -1.Binary
% step with threshold theta is used as activation. 
%
% The perceptron is a triggering circuit has we use step function to
% trigger the inputs. It is used in the weight updation of between the
% associate and response unit. 
%
% It's application include encode databases, points of entry, monitor access data, and routinely check the consistency of the database security.
%
%% Perceptron Learning Rule for AND gate
x=[1 1 -1 -1; 1 -1 -1 1];
t=[1 -1 -1 -1];
w=[0 0];
b=0;
alpha=1;
theta=0;
count=1;
epoch=1;
while count
    count=0;
     for i=1:4
         yin=b+x(1,i)*w(1)+x(2,i)*w(2);
      if yin>theta
          y=1;
      end
if yin<=theta & yin>=-theta 
          y=0;
      end
if yin<-theta 
          y=-1;
end
      if y-t(i)
          count=1;
     for j=1:2
         w(j)=w(j)+alpha*t(i)*x(j,i);
     end
      b=b+alpha*t(i);
      end
     end
     epoch=epoch+1;
end
disp('Perceptron for ANDfunction');
disp('final weight matrix');
disp(w);
disp('Final bias');
disp(b);
disp('the training is over');
%% Deep learning tool box
X = [[0;0], [0;1],[1;0],[1;1]];  % this defines 4 input training vectors
t = [0 0 0 1];    % this defines the corresponding output for each vector
net = perceptron;
[net,tr] = train(net,X,t);
view(net)
y = net(X);      
sim(net, [1; 1]) % test the network with the input [0;0]
figure(1)
figure(2)
plotpv(X, t)     % plot the network training data
figure(3)
plotpv(X, t)
plotpc(net.iw{1,1},net.b{1})
% figure(4)
% i=imread('C:\Users\acer\Documents\4th year\Final year Project\capture.png');
% imshow(i);
