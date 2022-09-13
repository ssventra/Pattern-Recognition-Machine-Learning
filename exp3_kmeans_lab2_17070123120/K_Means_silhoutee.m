clc;
clear all;
close all;
figure(1);
meas = readmatrix('C:\Users\acer\Documents\4th year\Machine Learning\17070123120_Shravani_Ventrapragada_Div_B\exp6_bp_alphabet\dataset_Trainingcsv2.csv'); %Input 24 features of A
[cidx2,cmeans2] = kmeans(meas,3,'dist','sqeuclidean');
[cidx3,cmeans3,sumd3] = kmeans(meas,3,'replicates',5,'display','final');
sum(sumd3)
[silh3,h] = silhouette(meas,cidx3,'sqeuclidean');
grid on
% Alphabet characterisation Using Hierarchical Clustering
eucD = pdist(meas,'euclidean');
clustTreeEuc = linkage(eucD,'average');
figure(2);
cophenet(clustTreeEuc,eucD)
[h,nodes] = dendrogram(clustTreeEuc,0);
h_gca = gca;
h_gca.TickDir = 'out';
h_gca.TickLength = [.002 0];
h_gca.XTickLabel = [];