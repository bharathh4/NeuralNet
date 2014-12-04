%clc
clear all
close all

load SubSetNormalizedFeaturesSet2.mat
y=SubSetNormalizedFeaturesSet2;
clear SubSetNormalizedFeaturesSet2;

class1=y(1:7660,1:16);
class2=y(7660:7660*2,1:16);
class3=y(7660*2:7660*3,1:16);

%Set number of times to re-randomize and calculate confusion matrix and
% average. Beware large values can take a long time to compute.
numIterations=2; 

%Specify amount of training data/search space
percentage_training=90;
%Specify amount of validation data
percentage_validation=10;
%Specify number of hidden layers
num_layers=5;

%Switch on the profiler for timing information
%profile on
% Plug a feature vector from class 1
label=nn_classify_highlevel(num_layers,percentage_training,percentage_validation,class1(200,1:16)) % Should give 1 when we plug the 200th observation of class 1

% % Plug a feature vector from class 2
% label=nn_classify_highlevel(num_layers,percentage_training,percentage_validation,class2(200,1:16))  % Should give 2 when we plug the 200th observation of class 2
% 
% % Plug a feature vector from class 3
% label=nn_classify_highlevel(num_layers,percentage_training,percentage_validation,class3(200,1:16))  % Should give 3 when we plug the 200th observation of class 3

% Check profiler for timing information of the function
% profile viewer
% p = profile('info');
% profsave(p,'profile_results')
% profile off
%%

%% Calculate average confusion matrix.For a quick test set numIterations=2 and num_Layers=5 
clc
clear all
close all
%Set number of times to re-randomize and calculate confusion matrix and
% average. Beware large values can take a long time to compute.
numIterations=1; 

%Specify amount of training data/search space
percentage_training=70;
%Specify amount of validation data
percentage_validation=15;
%Specify number of hidden layers
num_layers=5;

avgConfusion = statisticalAvgConfusionMatrix(numIterations,num_layers,percentage_training,percentage_validation)

