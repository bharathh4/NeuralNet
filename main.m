% testing function
clear all
clc
close all



load SubSetNormalizedFeaturesSet2.mat
y=SubSetNormalizedFeaturesSet2;
clear SubSetNormalizedFeaturesSet2;

class1=y(1:7660,1:16);
class2=y(7660:7660*2,1:16);
class3=y(7660*2:7660*3,1:16);

percentage_training=70;
percentage_testing=30;
[train_samples_class1 test_samples_class1]=selectSamples(class1,percentage_training,percentage_testing);

percentage_training=70;
percentage_testing=30;
[train_samples_class2 test_samples_class2]=selectSamples(class2,percentage_training,percentage_testing);


percentage_training=70;
percentage_testing=30;
[train_samples_class3 test_samples_class3]=selectSamples(class3,percentage_training,percentage_testing);


% Note that feature 1 is in the columns,feature 16 is in the columns 

cTrain1=[train_samples_class1 ones(length(train_samples_class1),1)];
cTrain2=[train_samples_class2 2*ones(length(train_samples_class2),1)];
cTrain3=[train_samples_class3 3*ones(length(train_samples_class3),1)];

trainFeatures=[cTrain1;cTrain2;cTrain3];

inputs=transpose(trainFeatures(:,1:16));
targets=transpose(trainFeatures(:,17));

% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);


% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;


% Train the Network
[net,tr] = train(net,inputs,targets);

% Test the Network
outputs = net(inputs);

actual=transpose(targets);
predicted=transpose(round(outputs));

confusionMatrix=confusionmat(predicted,actual);
normalMat=(1/length(cTrain1))*confusionMatrix;

% normalMat =
% 
%     0.9884    0.0011    0.0065
%     0.0097    0.9631    0.2007
%     0.0019    0.0360    0.7930
