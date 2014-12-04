% This is a neural network created to classifying the data I created.
% inputs  = [0 1 0 1; 0 0 1 1]; % feature vector as a column [1;2]
% targets = [0 1 1 0]; % Two classes .Class per column [road babble babble road]
% 

clear all
clc
close all

%Class 1
class_1_x1=[1+1*randn(1,1000) 3+1*randn(1,1000)];
class_1_x2=[3+1*randn(1,1000) 5+1*randn(1,1000)];
Class1_X=[class_1_x1 ;class_1_x2 ];
Class1_X=transpose(Class1_X);%Each feature in a column


%Class 2
class_2_x1=[5+1*randn(1,1000) 7+1*randn(1,1000)];
class_2_x2=[7+1*randn(1,1000) 9+1*randn(1,1000)];
Class2_X=[class_2_x1 ;class_2_x2 ];
Class2_X=transpose(Class2_X);%Each feature in a column


%Class 3
class_3_x1=[6+1*randn(1,1000) 6.5+1*randn(1,1000)];
class_3_x2=[1+1*randn(1,1000) 3+1*randn(1,1000)];
Class3_X=[class_3_x1 ;class_3_x2 ];
Class3_X=transpose(Class3_X);%Each feature in a column


X= [Class1_X ;Class2_X ;Class3_X];
Y=[ones(2000,1) ;2*ones(2000,1);3*ones(2000,1)];

inputs=transpose(X);
targets=transpose(Y);


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
normalMat=(1/2000)*confusionMatrix;

% errors = gsubtract(targets,outputs);
% performance = perform(net,targets,outputs);

% View the Network
% view(net)

% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
%figure, plotconfusion(targets,outputs)
% figure, ploterrhist(errors)
