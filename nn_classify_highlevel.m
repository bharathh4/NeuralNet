function predicted=nn_classify_highlevel(num_layers,percentage_training,percentage_validation,feature)

load SubSetNormalizedFeaturesSet2.mat
y=SubSetNormalizedFeaturesSet2;
clear SubSetNormalizedFeaturesSet2;

class1=y(1:7660,1:16);
class2=y(7660:7660*2,1:16);
class3=y(7660*2:7660*3,1:16);


percentage_testing=100-percentage_validation-percentage_training;
[train_samples_class1 test_samples_class1]=selectSamples(class1,percentage_training,percentage_testing);


percentage_testing=100-percentage_validation-percentage_training;
[train_samples_class2 test_samples_class2]=selectSamples(class2,percentage_training,percentage_testing);


percentage_testing=100-percentage_validation-percentage_training;
[train_samples_class3 test_samples_class3]=selectSamples(class3,percentage_training,percentage_testing);


% Note that feature 1 is in the columns,feature 16 is in the columns 

cTrain1=[train_samples_class1 ones(length(train_samples_class1),1)];
cTrain2=[train_samples_class2 2*ones(length(train_samples_class2),1)];
cTrain3=[train_samples_class3 3*ones(length(train_samples_class3),1)];

trainFeatures=[cTrain1;cTrain2;cTrain3];

inputs=transpose(trainFeatures(:,1:16));
targets=transpose(trainFeatures(:,17));

% Create a Pattern Recognition Network
hiddenLayerSize = num_layers;
net = patternnet(hiddenLayerSize);


% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = percentage_training/100;
net.divideParam.valRatio = percentage_validation/100;
net.divideParam.testRatio = (100-percentage_validation-percentage_training)/100;


% Train the Network
[net,tr] = train(net,inputs,targets);

% Test the Network
outputs = net(transpose(feature));

%actual=transpose(targets);
predicted=transpose(round(outputs));

end