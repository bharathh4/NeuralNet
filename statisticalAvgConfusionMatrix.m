function avgConfusion = statisticalAvgConfusionMatrix(num_Iterations,num_layers,percentage_training,percentage_validation)

load SubSetNormalizedFeaturesSet2.mat
y=SubSetNormalizedFeaturesSet2;
clear SubSetNormalizedFeaturesSet2;

% load NormalizedFeaturesSet2.mat
% y=NormalizedFeaturesSet2;
% clear NormalizedFeaturesSet2;

num_features=size(y,2)-1

[class1, class2 ,class3]=prepareData(y);

% class1=y(1:7660,1:num_features);
% class2=y(7660:7660*2,1:num_features);
% class3=y(7660*2:7660*3,1:num_features);

nMat=[];
for j=1:num_Iterations
percentage_testing=(100-percentage_validation-percentage_training);
[train_samples_class1, test_samples_class1]=selectSamples(class1,percentage_training,percentage_testing);


percentage_testing=(100-percentage_validation-percentage_training);
[train_samples_class2, test_samples_class2]=selectSamples(class2,percentage_training,percentage_testing);



percentage_testing=(100-percentage_validation-percentage_training);
[train_samples_class3, test_samples_class3]=selectSamples(class3,percentage_training,percentage_testing);


% Note that feature 1 is in the columns,feature 16 is in the columns 

cTrain1=[train_samples_class1] ;
cTrain2=[train_samples_class2] ;
cTrain3=[train_samples_class3] ;

actualTrainlabels=[(ones(length(train_samples_class1),1)) ; (2*ones(length(train_samples_class2),1));(3*ones(length(train_samples_class3),1))];
actualTrainlabels=labelConversion(actualTrainlabels,1);

trainFeatures=[cTrain1;cTrain2;cTrain3];

inputs=transpose(trainFeatures(:,1:num_features));
targets=transpose(actualTrainlabels);

% Create a Pattern Recognition Network
hiddenLayerSize = num_layers;
net = patternnet(hiddenLayerSize);


% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = (percentage_training)/100;
net.divideParam.valRatio = (percentage_validation)/100;
net.divideParam.testRatio = (100-percentage_validation-percentage_training)/100;
net.trainParam.showWindow=0;

% Train the Network
[net,tr] = train(net,inputs,targets);


% Test the Network
c1=[test_samples_class1] ;
c2=[test_samples_class2] ;
c3=[test_samples_class3] ;

actualtestlabels=[(ones(length(test_samples_class1),1)) ; (2*ones(length(test_samples_class2),1));(3*ones(length(test_samples_class3),1))];

testFeatures=[c1;c2;c3];


outputs = net(transpose(testFeatures(:,1:num_features)));
actual= transpose(actualtestlabels);
size(actual)

%transpose(testFeatures(:,17));

% actual=transpose(targets);
%predicted=transpose(round(outputs));



predicted=labelConversion(round(outputs),0);
size(predicted)

confusionMatrix=confusionmat(predicted,actual);
normalMat=(1/length(c1))*confusionMatrix;
normalMat=transpose(normalMat);
nMat(:,:,j)=normalMat;
end

temp=[0 0 0;0 0 0;0 0 0];

for k=1:num_Iterations
 temp=temp+nMat(:,:,k) ;
end
avgConfusion=temp/num_Iterations;


end