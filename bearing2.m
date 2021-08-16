datpath=fullfile('Bearing');
imageSize=[128 128 3];
imds=imageDatastore(datpath,'IncludeSubfolders',true,'LabelSource','foldernames');
table=countEachLabel(imds)
numTrainingFiles = 17;
[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainingFiles,'randomize');
aimds1=augmentedImageDatastore(imageSize,imdsTrain);
aimds2=augmentedImageDatastore(imageSize,imdsTest);
options=trainingOptions('sgdm','ValidationData',aimds1,'ValidationFrequency',1,'plots','training-progress');
net = trainNetwork(aimds1,part,options);
pred=classify(net,aimds2);
accuracy = mean(pred == imdsTest.Labels)
idx = randperm(numel(imdsTest.Files),4);
figure(1)
for i = 1:4
 subplot(2,2,i)
 I = readimage(imdsTest,idx(i));
 imshow(I)
end
figure(2)
plotconfusion(imdsTest.Labels,pred)
fruit=imread('C:\Users\Juvith\Desktop\Fruit\Grapes');
is=[227 227 1];
aim1=augmentedImageDatastore(is,fruit);
label=classify(net,aim1);
sprintf('The loaded image belongs to %s class', label)
