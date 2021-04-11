clc;
clear all;
close all;
outputFolder = fullfile(pwd, 'CamVid');
imgDir = fullfile(outputFolder,'imagesResized');
imds = imageDatastore(imgDir);
I = readimage(imds, 1);
I = histeq (I);
classes = [
    "Sky"
    "Building"
    "Pole"
    "Road"
    "Pavement"
    "Tree"
    "SignSymbol"
    "Fence"
    "Car"
    "Pedestrian"
    "Bicyclist"
    ];
labelIDs = camvidPixelLabelIDs();
labelDir = fullfile(outputFolder,'labelsResized');
pxds = pixelLabelDatastore (labelDir, classes, labelIDs);
C = readimage(pxds, 1);

cmap = camvidColorMap;
B = labeloverlay(I,C,'ColorMap',cmap);

tbl = countEachLabel(pxds);
frequency = tbl.PixelCount/sum(tbl.PixelCount);

figure
bar(1:numel(classes),frequency)
xticks(1:numel(classes))
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')
rng(0);
            numFiles = numel(imds.Files);
            % Returns a row vector containing a random permutation of the integers from 1 to n inclusive.
            shuffledIndices = randperm(numFiles);
            
            % Use 60% of the images for training.
            N = round(0.60 * numFiles);
            trainingIdx = shuffledIndices(1:N);
            
            % Use the rest for testing.
            testIdx = shuffledIndices(N+1:end);
            
            % Create image datastores for training and test.
            trainingImages = imds.Files(trainingIdx);
            testImages = imds.Files(testIdx);
            imdsTrain = imageDatastore(trainingImages);
            imdsTest = imageDatastore(testImages);
            
            % Extract class and label IDs info
            classes = pxds.ClassNames;
%             labelIDs = 1:numel(pxds.ClassNames);
            
            % Create pixel label datastores for training and test.
            trainingLabels = pxds.Files(trainingIdx);
            testLabels = pxds.Files(testIdx);
            pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
            pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
        
   numTrainingImages = numel(imdsTrain.Files); 
   numTestingImages = numel(imdsTest.Files);
   imageSize = [300 300];
numClasses = 11;
lgraph = fcnLayers(imageSize, numClasses, 'Type','8s');

% Replace first conv layer with a padding and conv layer to work around
% the bug where the convolution layer cannot have a pad size of 100.
convLayer = lgraph.Layers(2);

c = convolution2dLayer(convLayer.FilterSize,convLayer.NumFilters,...
    'NumChannels', convLayer.NumChannels,...
    'Padding',[0 0 0 0],'Name',convLayer.Name);

c.Weights = convLayer.Weights;
c.Bias = convLayer.Bias;

lgraph = removeLayers(lgraph,convLayer.Name);

% Create padding layer using custom layer attached to this work around.
padding = paddingLayer(100);
padding.Name = 'padding';
newLayers = [
    padding
    c
    ];

lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph,'input','padding');
lgraph = connectLayers(lgraph,'conv1_1','relu1_1');

figure;
plot(lgraph);
options = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 1e-3, ...
    'L2Regularization', 0.0005, ...
    'MaxEpochs',1, ...
    'MiniBatchSize', 4, ...
    'Shuffle', 'every-epoch', ...
    'Plots','training-progress', ...
    'VerboseFrequency', 100);
increase = imageDataAugmenter ( 'RandXReflection' , true, ... 
    'RandXTranslation' , [-10 10], 'RandYTranslation' , [- 10 10]);
datasource = pixelLabelImageSource (imdsTrain, pxdsTrain, ... 
    'DataAugmentation' , increase);
[net, info] = trainNetwork(datasource,lgraph,options);
idx = 147;
I = readimage(imdsTest,idx);
C = semanticseg(I, net);

% imshow????
B = labeloverlay(I, C, 'Colormap', cmap, 'Transparency',0.4);
figure, imshowpair(I, B, 'montage')
pixelLabelColorbar(cmap, classes);
expectedResult = readimage(pxdsTest,idx);
actual = uint8(C);
expected = uint8(expectedResult);
imshowpair(actual, expected);
iou = jaccard(C, expectedResult);
table(classes,iou);
pxdsResults = semanticseg(imdsTest,net,'WriteLocation',tempdir,'Verbose',false);
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);
metrics.DataSetMetrics
metrics.ClassMetrics
