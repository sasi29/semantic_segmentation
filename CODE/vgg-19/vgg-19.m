outputFolder = fullfile(pwd, 'CamVid');
imgDir = fullfile(outputFolder,'imagesResized');
imds = imageDatastore(imgDir);
pic_num = 30;
I_raw = readimage(imds, pic_num);
I = histeq(I_raw); 
figure
imshow(I_raw)
drawnow
title('Raw image');
figure
imshow(I)
drawnow
title('Image with equalized histogram');
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
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);
cmap = camvidColorMap;
C = readimage(pxds, pic_num);
B = labeloverlay(I,C,'ColorMap',cmap);
figure
imshow(B) 
drawnow
pixelLabelColorbar(cmap,classes);
tbl = countEachLabel(pxds);
frequency = tbl.PixelCount/sum(tbl.PixelCount);
figure
bar(1:numel(classes),frequency)
xticks(1:numel(classes)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')

%[imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionCamVidData(imds,pxds,labelIDs)
            % Partition CamVid data by randomly selecting 60% of the data for training. The
            % rest is used for testing.
            
            % Set initial random state for example reproducibility.
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
   imageSize = [360 480 3];
numClasses = numel(classes);
lgraph = segnetLayers(imageSize,numClasses,'vgg19');
lgraph.Layers
% Get the imageFreq using the data from the countEachLabel function
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
% The higher the frequency of a class the smaller the classWeight
classWeights = median(imageFreq) ./ imageFreq;
pxLayer = pixelClassificationLayer('Name','labels','ClassNames', tbl.Name, 'ClassWeights', classWeights);
% Plot the 91-Layer lgraph 
fig2=figure('Position', [100, 100, 800, 600]);
subplot(1,2,1)
plot(lgraph);
xlim([2.862 3.200])
ylim([-0.9 10.9])
axis off 
title('Initial last 9 Layers Graph')
% Remove last layer of and add the new one we created. 
lgraph = removeLayers(lgraph, {'pixelLabels'});
lgraph = addLayers(lgraph, pxLayer);
% Connect the newly created layer with the graph. 
lgraph = connectLayers(lgraph, 'softmax','labels');
lgraph.Layers
subplot(1,2,2)
plot(lgraph);
xlim([2.862 3.200])
ylim([-0.9 10.9])
axis off 
title(' Modified last 9 Layers Graph')
options = trainingOptions('sgdm', ... % This is the solver's name; sgdm: stochastic gradient descent with momentum
    'Momentum', 0.9, ...              % Contribution of the gradient step from the previous iteration to the current iteration of the training; 0 means no contribution from the previous step, whereas a value of 1 means maximal contribution from the previous step.
    'InitialLearnRate', 0.001, ...     % low rate will give long training times and quick rate will give suboptimal results 
    'L2Regularization', 0.0005, ...   % Weight decay - This term helps in avoiding overfitting
    'MaxEpochs', 20,...              % An iteration is one step taken in the gradient descent algorithm towards minimizing the loss function using a mini batch. An epoch is the full pass of the training algorithm over the entire training set.
    'MiniBatchSize', 1, ...           % A mini-batch is a subset of the training set that is used to evaluate the gradient of the loss function and update the weights.
    'Shuffle', 'every-epoch', ...     % Shuffle the training data before each training epoch and shuffle the validation data before each network validation.
    'Verbose', false,...        
    'Plots','training-progress');  
augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation', [-10 10], 'RandYTranslation',[-10 10]);
          

        datasource = pixelLabelImageSource(imdsTrain,pxdsTrain,...
   'DataAugmentation',augmenter); 
 
[net, info] = trainNetwork(datasource,lgraph,options);
save('PreTrainedCnn.mat','net','info','options');
disp('NN trained');
I = read(imdsTest);
C = semanticseg(I, net);
   
expectedResult = read(pxdsTest);
expected = uint8(expectedResult);
predicted = uint8(C);
% Compare differences between images - Image Processing toolbox
imshowpair(expected, predicted, 'montage')
title('Ground Truth labels vs Predicted labels')
pic_num = 30;
I = readimage(imds, pic_num);
Ib = readimage(pxds, pic_num);
IB = labeloverlay(I, Ib, 'Colormap', cmap, 'Transparency',0.8);
C = semanticseg(I, net);
CB = labeloverlay(I, C, 'Colormap', cmap, 'Transparency',0.8);
figure
imshowpair(IB,CB,'montage')
pixelLabelColorbar(cmap, classes);
title('Ground Truth vs Predicted')
I = read(imdsTest);
C = semanticseg(I, net);
B = labeloverlay(I, C, 'Colormap', cmap, 'Transparency',0.8);
figure
imshow(B)
pixelLabelColorbar(cmap, classes);
expectedResult = read(pxdsTest);
expected = uint8(expectedResult);
predicted = uint8(C);
% Compare differences between images - Image Processing toolbox
imshowpair(expected, predicted, 'montage')
title('Ground Truth labels vs Predicted labels')
iou = jaccard(C, expectedResult);
table(classes,iou)
  pxdsResults = semanticseg(imdsTest,net,'Verbose',true);
 metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',true); 