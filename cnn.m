close all;
clear;

load("generated_data/second_scenario_MVDR100.mat");

% input figure
carrierFreq = 100e6;
wavelength = physconst('LightSpeed')/carrierFreq;

ula = phased.ULA('NumElements',10,'ElementSpacing',wavelength/2);
ula.Element.FrequencyRange = [90e5 110e6];

figure;
tiledlayout(2,2)
for i = 200:200:800
    nexttile

    pattern(ula,carrierFreq,-180:180,0,'Weights',wMVDRArray{i},'Type','powerdb',...
    'PropagationSpeed',physconst('LightSpeed'),'Normalize',false,...
    'CoordinateSystem','rectangular');
    axis([-100 100 -50 20]);
end

% Start...
numObservations = numel(rxSignalArray);
[m, n] = size(rxSignalArray{1});

covariances = cell(numObservations,1);
weightVecors = zeros(numObservations,size(wMVDRArray{1},1)*2);

for i = 1:numObservations
    triu_r1_1d = triu(cov(rxSignalArray{i},1), 1);
    triu_r1_1d = triu_r1_1d(find(triu_r1_1d));
    covariances{i} = triu_r1_1d'; % use complex value
    % covariances{i} = cat(2, real(triu_r1_1d), imag(triu_r1_1d));
    weightVecors(i,:) = cat(1, real(wMVDRArray{i}), imag(wMVDRArray{i}));
end


numResponses = size(weightVecors, 2);

[idxTrain,idxValidation,idxTest] = trainingPartitions(numObservations, [0.8 0.1 0.1]);

XTrain = covariances(idxTrain);
XValidation = covariances(idxValidation);
XTest = covariances(idxTest);

TTrain = weightVecors(idxTrain,:);
TValidation = weightVecors(idxValidation,:);
TTest = weightVecors(idxTest,:);

%
filterSize = 6;
numFilters = 8;

layers = [ ...
    sequenceInputLayer(1,SplitComplexInputs=true)
    convolution1dLayer(filterSize,numFilters,Padding="same")
    eluLayer
    convolution1dLayer(filterSize,2*numFilters,Padding="same")
    eluLayer
    convolution1dLayer(filterSize,4*numFilters,Padding="same")
    eluLayer
    globalAveragePooling1dLayer
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions("adam", ...
    MaxEpochs=2500, ...
    ValidationData={XValidation, TValidation}, ...
    OutputNetwork="best-validation-loss", ...
    ExecutionEnvironment="gpu", ...
    Plots="training-progress", ...
    Verbose=true);

net = trainNetwork(XTrain,TTrain,layers,options);

YTest = predict(net,XTest,SequencePaddingDirection="left");

rmse = sqrt(mean((YTest-TTest).^2));

% 1x20 to 1x10 Complex Value 
real_part = YTest(:, 1:10);
imag_part = YTest(:, 11:20);

YTest = complex(double(real_part), double(imag_part));

figure;
tiledlayout(2,2)
for i = 20:20:80
    nexttile

    pattern(ula,carrierFreq,-180:180,0,'Weights',YTest(i,:)','Type','powerdb',...
    'PropagationSpeed',physconst('LightSpeed'),'Normalize',false,...
    'CoordinateSystem','rectangular');
    axis([-100 100 -50 20]);
end
