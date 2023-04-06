varies = -90:0.1:90;

rxSignalArray = cell(numel(varies),1);
wMVDRArray = cell(numel(varies),1);

h = waitbar(0, 'Processing data...'); % 초기화

for i = 1:1:numel(varies)
    for j = 0:100 
        [rxSignal, wMVDR] = generateCNNdata(varies(i));
        rxSignalArray{i} = rxSignal;
        wMVDRArray{i} = wMVDR;
    end
    waitbar(i / numel(varies), h, sprintf('Processing data... %d%%', floor(i/numel(varies)*100))); % 진행 상황 업데이트
end
close(h); % 종료

save('generated_data/second_scenario.mat','rxSignalArray','wMVDRArray')

function [rxSignal,wMVDR] = generateCNNdata(angle)
    t = 0:0.001:0.399;              % Time, sampling frequency is 1kHz, K=400
    s = zeros(size(t));  
    s = s(:);                       % Signal in column vector
    s(201:205) = s(201:205) + 1;    % Define the pulse
    % figure;
    % plot(t,s);
    % title('Pulse');xlabel('Time (s)');ylabel('Amplitude (V)');
    
    carrierFreq = 100e6;
    wavelength = physconst('LightSpeed')/carrierFreq;
    
    ula = phased.ULA('NumElements',10,'ElementSpacing',wavelength/2);
    ula.Element.FrequencyRange = [90e5 110e6];
    
    desiredSignalAngle = [angle; 0];
    x = collectPlaneWave(ula,s,desiredSignalAngle,carrierFreq);
    
    % Create and reset a local random number generator so the result is the
    % same every time.
    rs = RandStream.create('mt19937ar','Seed', 'Shuffle');
    % rs = RandStream.create('mt19937ar','Seed', 2008);
    noisePwr = 39.810717;   % noise power, 16dB ISR 
    

    noisePwr1 = 0.1;
    noise = sqrt(noisePwr1/2)*(randn(rs,size(x))+1i*randn(rs,size(x)));
    
    nSamp = length(t);
    s1 = noisePwr*randn(rs,nSamp,1);
    s2 = noisePwr*randn(rs,nSamp,1);
    s3 = noisePwr*randn(rs,nSamp,1);
    s4 = noisePwr*randn(rs,nSamp,1);
    s5 = noisePwr*randn(rs,nSamp,1);

    % % fft
    % y = s1;
    % Fs = 1000;
    % N = length(y);
    % Y = fft(y); 
    % f = (0:N-1)*(Fs/N);
    % 
    % figure;
    % plot(f,abs(Y));
    % xlabel('Frequency (Hz)');
    % ylabel('Magnitude');
    % title('Frequency spectrum of noisy signal');


    % interference
    narrowbandInterference = collectPlaneWave(ula,[s1, s2],[-20 20; 0 0], carrierFreq);
    widebandInterference = collectPlaneWave(ula,[s3 s4 s5],[-40 40 60; 0 0 0], carrierFreq);
    
    
    rxInt = narrowbandInterference + widebandInterference + noise;        % total interference + noise
    rxSignal = x + rxInt;                % total received Signal
    
    
    % Define the MVDR beamformer
    mvdrbeamformer = phased.MVDRBeamformer('SensorArray',ula,...
        'Direction',desiredSignalAngle,'OperatingFrequency',carrierFreq,...
        'WeightsOutputPort',true);
    
    mvdrbeamformer.TrainingInputPort = true;
    [yMVDR, wMVDR] = mvdrbeamformer(rxSignal,rxInt);

    % figure;
    % plot(t,abs(yMVDR)); axis tight;
    % title('Output of MVDR Beamformer With Presence of Interference');
    % xlabel('Time (s)');ylabel('Magnitude (V)');
    % % 
    % figure;
    % pattern(ula,carrierFreq,-180:180,0,'Weights',wMVDR,'Type','powerdb',...
    %     'PropagationSpeed',physconst('LightSpeed'),'Normalize',false,...
    %     'CoordinateSystem','rectangular');
    % axis([-100 100 -50 20]);
    % hold on
end