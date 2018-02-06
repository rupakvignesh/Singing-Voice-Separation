function [vocal_feats, background_feats] = extract_feats_mir1k


path_to_audio = '/Users/RupakVignesh/Desktop/fall17/7100/MIR-1K/Train/';
path_to_labels = '/Users/RupakVignesh/Desktop/fall17/7100/MIR-1K/vocal-nonvocalLabel/';

audio_files = dir(strcat(path_to_audio,'/*.wav')); 
win_size = 640;
hop_size = 320;
fft_size = 1024;
sr = 16000;
vocal_feats = [];
background_feats = [];
mix_feats = [];
for i=1:length(audio_files)
    [~, filename,~] = fileparts(audio_files(i).name);
    [x, fs] = audioread(strcat(path_to_audio,'/',audio_files(i).name));
    x = resample(x, sr, fs); %Downsample
    
    % Read Grount truth
%     fileID = fopen(strcat(path_to_labels,filename,'.vocal'),'r');
%     formatSpec = '%f';
%     GT = fscanf(fileID,formatSpec);
%     fclose(fileID);
    
    vocal_wav = x(:,2);
    vocals = abs(spectrogram(vocal_wav, hann(win_size), win_size-hop_size, fft_size, fs));
    GT = zeros(size(vocals,2),1);   %dummy GT
    vocal_feats = [vocal_feats, [vocals; GT']];
    
    back_wav = x(:,1);
    background = abs(spectrogram(back_wav, hann(win_size), win_size-hop_size, fft_size, fs));
    background_feats = [background_feats, [background; GT']];
    

end

end