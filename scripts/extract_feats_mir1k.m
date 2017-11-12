function [vocal_feats, background_feats] = extract_feats_mir1k(path_to_audio)

audio_files = dir(strcat(path_to_audio,'/*.wav')); 
win_size = 1024;
hop_size = 512;
sr = 16000;
vocal_feats = [];
background_feats = [];
for i=1:length(audio_files)
    [~, filename,~] = fileparts(audio_files(i).name);
    [x, fs] = audioread(strcat(path_to_audio,'/',audio_files(i).name));
    x = resample(x, sr, fs); %Downsample
    vocal_feats = [vocal_feats, log(abs(spectrogram(x(:,2), hann(win_size), win_size-hop_size, win_size, fs)))]; % Vocal
    background_feats = [background_feats, log(abs(spectrogram(x(:,1), hann(win_size), win_size-hop_size, win_size, fs)))]; %Background
end


end