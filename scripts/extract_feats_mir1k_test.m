
path_to_audio = '/Users/RupakVignesh/Desktop/fall17/7100/MIR-1K/Test/';
path_to_labels = '/Users/RupakVignesh/Desktop/fall17/7100/MIR-1K/vocal-nonvocalLabel/';
op_feat_path = '/Users/RupakVignesh/Desktop/fall17/7100/MIR-1K/Test_feats';

audio_files = dir(strcat(path_to_audio,'/*.wav')); 
win_size = 640;
hop_size = 320;
fft_size = 1024;
sr = 16000;

for i=1:length(audio_files)
    [~, filename,~] = fileparts(audio_files(i).name);
    [x, fs] = audioread(strcat(path_to_audio,'/',audio_files(i).name));
    x = resample(x, sr, fs); %Downsample
    
    % Read Grount truth
    fileID = fopen(strcat(path_to_labels,filename,'.vocal'),'r');
    formatSpec = '%f';
    GT = fscanf(fileID,formatSpec);
    fclose(fileID);
    
    vocal_wav = x(:,2)/max(x(:,2));
    vocals = abs(spectrogram(vocal_wav, hann(win_size), win_size-hop_size, fft_size, fs));
    vocals = vocals/max(max(vocals));
    
    back_wav = x(:,1)/max(x(:,1));
    background = abs(spectrogram(back_wav, hann(win_size), win_size-hop_size, fft_size, fs));
    background = background/max(max(background));

    feats = vocals + background;
    csv_file_name = strcat(op_feat_path, '/', filename,'.csv');
    csvwrite(csv_file_name, [feats; GT']');
end
