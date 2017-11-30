%Compute Results
%Take the estimated mask and overlay it with the input spectra. 
%Reconstruct the vocals and use them in ...
%...BSS_Eval

win_size = 640;
hop_size = 320;
sr = 16000;

mask_path = '/Users/RupakVignesh/Desktop/fall17/7100/Singing-Voice-Separation/experiments/expt7/test_out/';
test_feat_path = '/Users/RupakVignesh/Desktop/fall17/7100/MIR-1K/Test_feats/';
test_audio_path = '/Users/RupakVignesh/Desktop/fall17/7100/MIR-1K/Test/';

output_files = dir(strcat(mask_path, '/*.csv'));
test_feat_files = dir(strcat(test_feat_path, '/*.csv'));

N = length(output_files);
for i=1:N
    [~, filename, ~] = fileparts(output_files(i).name);
    estimated_mask = (csvread(strcat(mask_path,filename,'.csv')));
    input_poly_spectra = (csvread(strcat(test_feat_path, filename,'.csv')));
    estimated_vocal_spectra = estimated_mask .* input_poly_spectra;
    estimated_vocal_spectra = estimated_vocal_spectra(:,1:513)';
    [orig_sig, fs] = audioread(strcat(test_audio_path, filename, '.wav'));
    rec_vocals = reconstruct_waveform(estimated_vocal_spectra, mean(orig_sig,2), win_size, hop_size, fs);
    [SDR, SIR, SAR] = bss_eval(orig_sig(:,2), rec_vocals);
    
    
    
end

