%Compute Results
%Take the estimated mask and overlay it with the input spectra. 
%Reconstruct the vocals and use them in ...
%...BSS_Eval

win_size = 640;
hop_size = 320;
fft_size = 1024;
sr = 16000;

mask_path = '/Users/RupakVignesh/Desktop/fall17/7100/Singing-Voice-Separation/experiments/expt15/test_out/';
test_audio_path = '/Users/RupakVignesh/Desktop/fall17/7100/MIR-1K/Test/';

output_files = dir(strcat(mask_path, '/*.csv'));
test_audio_files = dir(strcat(test_audio_path, '/*.wav'));

N = length(output_files)/2;
SDR = zeros(2, N);
SIR = zeros(2, N);
SAR = zeros(2, N);
GNSDR = zeros(2,1);
GSIR = zeros(2,1);
GSAR = zeros(2,1);
total_L = 0;

for i=1:N
    [~, filename, ~] = fileparts(test_audio_files(i).name);
    disp(filename);
    
    est_vocal_spectra = abs(csvread(strcat(mask_path,'vocal_', filename,'.csv')));
    est_inst_spectra = abs(csvread(strcat(mask_path,'background_', filename,'.csv')));
    
    est_vocal_spectra = est_vocal_spectra(:,1:513)';
    est_inst_spectra = est_inst_spectra(:,1:513)';
    
    [orig_sig, fs] = audioread(strcat(test_audio_path, filename, '.wav'));
    [rec_vocals, rec_inst, fs] = reconstruct_waveform(est_vocal_spectra, est_inst_spectra, mean(orig_sig,2), win_size, hop_size, fft_size, fs);
    L = size(orig_sig,1);
    
    rec_vocals = rec_vocals/max(abs(rec_vocals));
    rec_inst = rec_inst/max(abs(rec_inst));
    orig_sig(:,1) = orig_sig(:,1)/max(abs(orig_sig(:,1)));
    orig_sig(:,2) = orig_sig(:,2)/max(abs(orig_sig(:,2)));
    
    mix_signal = mean(orig_sig,2);
    mix_sources = [mix_signal/max(abs(mix_signal)), mix_signal/max(abs(mix_signal))]';
    estimated_sources = [rec_inst(1:L), rec_vocals(1:L)]';
    
    [SDR(:,i), SIR(:,i), SAR(:,i)] = bss_eval_sources(estimated_sources, orig_sig');
    [SDR_MIXED, ~, ~] = bss_eval_sources(mix_sources, orig_sig');
    
    NSDR = SDR(:,i) - SDR_MIXED;
    GNSDR = GNSDR + L*NSDR;
    GSIR = GSIR + L*SIR(:,i);
    GSAR = GSAR + L*SAR(:,i);
    
    total_L = total_L + L;
end

GNSDR = GNSDR/total_L;
GSAR = GSAR/total_L;
GSIR = GSIR/total_L;

