function [est_vocal, est_inst, fs] = reconstruct_waveform(est_vocal_spec, est_inst_spec, orig_sig, win_size, hop_size, fft_size, fs)

num_blocks = size(est_vocal_spec,2);
hann_win = hann(win_size);
orig_X = spectrogram(orig_sig, hann_win, win_size-hop_size, fft_size, fs);
orig_X_phase = angle(orig_X);

rec_vocal_spec = bsxfun(@times, est_vocal_spec, exp(1i .* orig_X_phase));
rec_inst_spec = bsxfun(@times, est_inst_spec, exp(1i .* orig_X_phase));

len_x = (num_blocks-1)*hop_size + fft_size;
est_vocals = zeros(len_x, 1);
for i=1:num_blocks
    temp_sig = ifft([rec_vocal_spec(:,i); conj(flipud(rec_vocal_spec(2:end-1,i)))]);
    temp_sig = real(temp_sig);
    est_vocals((i-1)*hop_size+1:(i-1)*hop_size+fft_size) = est_vocals((i-1)*hop_size+1:(i-1)*hop_size+fft_size) + hann(fft_size).*temp_sig;
end

est_inst = zeros(len_x, 1);
for i=1:num_blocks
    temp_sig = ifft([rec_vocal_spec(:,i); conj(flipud(rec_vocal_spec(2:end-1,i)))]);
    temp_sig = real(temp_sig);
    est_inst((i-1)*hop_size+1:(i-1)*hop_size+fft_size) = est_inst((i-1)*hop_size+1:(i-1)*hop_size+fft_size) + hann(fft_size).*temp_sig;
end

end