function [x,fs] = reconstruct_waveform(est_mag_spec, orig_sig, win_size, hop_size, fs)

num_blocks = size(est_mag_spec,2);
hann_win = hann(win_size);
orig_X = spectrogram(orig_sig, hann_win, win_size-hop_size, win_size, fs);
orig_X_phase = angle(orig_X);

rec_spec = bsxfun(@times, est_mag_spec, exp(1i .* orig_X_phase));

len_x = (num_blocks-1)*hop_size + win_size;
x = zeros(len_x, 1);
for i=1:num_blocks
    temp_sig = ifft([rec_spec(:,i); conj(flipud(rec_spec(2:end-1,i)))]);
    temp_sig = real(temp_sig);
    x((i-1)*hop_size+1:(i-1)*hop_size+win_size) = x((i-1)*hop_size+1:(i-1)*hop_size+win_size) + temp_sig;
end

end