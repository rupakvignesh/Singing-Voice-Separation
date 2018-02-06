function [Y_clean] = spectral_sub(Y)
%
% Y - (fft_size/2 +1, num_frames) dimension
%
    spec_noise = Y(:,1);
    spec_sub = bsxfun(@minus, Y,spec_noise);
    Y_clean = abs(max(spec_sub, 0.1*(Y)));
end