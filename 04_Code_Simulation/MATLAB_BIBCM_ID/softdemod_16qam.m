function [llr_n] = softdemod_16qam(rx,N0,S)

M = length(S); 
frameLen = length(rx);

llr_n = zeros(M,frameLen);
for cnt = 1:frameLen
    for k = 1:M
        llr_n(k,cnt) = -(rx(cnt) - S(k))*conj(rx(cnt) - S(k));
    end
end
llr_n = llr_n./N0;