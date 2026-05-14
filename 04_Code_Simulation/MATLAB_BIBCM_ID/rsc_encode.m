function [y] = rsc_encode(g, x)

% encodes a block of data x (0/1)with a recursive systematic
% convolutional code with generator vectors in g, and
% returns the output in y (0/1).

% determine the constraint length (K), memory (m), and rate (1/n)
% and number of information bits.
[n,K] = size(g);
m = K - 1;
dataLen = length(x); 

% initialize the state vector
state = zeros(1,m);

% generate the codeword
for i = 1:dataLen
   d_k = x(1,i);
   a_k = rem( g(1,:)*[d_k state]', 2 );
   [output_bits, state] = encode_bit(g, a_k, state);
   % since systematic, first output is input bit
   output_bits(1,1) = d_k;
   y(n*(i-1)+1:n*i) = output_bits;
end