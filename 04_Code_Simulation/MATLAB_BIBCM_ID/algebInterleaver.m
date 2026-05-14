function alp = algebInterleaver(pr,add,Np,m,n,type)

%%Description
% creating a generalized block interleaver
% Np: the length of the interleaver
% n : the codeword length = codelen
% m : the modulation degree = bps
% type: 0 = overall, 1 = inline
% use  [matrix] = connectMatrix(alp,n,m) to create the connection matrix
%%
M = Np/n; N = Np/m;
alp = (1:Np); 
if type == 0
    J = n; I = M;
elseif type == 1
    J = m; I = N;
end

for j = 1:J
    for i = 1:I
        if type == 0
            %alp((i-1)*n + (j-1) + 1) = (rem(pr(j)*(i-1) + add(j),I)) + (j-1)*I + 1; 
            alp(rem(pr(j)*(i-1) + add(j),I) + (j-1)*I + 1) = (i-1)*n + (j-1) + 1 ;
        elseif type == 1
            alp(j + i*m) = (j-1)*I + rem(pr(j)*i + add(j),I) + 1;
        end
    end
end
            

    