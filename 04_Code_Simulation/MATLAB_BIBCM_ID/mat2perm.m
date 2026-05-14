function [alp] = mat2perm(H,bps)
matrix = H';
[Ns,Ncw] =size(matrix);
m = bps; Ncb = Ns*m; n = Ncb/Ncw; 
alp = (1:Ncb);
T = zeros(Ns,Ncw);
for i = 1:Ncw 
    for j = 1:Ns        
        if matrix(j,i) > 0
            T(j,i) = T(j,i) + matrix(j,i); %matrix(j,i) may be > 1
            alp((j-1)*m + sum(T(j,:))) = (i-1)*n + sum(T(:,i));                
        end
    end    
end
