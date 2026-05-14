function [b] = modulate16qam(x, S)

m = log2(length(S));
[usernum, bitnum] = size(x);
symnum = bitnum/m;

b = zeros(usernum, symnum);
bipower = [2.^[m-1:-1:0]];
for user = 1:usernum
    y = reshape(x(user,:),m,symnum); 
    for cnt = 1:symnum
       b(user,cnt) = S( bipower*y(:,cnt) + 1 );       
    end
end