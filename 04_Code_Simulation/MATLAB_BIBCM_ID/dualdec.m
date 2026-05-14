function vv = dualdec(x,H)

[rows,cols] = size(H);
temp = de2bi((1:2^rows-1),rows,'left-msb');
C = rem(temp*H, 2);
p = (1.0-exp(x))./(1.0+exp(x)); 

E = eye(cols);
vv = zeros(1,cols);
v2 = 1.000000000001;

for zz = 1:size(C,1)
    giatricot = find(C(zz,:));
    tich = 1;
    for kk = 1:length(giatricot)
       tich = tich*p(giatricot(kk));
    end
    v2 = v2 + tich;
end 

for jj = 1:cols
    v1 = p(jj);
    for zz = 1:size(C,1)
        giatricot = find( rem(C(zz,:)+ E(jj,:),2) );
        tich = 1;
        for kk = 1:length(giatricot)
           tich = tich*p(giatricot(kk));
        end
        v1 = v1 + tich;
    end 
 
    vv(jj) = log(v2^2-v1^2)- log((v2+v1)^2);
end


end


