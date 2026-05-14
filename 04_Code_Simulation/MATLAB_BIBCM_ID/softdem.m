function  [Lc] = softdem(llr, La, symMatrix)

[symnum, symlen] = size(llr);
bitnum = length(La);
m = bitnum/symlen;
Lc = zeros(1, bitnum);
Infty = 1e5;
num = zeros(1,m); den = num;
A = symMatrix;
for n = 0:symlen-1
    for k = 1:m
        num(k) = -Infty;			
		den(k) = -Infty;			
    end
    
    for i = 1:symnum
		metric = llr(n*symnum + i); % channel metric for this symbol 
		   
		for k = 1:m 
			if A(i,k) == 1
				metric = metric +  La(n*m+k);
            end
        end

		for k = 1:m 
			if A(i,k) == 1
                delta1 = num(k); delta2 = metric - La(n*m+k);
                if delta1 > delta2	
		           num(k) = delta1 + log( 1 + exp(delta2 - delta1) );
                else
                   num(k) = delta2 + log( 1 + exp(delta1 - delta2) );
                end
            else
                delta1 = den(k); delta2 = metric;
                if delta1 > delta2	
		           den(k) = delta1 + log( 1 + exp(delta2 - delta1) );
                else
                   den(k) = delta2 + log( 1 + exp(delta1 - delta2) );
                end
           end
        end
    end
    for k = 1:m
		Lc(n*m+k) = num(k) - den(k);
    end
end
end