function [num_of_4, num_of_6, num_of_8] = girth8(H)
% tinh so vong 4, so vong 6 va so vong 8

E1 = (H*H');
E1 = E1 - diag(diag(E1));

E2 = E1*E1;
E2 = (E2 - diag(diag(E2)));

num_of_4 = sum(sum(E1.*(E1-1)))/4;
num_of_6 = sum(sum( (E2.*E1) ))/6;
num_of_8 = sum(sum(E2.*(E2-1)))/8;
