function Uhat = multidcov_steadyICA(S,alpha)
% Computes the symmetric multivariate distance covariance
% Function transcribed from Steady ICA package
% https://rdrr.io/cran/steadyICA/man/multidcov.html
if nargin == 1
    alpha = 1;
end
[n,d] = size(S);  
 
U = zeros(n,d);
for i = 1:d  
     data_sorted = sort(S(:,i));
     [~, U(:,i)] = ismember(S(:,i),data_sorted); 
end
U = U/n;  
Ihat = 0;
for i = 1:d-1
    Ihat = Ihat + dcovustat(U(:,i),U(:,[i+1:d]),alpha);
end
Uhat = n*Ihat;
end
% Wrapper for c function
function f = dcovustat(x,y,alpha) 
    x = x/length(x);
    y = y/length(y); 
    f = dcovustatC(x,y,alpha);
end

