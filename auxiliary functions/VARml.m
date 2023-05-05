function [Aml, Sigma_u, mu, uhat, AValpha, AVsigma] = VARml(yraw,p)
%-------------------------------------------------------------------------
% [Aml, Sigma_u, mu, uhat, AValpha, AVsigma] = VARml(yraw, p)
%
% DESCRIPTION:
%   This function computes the maximum likelihood (ML) estimates of the
%   VAR(p) model with an intercept term.
%
% INPUTS:
%   - yraw: The raw data of size (TxK), where T is the number of time points
%     and K is the number of variables.
%   - p: The VAR lag order.
%
% OUTPUTS:
%   - Aml: The ML estimates of the AR parameters.
%   - Sigma_u: The ML estimate of the covariance matrix of the residuals.
%   - mu: The mean of the data.
%   - uhat: The residuals.
%   - AValpha: The asymptotic variance-covariance matrix of the AR parameters.
%   - AVsigma: The asymptotic variance-covariance matrix of the residuals.
%
% AUTHOR:
%   Robin Braun
%   robin.andreas.braun@gmail.com
%   Bank of England
%
% DATE:
%   04/05/2023
%
%-------------------------------------------------------------------------

[TpP, K] = size(yraw); 
T = TpP - p;
y = yraw(p+1:end,:)';   
mu = mean(y,2);
y = y - mu;  
rhs = lagmatrix(yraw-mu',1:p); 
X = rhs(p+1:end,:)'; 
Aml = (y*X')/(X*X');
uhat = (y-Aml*X)';
Sigma_u = 1/T*(uhat')*uhat; 
if nargout > 3
    AValpha = kron( inv(X*X'/T) , Sigma_u );
    D = duplication(K); 
    Dk = (D'*D)\D';
    AVsigma = 2*Dk*kron(Sigma_u,Sigma_u)*Dk';
end

 

 
function d = duplication(n)
% Author: Thomas P Minka (tpminka@media.mit.edu)
% duplication(n)
% Returns Magnus and Neudecker's duplication matrix of size n
a = tril(ones(n));
i = find(a);
a(i) = 1:length(i);
a = a + tril(a,-1)';
j = vec(a);

m = n*(n+1)/2;
d = zeros(n*n,m);
for r = 1:size(d,1)
    d(r, j(r)) = 1;
end
d = sparse(d);