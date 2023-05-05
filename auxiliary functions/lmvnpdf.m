
function den = lmvnpdf(X,mu,sigma2)
%-------------------------------------------------------------------------
% den = lmvnpdf(X, mu, sigma2)
%
% DESCRIPTION:
%   This function computes the log of the multivariate normal probability density
%   function (PDF) evaluated at X, given the mean vector mu and the covariance
%   matrix sigma2. It assumes X is a row vector, mu is a row vector, and sigma2
%   is a symmetric positive definite matrix.
%
% INPUTS:
%   - X: The row vector at which the PDF is evaluated.
%   - mu: The mean vector of the multivariate normal distribution.
%   - sigma2: The covariance matrix of the multivariate normal distribution.
%
% OUTPUTS:
%   - den: The log of the PDF evaluated at X. 
%------------------------------------------------------------------------- 
N = length(X);
err = X-mu;
kernel = -err'*(sparse(sigma2)\err)/2;
den = (-N/2)*log(2*pi) - 1/2*log(det(sigma2))+kernel;
end