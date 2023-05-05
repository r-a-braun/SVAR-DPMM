
function den = lmvtpdf(X,mu,sigma2,nu) 
%------------------------------------------------------------------------- 
% DESCRIPTION:
%   This function computes the log of the multivariate t probability density
%   function (PDF) evaluated at X, given the mean vector mu, the covariance
%   matrix sigma2, and the degrees of freedom nu. It assumes X is a row vector,
%   mu is a row vector, sigma2 is a symmetric positive definite matrix, and nu
%   is a scalar. 
% INPUTS:
%   - X: The row vector at which the PDF is evaluated.
%   - mu: The mean vector of the multivariate t distribution.
%   - sigma2: The covariance matrix of the multivariate t distribution.
%   - nu: The degrees of freedom of the multivariate t distribution. 
% OUTPUTS:
%   - den: The log of the PDF evaluated at X. 
%-------------------------------------------------------------------------  
p = length(X);
err = X-mu;
c = gammaln((nu+p)/2) - gammaln(nu/2)...
    -(p/2)*log(nu)-(p/2)*log(pi) - 1/2*log(det(sigma2));
den = c -(nu+p)/2*log( 1 + 1/nu*(err'*(sigma2\err)) );
end