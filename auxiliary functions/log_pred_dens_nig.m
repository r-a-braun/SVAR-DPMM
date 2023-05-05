function lp = log_pred_dens_nig(y,mu,Vn,an,bn)
%-------------------------------------------------------------------------  
% DESCRIPTION:
%   This function computes the log of the predictive density of the Normal
%   Inverse Gamma (NIG) distribution evaluated at y, given the mean mu,
%   inverse variance Vn, shape parameter an, and scale parameter bn.  
% INPUTS:
%   - y: The value at which the predictive density is evaluated.
%   - mu: The mean of the NIG distribution.
%   - Vn: The inverse variance of the NIG distribution.
%   - an: The shape parameter of the NIG distribution.
%   - bn: The scale parameter of the NIG distribution. 
% OUTPUTS:
%   - lp: The log of the predictive density evaluated at y. 
nu = 2.*an;
sig2 = bn.*(1+Vn)./an; 
lp = gammaln((nu+1)./2)-gammaln(nu./2)...
    -0.5*log(nu.*pi.*sig2) ...  
    -(nu+1)./2.*log(1 + 1./nu.*((y-mu).^2./sig2));  
end

 