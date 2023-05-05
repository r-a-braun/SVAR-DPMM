function p=beta_prior(x,alpha_k,beta_k)
%------------------------------------------------------------------------- 
% DESCRIPTION:
%   This function computes the probability density values of the beta
%   distribution at the given input values x, with shape parameters alpha_k
%   and beta_k. 
% INPUTS:
%   - x: The input values at which the PDF will be evaluated.
%   - alpha_k: The shape parameter "alpha" of the beta distribution.
%   - beta_k: The shape parameter "beta" of the beta distribution. 
% OUTPUTS:
%   - p: The probability density values calculated for the given input values x.  
%------------------------------------------------------------------------- 
p=betapdf(x,alpha_k,beta_k);