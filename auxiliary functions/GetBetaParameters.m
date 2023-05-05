function [alpha,beta]=GetBetaParameters(meanB,stdB) 
%------------------------------------------------------------------------- 
% DESCRIPTION:
%   This function calculates the parameters alpha and beta for a Beta
%   distribution such that the mode and the standard deviation of the
%   distribution are equal to modeB and stdB, respectively.
%
% INPUTS:
%   - meanB: The desired mean of the Beta distribution.
%   - stdB: The desired standard deviation of the Beta distribution.
%
% OUTPUTS:
%   - alpha: The calculated alpha parameter of the Beta distribution.
%   - beta: The calculated beta parameter of the Beta distribution. 
%------------------------------------------------------------------------- 
varB = stdB^2;
alpha = ( (1-meanB)./varB - 1./meanB)*meanB^2;
beta = alpha*(1./meanB - 1);
 