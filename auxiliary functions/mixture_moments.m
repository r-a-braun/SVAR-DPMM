function [M, V, S, K] = mixture_moments(mus,sigmas,mom3,mom4,W) 
%------------------------------------------------------------------------- 
% DESCRIPTION:
%   This function computes the mean, variance, skewness, and kurtosis of any
%   k mixture distribution with weights W. All inputs are kx1 vectors.
%
% INPUTS:
%   - mus: The mean values of the mixture components.
%   - sigmas: The variances of the mixture components.
%   - mom3: The third central moments of the mixture components.
%   - mom4: The fourth central moments of the mixture components.
%   - W: The weights of the mixture components.
%
% OUTPUTS:
%   - M: The mean of the mixture distribution.
%   - V: The variance of the mixture distribution.
%   - S: The skewness of the mixture distribution.
%   - K: The kurtosis of the mixture distribution. 
%------------------------------------------------------------------------- 
n = length(W);
M = W'*mus ;
V = W'*(sigmas + mus.^2 - M.^2) ;

%% Skewness and Kurtosis
ECMm = zeros(2,1);
CentralMoments0to4  = [ones(n,1), zeros(n,1), sigmas, mom3, mom4 ];
a = 1;
for j = [3,4]
    temp = zeros(n,1);
    for k = 0:j
        temp = temp + nchoosek(j,k).*(mus-M).^(j-k).*CentralMoments0to4(:,k+1);
    end
    ECMm(a) =  W'*temp;
    a = a + 1;
end
S = ECMm(1)/(sqrt(V).^3);
K = ECMm(2)./(V.^2);  
end






 