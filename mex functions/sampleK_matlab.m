function [Kal,SumX,SumX2,n_ii] = sampleK_matlab(Kal,estar,psr,m,tau,s,S,alpha) 
% DESCRIPTION:
%   This function updates the cluster allocation and computes the necessary
%   statistics for a given set of observations, cluster indicators, and prior
%   parameters. It uses a MATLAB implementation of the sampling algorithm. 
% INPUTS:
%   - Kal: A column vector of length T representing the cluster indicators.
%   - estar: A column vector of length T representing the shock values.
%   - psr: A column vector of length T representing uniform (0,1) random numbers.
%   - s: The parameter s in the Normal inverse Gamma Base Distribution.
%   - S: The parameter S in the Normal inverse Gamma Base Distribution.
%   - tau: The parameter tau in the Normal inverse Gamma Base Distribution.
%   - m: The parameter m in the Normal inverse Gamma Base Distribution.
%   - alpha: The parameter alpha in the DPMM
%
% OUTPUTS:
%   - Kal: Updated column vector of length T representing the cluster indicators.
%   - SumX: A 1xk vector for each cluster's sums of shocks
%   - SumX2: A 1xk vector for each cluster's sums of squared shocks
%   - n_ii: A 1xk vector of cluster sizes 
% AUTHOR:
%   Robin Braun
%   robin.braun@bankofengland.co.uk
%   Bank of England
%
% DATE:
%   04/05/2023
%-------------------------------------------------------------------------

%% First, lets count the number of unique elements and compute T
k = length(unique(Kal));
T = length(estar);   

%% Next, initialize n_ii, SumX and SumX2 for the current state of Kal
n_ii = zeros(1,k);  
SumX = zeros(1,k);  
SumX2 = zeros(1,k);   
for i=1:k
    n_ii(i) = sum(Kal==i);
    xi = estar(Kal==i);  
    SumX(i) = sum(xi); SumX2(i) = xi'*xi; 
end 

%% Iterate over each element of Kal, potentially updating the cluster allocation
for t = 1:T
    c_ii = Kal(t);
    n_ii(c_ii) = n_ii(c_ii) - 1; 
    SumX(c_ii) = SumX(c_ii) - estar(t);
    SumX2(c_ii) = SumX2(c_ii) - estar(t)^2;
    % If c_ii is not associated with any cluster but itself
    if n_ii(c_ii) == 0 
        n_ii(c_ii) = []; % remove it from n_ii
        SumX(c_ii) = []; % remove it from SumX
        SumX2(c_ii) = []; % remove it from SumX2
        Kal(Kal>c_ii) = Kal(Kal>c_ii) - 1; % adjust indices
        k = k - 1;
    end 
    % Compute the posterior minus the ii-th observation 
    Vn = 1./( 1./tau + n_ii );
    mn = Vn.* (m./tau + SumX );
    an = s + n_ii./2;
    bn = S + 0.5.*( m^2/tau + SumX2 - mn.^2./Vn);
    % Comnpute the categorial distribution:  
    nu_a = 2.*s;
    sig2_a = S.*(1+tau)./s;
    logint1 = gammaln((nu_a+1)./2)-gammaln(nu_a./2)...
        -0.5*log(nu_a.*pi.*sig2_a) ...
        -(nu_a+1)./2.*log(1 + 1./nu_a.*((estar(t)-m).^2./sig2_a)); 
    nu = 2.*an;
    sig2 = bn.*(1+Vn)./an;
    logintk = gammaln((nu+1)./2)-gammaln(nu./2)...
        -0.5*log(nu.*pi.*sig2) ...
        -(nu+1)./2.*log(1 + 1./nu.*((estar(t)-mn).^2./sig2)); 
    int1 = exp(logint1); intk = exp(logintk); 
    pi_k = [n_ii.*intk, alpha.*int1];
    pi_k = pi_k./sum(pi_k);
    % Draw from the categorical distribution using random numbers psr
    Kal(t) = sum(cumsum(pi_k) < psr(t)) + 1; 
     
    % If Indicator is new, add a parameter to the cluster locations:
    if Kal(t) ==  k + 1   % add a new cluster 
        n_ii = [n_ii, 1];
        SumX = [SumX, estar(t)];
        SumX2 = [SumX2, estar(t).^2];
        k = k + 1; % adjust number of clusters 
    else
        n_ii(Kal(t))  = n_ii(Kal(t)) + 1;
        SumX(Kal(t))  = SumX(Kal(t)) + estar(t);
        SumX2(Kal(t)) = SumX2(Kal(t)) + estar(t).^2;
    end
end 
end