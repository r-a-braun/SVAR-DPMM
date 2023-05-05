function [Kal, alpha,  mut , vt, mean_pd, var_pd, skew_pd, kurt_pd, pred_den] = ...
    dpm_sampler_neal( estar, Kal, alpha, prior)
%-------------------------------------------------------------------------
% DESCRIPTION:
%   This function performs sampling from a Gaussian Dirichlet Process Mixture model using
%   Neals (2000) algorithm 3. It updates the cluster indicators, cluster parameters,
%   and potentially the alpha parameter in the DPM. It also computes some
%   moments and the predictive density of the mixture distribution.
%
% INPUTS:
%   - estar: A column vector of length T representing the structural shocks
%   - Kal: A column vector of length T representing the current cluster indicators for
%           each observation.
%   - alpha: The parameter alpha in the DPM.
%   - prior: A struct containing the prior information with the following fields:
%       - s: The parameter s in the Normal inverse Gamma Base Distribution.
%       - S: The parameter S in the Normal inverse Gamma Base Distribution.
%       - tau: The parameter tau in the Normal inverse Gamma Base Distribution.
%       - m: The parameter m in the Normal inverse Gamma Base Distribution.
%       - a_alpha: (Optional) The shape parameter a_alpha in the gamma prior for alpha.
%       - b_alpha: (Optional) The rate parameter b_alpha in the gamma prior for alpha.
%
% OUTPUTS:
%   - Kal: Updated column vector of length T representing the cluster indicators.
%   - alpha: (Potentially) updated value of the parameter alpha.
%   - mut: A column vector of length T representing the mean parameter of each cluster.
%   - vt: A column vector of length T representing the variance parameter of each cluster.
%   - mean_pd: The mean of the predictive density of the mixture distribution.
%   - var_pd: The variance of the predictive density of the mixture distribution.
%   - skew_pd: The skewness of the predictive density of the mixture distribution.
%   - kurt_pd: The kurtosis of the predictive density of the mixture distribution.
%   - pred_den: A vector representing the predictive density of the (0,1) standardized mixture
%               distribution evaluated at "prior.domain"
%
% AUTHOR:
%   Robin Braun
%   robin.andreas.braun@gmail.com
%   Bank of England
%
% DATE:
%   04/05/2023
%-------------------------------------------------------------------------


T = length(Kal);
%% Step 0: Read in the prior values
% First, the Normal inverse Gamma Base Distribution p(mu,sigma^2)~ NIG( m, tau, s, S)
s = prior.s;
S = prior.S;
tau = prior.tau;
m = prior.m;
% Second, the gamma prior for alpha in case there is one provided
if isfield(prior, 'a_alpha') % alpha ~ Gamma(a_alpha,b_alpha)
    a_alpha = prior.a_alpha;
    b_alpha = prior.b_alpha;
end

%% Step 1: Sample the Mixture indicators
% [Kal,SumX,SumX2,n_ii] = sampleK_matlab(Kal,estar,psr,m,tau,s,S,alpha)
[Kal,SumX,SumX2,n_ii] = sampleK_mex(Kal,estar,rand(T,1),m,tau,s,S,alpha); % Mex function based on C++

%% Step 2: Update the cluster parameters drawing from their conditional posterior distribution and potentially alpha
k = length(n_ii);
phi = zeros(2,k);
for t = 1:k
    ind = (Kal==t);
    phi(:,t) = post_draw_NIG( estar(ind), m, tau, s, S );
end
mut = zeros(T,1);
vt = zeros(T,1);
for j = 1:k
    mut(Kal==j) = phi(1,j);
    vt(Kal==j) = phi(2,j);
end
% Update alpha in case it's a random variable
if isfield(prior, 'a_alpha')
    eta = betarnd(alpha+1,T);
    imed = (a_alpha+k-1)/(T*(b_alpha-log(eta)));
    pi_c = imed/(1+imed);
    mix = pi_c < rand;
    alpha = gamrnd(a_alpha+k-mix, 1./(b_alpha-log(eta)) );
end

%% Compute the standardized predictive density of the mixture distribution
if and(nargout>4, isfield(prior,'domain'))
    %  predictive density
    Vn = 1./( 1./tau + n_ii )  ;
    mn = Vn.* (m./tau + SumX ) ;
    an =  s + n_ii./2 ;
    bn =  S + 0.5.*( m^2/tau + SumX2 - mn.^2./Vn) ;
    Vall = [ Vn, tau];
    mall = [mn, m];
    aall = [an, s];
    ball = [bn, S];
    weights = [n_ii./(T+alpha),alpha./(T+alpha)]';
    nuall = 2.*aall;
    sig2all = ball.*(1+Vall)./aall;
    mean_c = mall;
    var_c =  (sig2all).*(nuall./(nuall-2));
    mom3 = zeros(k+1,1);
    mom4 = ((3+(6./(nuall-4))).*var_c.^2)';
    [mean_pd, var_pd, skew_pd, kurt_pd] = mixture_moments(mean_c',var_c',mom3,mom4,weights );
    logc = gammaln((nuall+1)./2)-gammaln(nuall./2)-0.5*log(nuall.*pi.*sig2all);
    c = exp(logc);
    pden = @(x) (c.*(1 + 1./nuall.*((x-mall).^2./sig2all)).^(-(nuall+1)./2))*weights ;
    sig_pd = sqrt(var_pd);
    pred_denfunction = @(y)(sig_pd).*pden(mean_pd+sig_pd.*y);
    pred_den = pred_denfunction(prior.domain);
    
end
end

function [ phi ] = post_draw_NIG( x, m, V, a, b )
%-------------------------------------------------------------------------
% DESCRIPTION:
%   This function draws a value of phi = (mu, sigma^2) from the posterior
%   distribution, given the observed data x and prior parameters m, V, a, b.
%   The posterior distribution is a Normal inverse Gamma distribution.
%
% INPUTS:
%   - x: A column vector of observed data.
%   - m: The prior mean parameter.
%   - V: The prior variance parameter.
%   - a: The prior shape parameter for the inverse gamma distribution.
%   - b: The prior scale parameter for the inverse gamma distribution.
%
% OUTPUTS:
%   - phi: A column vector representing the drawn value of phi = (mu, sigma^2).
%-------------------------------------------------------------------------
n = length(x);
sum_x = sum(x);
sum_x2 = x'*x;
Vn = 1./( 1./V + n );
mn = Vn* (m/V + sum_x );
an = a + n/2;
bn = b + 0.5*( m^2/V + sum_x2 - mn^2/Vn);
s2 = Draw_iGamma(an,bn);
mu = mn + sqrt(s2.*Vn)*randn;
phi = [mu;s2];
end

function y = Draw_iGamma(a,b)
%-------------------------------------------------------------------------
% DESCRIPTION:
%   This function generates inverse gamma random draws given the shape
%   parameter (a) and rate parameter (b) of the inverse gamma distribution.
%
% INPUTS:
%   - a: The shape parameter of the inverse gamma distribution.
%   - b: The rate parameter of the inverse gamma distribution.
%
% OUTPUTS:
%   - y: The generated inverse gamma random draw.
%-------------------------------------------------------------------------

y = 1./gamrnd(a,1./b);

end

function [M, V, S, K] = mixture_moments(mus,sigmas,mom3,mom4,W)
%-------------------------------------------------------------------------
% [M, V, S, K] = mixture_moments(mus, sigmas, mom3, mom4, W)
%
% DESCRIPTION:
%   This function computes the mean, variance, skewness, and kurtosis of
%   any k mixture distribution with weights W. The input variables mus,
%   sigmas, mom3, and mom4 are kx1 vectors representing the means, variances,
%   third moments, and fourth moments of the individual components in the
%   mixture distribution.
%
% INPUTS:
%   - mus: A kx1 vector representing the means of the individual components.
%   - sigmas: A kx1 vector representing the variances of the individual components.
%   - mom3: A kx1 vector representing the third moments of the individual components.
%   - mom4: A kx1 vector representing the fourth moments of the individual components.
%   - W: A kx1 vector representing the weights of the individual components.
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
%  Skewness and Kurtosis
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
 