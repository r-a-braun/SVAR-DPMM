function [A,B] = solve_moments_gamma(mu,sigma2)
%-------------------------------------------------------------------------
% DESCRIPTION:
%   This function solves for the parameters A and B of the gamma distribution
%   that match the given mean and variance.
%
% INPUTS:
%   - mu: The desired mean.
%   - sigma2: The desired variance.
%
% OUTPUTS:
%   - A: The shape parameter of the gamma distribution.
%   - B: The rate parameter of the gamma distribution.
%-------------------------------------------------------------------------

theta0 = [1,1];
options = optimoptions('fsolve','Display','none','Algorithm','levenberg-marquardt');
theta = fsolve( @(theta)objfun_gam(theta,mu,sigma2) , theta0 , options );
A = theta(1); B = theta(2);  
end

function x = objfun_gam(theta,mu,sigma2) 
[M,V] = gamstat(theta(1),theta(2));
x = (mu-M)^2 + (V-sigma2).^2;
end