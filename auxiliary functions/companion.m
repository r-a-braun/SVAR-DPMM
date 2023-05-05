function [ A, J , nu] = companion( Bhat , inc)
%-------------------------------------------------------------------------
% DESCRIPTION:
%   This function converts the parameter vector [c, A_1, ..., A_p] into
%   the companion form of a VAR(1) model, represented by the matrix A.
%   The companion form is used in vector autoregressive (VAR) models.
%
% INPUTS:
%   - Bhat: Parameter matrix of size (K, N*K+1), where K is the number of
%           variables and N is the number of lags in the VAR model.
%   - inc:  Indicator for including the constant term. Set inc = 1 to
%           include the constant, and inc = 0 to exclude it.
%
% OUTPUTS:
%   - A:   Sparse matrix representing the VAR(1) companion form. Its size
%          is (K*N, K*N), where K is the number of variables and N is the
%          number of lags in the VAR model.
%   - J:   Sparse matrix representing the selection matrix. It is used to
%          extract the current period's endogenous variables from the VAR
%          companion form. Its size is (K, K*N).
%   - nu:  Vector of zeros with a length of K*N. It represents the
%          constant term in the VAR model. If inc = 1, the constant term
%          is extracted from Bhat and placed in the first K elements of nu. 
%-------------------------------------------------------------------------

[K, Kpinc]=size(Bhat);
p = (Kpinc-inc)/K; 
nu = zeros(K*p,1);
if inc == 1
    nu(1:K) = Bhat(:,1);
end  
A = [Bhat(:,1+inc:end);[speye(K*max((p-1),0)),sparse(K*max((p-1),0),K)]]; 
J = [speye(K),sparse(K,K*max((p-1),0))];   
