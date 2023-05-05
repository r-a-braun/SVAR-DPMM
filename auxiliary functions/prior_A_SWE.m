function [ check, logprior ] = prior_A_SWE( Ai, i ) 
%------------------------------------------------------------------------- 
% DESCRIPTION:
%   This function checks the constraints and computes the log prior
%   probability for the parameters underlying the ith row of the
%   structural matrix A in the model underlying the small working example 
%
% INPUTS:
%   - Ai: The i-th row of A
%   - i: 
%
% OUTPUTS:
%   - check: A flag indicating if the parameter satisfies the constraints.
%   - logprior: The log prior probability for the parameter. 
%-------------------------------------------------------------------------  




check = 0; logprior = -inf;
if i == 1
    a_qp = -Ai(2); % alpha(qp): short-run price elasticity of oil supply (sign: positive)
    if a_qp>0
        check = 1;
        c_alpha_qp = 0.1; sigma_alpha_qp = 0.2; nu_alpha_qp = 3;
        logprior =  -.5*(nu_alpha_qp+1)./2*log(1+((a_qp-c_alpha_qp)./sigma_alpha_qp).^2./nu_alpha_qp);
    end
elseif i == 2
    b_qp = -Ai(2); % beta(qp): short-run price elasticity of oil demand (sign: negative)
    if b_qp<0
        check = 1;
        c_beta_qp = -0.1; sigma_beta_qp = 0.2; nu_beta_qp = 3;
        logprior =   -.5*(nu_beta_qp+1)./2*log(1+((b_qp-c_beta_qp)./sigma_beta_qp).^2./nu_beta_qp);
    end
end



end

