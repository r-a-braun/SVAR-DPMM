function [ check, logprior ] = prior_A_BH19_R1( Ai, i )
%------------------------------------------------------------------------- 
% DESCRIPTION:
%   This function checks the constraints and computes the log prior
%   probability for the parameters underlying the ith row of the
%   structural matrix A in the oil market model
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
% The A matrix in the BH19 model is:
% A =  [1,     0,    -a_qp,       0,          0;...
%     0,      1,    -a_yp,       0,          0;...
%     1,  -b_qy,    -b_qp,       0,         -1;...
%     -psi_1,     0 ,   -psi_3,       0,          1;...
%     0,       0,        0, 1./sig5, -chi./sig5];


if i == 1
    a_qp = -Ai(3); % alpha(qp): short-run price elasticity of oil supply (sign: positive)
    if a_qp>0
        check = 1;
        c_alpha_qp = 0.1; sigma_alpha_qp = 0.2; nu_alpha_qp = 3;
        logprior = -.5*(nu_alpha_qp+1)./2*log(1+((a_qp-c_alpha_qp)./sigma_alpha_qp).^2./nu_alpha_qp);
    end
elseif i == 2
    a_yp = -Ai(3); % alpha(yp): short-run oil price elasticity of global demand (sign: negative)
    if a_yp<0
        check = 1;
        c_alpha_yp = -0.05; sigma_alpha_yp = 0.1; nu_alpha_yp = 3;
        logprior = -.5*(nu_alpha_yp+1)./2*log(1+((a_yp-c_alpha_yp)./sigma_alpha_yp).^2./nu_alpha_yp);
    end
elseif i == 3
    b_qy = -Ai(2); % beta(qy): income elasticity of oil demand (sign: positive)
    b_qp = -Ai(3); % beta(qp): short-run price elasticity of oil demand (sign: negative & <=0.8)
    if  b_qy>0 && b_qp<0 && b_qp>-0.8 
        check = 1;
        c_beta_qy = 0.7; sigma_beta_qy = 0.1; nu_beta_qy = 3;
        c_beta_qp = -0.1; sigma_beta_qp = 0.2; nu_beta_qp = 3;
        lp1 = -.5*(nu_beta_qy+1)./2*log(1+((b_qy-c_beta_qy)./sigma_beta_qy).^2./nu_beta_qy);
        lp2 = -.5*(nu_beta_qp+1)./2*log(1+((b_qp-c_beta_qp)./sigma_beta_qp).^2./nu_beta_qp);
        logprior = lp1+lp2;
    end
elseif i == 4
    psi_1 = -Ai(1); % psi1: short-run production elasticity of inventory demand (sign: unrestricted)
    psi_3 = -Ai(3); % psi3: short-run price elasticity of inventory demand (sign: unrestricted)
    check = 1; % no restriction
    c_psi1 = 0; sigma_psi1 = 0.5; nu_psi1 = 3;
    lp1 = -.5*(nu_psi1+1)./2*log(1+((psi_1-c_psi1)./sigma_psi1).^2./nu_psi1);
    c_psi3 = 0; sigma_psi3 = 0.5; nu_psi3 = 3;
    lp2 = -.5*(nu_psi3+1)./2*log(1+((psi_3-c_psi3)./sigma_psi3).^2./nu_psi3);
    logprior = lp1+lp2;
elseif i== 5
    sig512 = 1./Ai(4); %  sig5: measurement error standard dev
    chi = -Ai(5)*sig512; %  chi: OECD fraction of true oil inventories (about 60-65%)
    rhostar = sig512^2/((1.3)/(chi^2)+sig512^2);
    if  rhostar>0  && chi>0 &&  rhostar<1 && chi<1
        check = 1;
        alpha_k = 15; beta_k = 10;
        p1 = beta_prior(chi,alpha_k,beta_k);
        [alpha_l, beta_l] = GetBetaParameters(0.25,0.12);
        p2 = beta_prior(rhostar,alpha_l,beta_l); 
        logprior = log(p1)+log(p2);
    end 
end
 


end

