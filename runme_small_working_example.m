clear; clc; 
addpath('auxiliary functions', 'mex functions', 'models')
 

%% Simulate a VAR(0) model with simultanous equation model for the prediction errors
% This model is estimated including marginal likelihood, as outlined in
% Appendix B and C
% supply: q_t = alpha*p_t + sig1*eps_1t
% demand: q_t = beta*p_t + sig2*eps_2t
a_pq =  0.15; sig1 = 1;
b_pq = -0.35; sig2 = 0.5;
A = [1, -a_pq;...
    1,  -b_pq]; 
Sig12 = diag([sig1,sig2]); 
T = 500; n = 2; p = 1;
nu_t = 3;
struct_shock = zscore(trnd(nu_t,[T+1,n]));   
y = (inv(A)*Sig12*struct_shock')';   

%% Specify the prior distributions:
% 1) Priors on autoregressive parameters and shock variances
[TpP , n]=size(y); 
T = TpP - p ; 
prior.iV_vA = 1./100*speye(n*(p*n)); % prior precision of a+
prior.m_vA = zeros(n*(p*n),1); % prior mean of a+ 
% 2) Set the parameters of the DPMM: alpha, s, S, m, tau
nclusters_prior = 3;  
options = optimoptions('lsqnonlin','Display','off');
[alpha_rot, res] = lsqnonlin(@(alpha)alpha*(psi(alpha+T)-psi(alpha))-nclusters_prior,1,0,[],options);   
prior.alpha = alpha_rot*ones(n,1); % Rule of thumb for alpha
[prior.s, prior.S] = solve_moments_gamma(2,10); % Weakly informative hyperprior 
prior.m = 0; % Uninformative hyperprior
prior.tau = 5;  % Uninformative hyperprior 
prior.domain = linspace(-5,5,200)'; % Domain at which the standardized posterior predictive is evaluated 
% 3) Set the prior for each column of Ai (see function prior_A_Bh19.m)
prior.priorAi = @(Ai,i)prior_A_SWE( Ai, i );

%% Specify input and estimate the model by Monte Carlo:
input.y = y; % time series  
input.p = p; % lag length
input.ndraws = 10000;
input.nthin = 1;
input.nburn = 1000;   

input.pr = prior;
input.Ainit = [ 1, -.3 ; 1, .2 ]; % some initialization for the A matrix
input.compute_independence_test = 0; 
output = model_SWE_NG(input);

%% Print the Posteriors of alpha and beta 
alpha_draws = - output.vA_mcmc(3,:);
beta_draws = - output.vA_mcmc(4,:);  
% alpha(qp): short-run price elasticity of oil supply (sign: positive)
c_alpha_qp = 0.1; sigma_alpha_qp = 0.2; nu_alpha_qp = 3;
prior_alphaqp = makedist('tlocationscale',c_alpha_qp,sigma_alpha_qp,nu_alpha_qp);
priors{1} = truncate(prior_alphaqp,0,inf);
% beta(qp): short-run price elasticity of oil demand (sign: negative)
c_beta_qp = -0.1; sigma_beta_qp = 0.2; nu_beta_qp = 3;
prior_betaqp = makedist('tlocationscale',c_beta_qp,sigma_beta_qp,nu_beta_qp);
priors{2} = truncate(prior_betaqp,-inf,0);
color_2 = [0, 102, 255]./255;   
color_1 = [255, 0, 0]./255;   
xlims = [0,1; -1,0];
color_g = [255, 153, 0]./255;
alpha_fc = 0.3;
hfig2=figure(2);
subplot(2,2,1);
hold on; grid on;
plot(alpha_draws,'k')
title('Markov Chain $\alpha_{qp}$','interpreter','latex')  
subplot(2,2,2) 
hold on; grid on; 
plot(beta_draws,'k') 
title('Markov Chain $\beta_{qp}$','interpreter','latex') 
subplot(2,2,3) 
histogram(alpha_draws,'normalization','pdf','FaceColor',color_2,'FaceAlpha',alpha_fc)
hold on; grid on;
xevalu = linspace(xlims(1,1),xlims(1,2),1000);
plot(xevalu,priors{1}.pdf(xevalu),'color',color_g,'LineWidth',2)
xlim([xevalu(1),xevalu(end)]) 
title('Prior vs posterior $\alpha_{qp}$','interpreter','latex')  
subplot(2,2,4) 
histogram(beta_draws,'normalization','pdf','FaceColor',color_1,'FaceAlpha',alpha_fc) 
hold on; grid on;
xevalu = linspace(xlims(2,1),xlims(2,2),1000);
plot(xevalu,priors{2}.pdf(xevalu),'color',color_g,'LineWidth',2)
xlim([xevalu(1),xevalu(end)])  
title('Prior vs posterior $\beta_{qp}$','interpreter','latex') 


%% Next, estimate the Marginal Likelihood of the unrestricted model via the Cross Entropy Method
% First, start with the ML estimate of the unrestricted model
prior.n = n;
M_is = 500; % repetitions 
dof_IS = 7; 
alphaplushat_IS = mean(output.vAplus_mcmc,2); alphaplusvar_IS = cov(output.vAplus_mcmc');
gamma_A_mcmc = - output.vA_mcmc([3,4],:); % free elements in A 
gamma_var_IS = cov(gamma_A_mcmc'); 
gamma_Ahat_IS = mean(gamma_A_mcmc,2);
Scale_Ahat = (dof_IS-2)/dof_IS*gamma_var_IS; 
q = size(output.vA_mcmc,1);
y = detrend(y,0); % Demean the data to omit the intercept (not identified)
Y = y(p+1:end,:);  
X = lagmatrix(y,1:p); 
X = X(p+1:end,:);  
nbigloop = 10; k = n*p;
sml = zeros(nbigloop,1);
for bigloop = 1:nbigloop
    disp(strcat('loop:',num2str(bigloop)))
    sllike = zeros(M_is,1); likeonly =sllike;
    salphaplusc = repmat(alphaplushat_IS',M_is,1) + (chol(alphaplusvar_IS)' * randn(q,M_is))'; 
    nA = length(gamma_Ahat_IS);
    uchi2 = chi2rnd(dof_IS,[M_is,1]) ;
    sgammaac = gamma_Ahat_IS' + sqrt(dof_IS./uchi2).*(chol(Scale_Ahat)' * randn(nA,M_is))';
    prior.V_vA  = inv(prior.iV_vA);
    for loop = 1:M_is
        alphaplusdraw = salphaplusc(loop,:)';
        Aplus = reshape(alphaplusdraw,k,n);
        u = Y-X*Aplus;
        gamma_adraw = sgammaac(loop,:)';
        A = [1, -gamma_adraw(1);...
            1,  -gamma_adraw(2)];
        E = (A*u')';
        lpriorA = 0;
        for i = 1:size(u,2)
            [~,lpriori] = prior_A_SWE( A(i,:), i );
            lpriorA = lpriorA + lpriori; 
        end
        if lpriorA == -inf
            llike = - inf; % don't bother computing it, SR not satisfied
        else
            llike = + T*log(abs(det(A))) ;
            for i = 1:size(u,2)
                [liki, Ri]=SIS_DPM(E(:,i),prior.alpha(i),prior, 50);  
                llike = llike + liki;
            end
        end 
        likeonly(loop) = llike;
        sllike(loop) = llike + ...
            lmvnpdf(alphaplusdraw, prior.m_vA,prior.V_vA) + lpriorA ... % PRIOR
            - lmvnpdf(alphaplusdraw,alphaplushat_IS,alphaplusvar_IS)... % Importance Density
            - lmvtpdf(gamma_adraw,gamma_Ahat_IS,Scale_Ahat,dof_IS);
        clc; disp( ((bigloop-1)*M_is+loop)/(M_is*nbigloop))
    end
    meanllike = max(sllike);
    mlIS = log(mean(exp(sllike-meanllike))) + meanllike;
    sml(bigloop) = mlIS;
end
ML_DPM = mean(sml); MLstd_DPM = std(sml)/sqrt(nbigloop);
disp(strcat( 'ML of the unrestricted Nongaussian Model: ' , ' ' , num2str(round(ML_DPM,2)) ) )
disp(strcat( 'Standard error:' , ' ' , num2str(round(MLstd_DPM,4)) ) )

% Estimate the restricted Model:
input.Ainit = [ 1, -.3 ; 1, .2 ]; %some initialization for the A matrix
input.Ainit(1,2) = 0; % Set this to 0 (restriction to be tested)
input.compute_independence_test = 0; 
output_restricted = model_SWE_NG_restricted(input); % Obtain posterior output for the restricted model

% Finally, compute the Marginal Likelihood of the restricted model via the Cross Entropy Method
gamma_A_mcmc_r = - output_restricted.vA_mcmc(4,:); % free elements in A 
vAplus_mcmc_r = output_restricted.vAplus_mcmc;
alphaplushat_IS_r = mean(vAplus_mcmc_r,2); alphaplusvar_IS_r = cov(vAplus_mcmc_r');
gamma_var_IS_r = cov(gamma_A_mcmc_r'); gamma_Ahat_IS_r = mean(gamma_A_mcmc_r,2);
Scale_Ahat_r = (dof_IS-2)/dof_IS*gamma_var_IS_r;
nbigloop = 10;
sml_r = zeros(nbigloop,1);
for bigloop = 1:nbigloop
    disp(strcat('loop:',num2str(bigloop)))
    sllike = zeros(M_is,1); likeonly = sllike;
    salphaplusc = repmat(alphaplushat_IS_r',M_is,1) + (chol(alphaplusvar_IS_r)' * randn(q,M_is))';
    dof_IS = 7; 
    nA = length(gamma_Ahat_IS_r);
    uchi2 = chi2rnd(dof_IS,[M_is,1]) ;
    sgammaac = gamma_Ahat_IS_r' + sqrt(dof_IS./uchi2).*(chol(Scale_Ahat_r)' * randn(nA,M_is))';
    prior.V_vA  = inv(prior.iV_vA);
    for loop = 1:M_is
        alphaplusdraw_r = salphaplusc(loop,:)';
        Aplus = reshape(alphaplusdraw_r,k,n);
        u = Y-X*Aplus;
        gamma_adraw_r = sgammaac(loop,:)';
        A = [1, 0;...
            1,  -gamma_adraw_r(1)];
        E = (A*u')';
        lpriorA = 0;
        for i = 2
            [~,lpriori] = prior_A_SWE( A(i,:), i );
            lpriorA = lpriorA + lpriori; 
        end
        if lpriorA == -inf
            llike = - inf; % don't bother computing it, SR not satisfied
        else
            llike = + T*log(abs(det(A))) ;
            for i = 1:size(u,2)
                [liki, Ri]=SIS_DPM(E(:,i),prior.alpha(i),prior, 50);  
                llike = llike + liki;
            end
        end 
        likeonly(loop) = llike;
        sllike(loop) = llike + ...
            lmvnpdf(alphaplusdraw_r, prior.m_vA,prior.V_vA) + lpriorA ... % PRIOR
            - lmvnpdf(alphaplusdraw_r,alphaplushat_IS_r,alphaplusvar_IS_r)... % Importance Density
            - lmvtpdf(gamma_adraw_r,gamma_Ahat_IS_r,Scale_Ahat_r,dof_IS);
        clc; disp( ((bigloop-1)*M_is+loop)/(M_is*nbigloop))
    end
    meanllike = max(sllike);
    mlIS = log(mean(exp(sllike-meanllike))) + meanllike;
    sml_r(bigloop) = mlIS;
end
ML_DPM_r = mean(sml_r); MLstd_DPM_r = std(sml_r)/sqrt(nbigloop);


% Print the output of the Marginal Likelihood Estimation Exercise:
BF2LN = 2*(ML_DPM-ML_DPM_r); 
disp(strcat( 'ML of the unrestricted Nongaussian Model: ' , ' ' , num2str(round(ML_DPM,2)) ) )
disp(strcat( 'Standard error:' , ' ' , num2str(round(MLstd_DPM,4)) ) )
disp(strcat( 'ML of the restricted Nongaussian Model: ' , ' ' , num2str(round(ML_DPM_r,2)) ) )
disp(strcat( 'Standard error:' , ' ' , num2str(round(MLstd_DPM_r,4)) ) )
disp(strcat( 'The estimated Bayes Factor is (2ln scale): ' , ' ' , num2str(round(BF2LN,2)) ) )




