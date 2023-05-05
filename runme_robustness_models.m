
clear; clc; 
addpath('auxiliary functions','mex functions','models') 
 

%% Load and set up the dataset used:
BH19data = readtimetable('Dataset\data_BH_AER_updated.xlsx' ,'Sheet','Export');  
Yraw = [100*diff(log(BH19data.Production)),  100*diff(log(BH19data.WIP)),...
    100*diff(log(BH19data.rRAC)),  BH19data.dInv(2:end)];
dates = BH19data.dates(2:end);
start_date = datetime(1975,2,1); % First date available for the RAC
end_date = datetime(2019,12,31);
idx_sample = and(dates>=start_date, dates<=end_date);
idx_train = dates<=start_date;  
y = Yraw(idx_sample,:); 
dates = dates(idx_sample); 
p = 12;

%% Specify the prior distributions:
% 1) Priors on autoregressive parameters and shock variances
[TpP , n]=size(y); 
T = TpP - p ; 
prior.iV_vA = 1./100*speye(n*(p*n)); % prior precision of a+
prior.m_vA = zeros(n*(p*n),1); % prior mean of a+ 
% 2) Set the parameters of the DPMM: alpha, s, S, m, tau
nclusters_prior = 3;  
options = optimoptions('lsqnonlin','Display','off');
[alpha_rot, ~] = lsqnonlin(@(alpha)alpha*(psi(alpha+T)-psi(alpha))-nclusters_prior,1,0,[],options);   
prior.alpha = alpha_rot*ones(n,1); % Rule of thumb for alpha
[prior.s, prior.S] = solve_moments_gamma(2,1.5); % Weakly informative hyperprior 
prior.m = 0; % Uninformative hyperprior
prior.tau = 1;  % Uninformative hyperprior 
prior.domain = linspace(-5,5,200)'; % Domain at which the standardized posterior predictive is evaluated 
% 3) Set the prior for each column of Ai (see function prior_A_Bh19.m)
prior.priorAi = @(Ai,i)prior_A_BH19( Ai, i );

%% Specify the Input
input.y = y; % time series  
input.p = p; % lag length
input.ndraws = 10000;
input.nthin = 10;
input.nburn = 1000;
input.pr = prior;
   
tic;

%% Estimate the models for R1: use prior "prior_A_BH19_R1"
input.pr.priorAi = @(Ai,i)prior_A_BH19_R1( Ai, i ); 
[output_robustness_R1_NG] = model_BH19_NG(input);
save('results/output_robustness_R1_NG.mat','output_robustness_R1_NG')
[output_robustness_R1_Gaussian] = model_BH19_Gaussian(input);
save('results/output_robustness_R1_Gaussian.mat','output_robustness_R1_Gaussian')

%% Estimate the models for R2: use shorter sample
start_date = datetime(1986,1,1) - calmonths(p); 
idx_sample2 = and(dates>=start_date, dates<=end_date); 
inv_tr = detrend(BH19data.dInv(idx_train),0); 
y2 = y(idx_sample2,:); 
dates2 = dates(idx_sample2); 
input.y = y2; % adjusted sample
input.pr.priorAi = @(Ai,i)prior_A_BH19( Ai, i ); % back to standard prior
[output_robustness_R2_NG] = model_BH19_NG(input);
save('results/output_robustness_R2_NG.mat','output_robustness_R2_NG')
[output_robustness_R2_Gaussian] = model_BH19_Gaussian(input);
save('results/output_robustness_R2_Gaussian.mat','output_robustness_R2_Gaussian')


%% R3: Estimate the non-Gaussian model with student-t errors  
input.y = y; % back to full sample
input.pr.nuub = 100;  
[output_robustness_R3_NG] = model_BH19_NG_student_t(input);
save('results/output_robustness_R3_NG.mat','output_robustness_R3_NG')


%% R4: Estimate the non-Gaussian model with a very tight prior around one Gaussian component
nclusters_prior_R4 = 1; % Concentrate prior around a single component
[alpha_R4,res] = lsqnonlin(@(alpha)alpha*(psi(alpha+T)-psi(alpha))-nclusters_prior_R4,1,0,[],options);   
input.pr.alpha = alpha_R4*ones(n,1); % Rule of thumb for alpha under R3
[output_robustness_R4_NG] = model_BH19_NG(input);
save('results/output_robustness_R4_NG.mat','output_robustness_R4_NG')


toc;

 