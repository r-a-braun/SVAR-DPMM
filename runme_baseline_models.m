clear; clc; 
addpath('auxiliary functions','mex functions','models')
 
%% Load and set up the dataset used:
BH19data = readtimetable('Dataset\data_BH_AER_updated.xlsx' ,'Sheet','Export');
Yraw = [100*diff(log(BH19data.Production)),  100*diff(log(BH19data.WIP)),...
    100*diff(log(BH19data.rRAC)),  BH19data.dInv(2:end)];
dates = BH19data.dates(2:end);
start_date = datetime(1975,2,1); % First date of the sample
end_date = datetime(2019,12,31); % Last date of the sample  
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
% 2)Prior parameters of the DPMM: alpha, s, S, m, tau
nclusters_prior = 3;
options = optimoptions('lsqnonlin','Display','off');
[alpha_rot, res] = lsqnonlin(@(alpha)alpha*(psi(alpha+T)-psi(alpha))-nclusters_prior,1,0,[],options);
prior.alpha = alpha_rot*ones(n,1); % Rule of thumb for alpha
[prior.s, prior.S] = solve_moments_gamma(2,1.5); % Weakly informative hyperprior
prior.m = 0; % Uninformative hyperprior
prior.tau = 1;  % Uninformative hyperprior
prior.domain = linspace(-5,5,200)'; % Domain at which the standardized posterior predictive is evaluated
% 3) Set the prior for each column of Ai (see function prior_A_Bh19.m)
prior.priorAi = @(Ai,i)prior_A_BH19( Ai, i );
%% Input
input.y = y; % time series
input.p = p; % lag length
input.ndraws = 10000;
input.nthin = 10;
input.nburn = 1000; 
input.pr = prior;


%% Estimate the non-Gaussian model
tic;
[output_baseline_NG] = model_BH19_NG(input);
toc;
save('results/output_baseline_NG.mat','output_baseline_NG')

%% Estimate the Gaussian model
tic;
output_baseline_Gauss = model_BH19_Gaussian(input);
toc;
save('results/output_baseline_Gauss.mat','output_baseline_Gauss')






