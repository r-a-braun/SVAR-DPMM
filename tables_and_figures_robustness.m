%% Note: [see readme]
% Run this code only after estimating 
% the baseline (runme_baseline_models.m)
% and robustness models (runme_robustness_models.m)
%% Add paths and load posterior output
clear; clc; 
addpath('auxiliary functions','results')
load('results/output_baseline_NG.mat') ; 
load('results/output_robustness_R1_NG.mat') ;
load('results/output_robustness_R1_Gaussian.mat') ;
load('results/output_robustness_R2_NG.mat') ;
load('results/output_robustness_R2_Gaussian.mat') ;
load('results/output_robustness_R3_NG.mat') ; 
load('results/output_robustness_R4_NG.mat') ;
color_1 = [255, 0, 0]./255;  
color_2 = [0, 102, 255]./255; 
color_3 = [245, 197, 66]./255;  
color_g = [255, 153, 0]./255;
quants = [0.05,  0.5,  0.95];  
h = 16; 
n = size(output_robustness_R1_NG.input.y,2);  
p = output_robustness_R1_NG.input.p;

%% Table 4: Robustness analysis for the main empirical findings 
model_set = {'baseline_NG','robustness_R1_NG','robustness_R1_Gaussian','robustness_R2_NG','robustness_R2_Gaussian',...
    'robustness_R3_NG','robustness_R4_NG'};
for jj = 1:size(model_set,2)
    disp(strcat("compute FEVDs for model ", model_set{jj})) 
    eval(strcat("output_j = output_",model_set{jj},";")); 
    nrep = size(output_j.gamma_A_mcmc,2);
    FEVD_save = zeros(h+1,n*(n+1),nrep);
    for j=1:nrep 
        % IRFs
        Aplus_j =  reshape(output_j.vAplus_mcmc(:,j),n*p,n)';
        [AA, J] = companion(Aplus_j(:,1:n*p), 0);
        Apoweri = AA;
        Phis = zeros(n,n,h+1);
        Phis(:,:,1) = eye(n);
        for ii = 1:h
            Phis(:,:,ii+1) = J*Apoweri*J';
            Apoweri = Apoweri * AA;
        end
        Aj = reshape(output_j.vA_mcmc(:,j),n+1,n+1);         
        Sig2j = [output_j.shockvar_mcmc(:,j);1];
        Bj = inv(Aj)*diag(sqrt(Sig2j));
        FEVD = zeros(n,n+1,h+1);
        Theta2 = zeros(n,n+1,h+1);
        for ii = 0:h
            Theta2(:,:,ii+1) = Phis(:,:,ii+1)*Bj(1:n,:);
            FEVD(:,:,ii+1) = sum(Theta2(:,:,1:ii+1).^2,3)./sum(sum(Theta2(:,:,1:ii+1).^2,3),2);
        end
        FEVD_save(:,:,j) = reshape(FEVD,n*(n+1),h+1)';
    end
    eval(strcat("FEVD_",model_set{jj},"= quantile(FEVD_save,quants,3);")); 
end

% Create the Table Panel by Panel
horizon = [4,16]';
T_hor = table([horizon;horizon]);  T_hor.Properties.VariableNames={'h'};
fevd_rpo_ir1 = squeeze(round(squeeze(FEVD_robustness_R1_NG([4,16]+1,(1*n-n)+3 ,[1,2,3])),2));
fevd_rpo_ir2 = squeeze(round(squeeze(FEVD_robustness_R2_NG([4,16]+1,(1*n-n)+3 ,[1,2,3])),2)); 
T_i = table([fevd_rpo_ir1;fevd_rpo_ir2]);
T_i.Properties.VariableNames  = {'Contr. of supply (Non-Gaussian)'}; 
fevd_rpo_i2r1 = squeeze(round(squeeze(FEVD_robustness_R1_Gaussian([4,16]+1,(1*n-n)+3 ,[1,2,3])),2));
fevd_rpo_i2r2 = squeeze(round(squeeze(FEVD_robustness_R2_Gaussian([4,16]+1,(1*n-n)+3 ,[1,2,3])),2));
T_i2 = table([fevd_rpo_i2r1;fevd_rpo_i2r2]);
T_i2.Properties.VariableNames  = {'Contr. of supply (Gaussian)'};
T4_FEVD = [T_hor,T_i2,T_i]; 
T4_FEVD.h = {'4 (R1)','16 (R1)','4 (R2)','16 (R2)'}';
table2latex(T4_FEVD, 'results/Tables/table_4_panel_C_fevds.tex');
disp(T4_FEVD) 

AlphaG = [round(quantile(output_robustness_R1_Gaussian.gamma_A_mcmc(1,:),[0.05,0.5,.95]),2) ;...
    round(quantile(output_robustness_R2_Gaussian.gamma_A_mcmc(1,:),[0.05,0.5,.95]),2) ];
AlphaNG = [round(quantile(output_robustness_R1_NG.gamma_A_mcmc(1,:),[0.05,0.5,.95]),2) ;...
    round(quantile(output_robustness_R2_NG.gamma_A_mcmc(1,:),[0.05,0.5,.95]),2) ];
T4_alpha = table(AlphaG,AlphaNG);
T4_alpha = addvars(T4_alpha,{'R1';'R2'},'Before','AlphaG');
T4_alpha.Properties.VariableNames = {'Model','Posterior Supply Elasticity (Gaussian)','Posterior Supply Elasticity (Non-Gaussian)'};
table2latex(T4_alpha, 'results/Tables/table_4_panel_A_alpha.tex');
disp(T4_alpha)

BetaG = [round(quantile(output_robustness_R1_Gaussian.gamma_A_mcmc(4,:),[0.05,0.5,.95]),2) ;...
    round(quantile(output_robustness_R2_Gaussian.gamma_A_mcmc(4,:),[0.05,0.5,.95]),2) ];
BetaNG = [round(quantile(output_robustness_R1_NG.gamma_A_mcmc(4,:),[0.05,0.5,.95]),2) ;...
    round(quantile(output_robustness_R2_NG.gamma_A_mcmc(4,:),[0.05,0.5,.95]),2) ];
T4_beta = table(BetaG, BetaNG);
T4_beta = addvars(T4_beta,{'R1'; 'R2'},'Before','BetaG');
T4_beta.Properties.VariableNames = {'Model','Posterior Demand Elasticity (Gaussian)','Posterior Demand Elasticity (Non-Gaussian)'};
table2latex(T4_beta, 'results/Tables/table_4_panel_B_beta.tex');
disp(T4_beta)

%%  Table F.1 (Appendix): Posterior distribution of the degrees of freedom in model R3 (t-errors) 
nu_posterior = quantile(output_robustness_R3_NG.nu_mcmc,[0.05,0.5,0.95],2);
T6_nu = table((1:4)', round(nu_posterior,1));
T6_nu.Var1 = {'$\eta_1$';'$\eta_2$';'$\eta_3$';'$\eta_4$'};
T6_nu.Properties.VariableNames = {'DoF','posterior quantiles (5%, 50%, 95%)'};
table2latex(T6_nu, 'results/Tables/table_F1_dofs_posterior.tex');
disp(T6_nu)

%% Figure F.1 (Appendix): posterior predictive densities of the oil market shocks in model R4 (strong Gaussian prior)
quants = [.05,.5,.95];
FVALP = quantile(output_robustness_R4_NG.pred_mcmc,quants,3);
color_2 = [0, 102, 255]./255;
color_1 = [255, 0, 0]./255;
facealpha = 0.2;
hFig18=figure(18);
shocks = {' \varepsilon_{t}^{s}', ' \varepsilon_{t}^{ea}',' \varepsilon_{t}^{cd}',' \varepsilon_{t}^{id}', ' \varepsilon_{t}^{me}' };
for i =1:4
    subplot(2,2,i)
    horizon = output_robustness_R4_NG.input.pr.domain;
    hold on; grid on;
    xgraph = [horizon',fliplr(horizon')];
    plot(horizon, normpdf(output_robustness_R4_NG.input.pr.domain), 'b','LineWidth', 1.3,'markersize',3)
    plot(horizon, FVALP(:,i ,2), 'r','LineWidth', 1.3,'markersize',3)
    Y_1 = [FVALP( :, i,1)', fliplr(FVALP(:,i,3)')];
    bounds = fill(xgraph,Y_1,color_1,'LineStyle','none');
    set(bounds,'FaceColor',color_1,'EdgeColor',color_1,'FaceAlpha',facealpha,'EdgeAlpha',facealpha);
    xlim([horizon(1),horizon(end)])
    title(strcat('$',shocks{i},'$'),'Interpreter','latex')
    hold off
end
hFig18 = tightfig(hFig18); 
print(hFig18,'results/Figures/Figure_F1_PredictiviesR4', '-painters' ,'-dpdf')
close(hFig18)

%% Table F.2 (Appendix): Further Robustness analysis for the main empirical findings (based on model R3 and R4)
 
% Panel A and B:
PostGammabase = quantile(output_baseline_NG.gamma_A_mcmc,[0.05,0.5,0.95],2); 
PostGammaR3 = quantile(output_robustness_R3_NG.gamma_A_mcmc,[0.05,0.5,0.95],2); 
PostGammaR4 = quantile(output_robustness_R4_NG.gamma_A_mcmc,[0.05,0.5,0.95],2); 
 
T7supplyela = table(round([PostGammabase(1,:); PostGammaR3(1,:); PostGammaR4(1,:)],3));
T7supplyela = splitvars( T7supplyela );
T7supplyela = addvars(T7supplyela,{'Baseline';'R3';'R4'},'Before','Var1_1');
T7supplyela.Properties.VariableNames = {'model','$\alpha_{qp}$ (5\%)', '$\alpha_{qp}$ (50\%)','$\alpha_{qp}$ (95\%)'};
table2latex(T7supplyela, 'results/Tables/table_F2_panel_A_alpha.tex');
disp(T7supplyela)
 
T7betaela = table(round([PostGammabase(4,:); PostGammaR3(4,:); PostGammaR4(4,:)],3));
T7betaela = splitvars( T7betaela );
T7betaela = addvars(T7betaela,{'Baseline';'R3';'R4'},'Before','Var1_1');
T7betaela.Properties.VariableNames = {'model','$\beta_{qp}$ (5\%)', '$\beta_{qp}$ (50\%)','$\beta_{qp}$ (95\%)'};
table2latex(T7betaela, 'results/Tables/table_F2_panel_B_beta.tex');
disp(T7betaela)

% Panel C: FEVD 
FEVD_baseline = round(squeeze(FEVD_baseline_NG([4,16]+1,((1)*n-n)+3 ,[1,2,3])),2); 
FEVD_R3 = round(squeeze(FEVD_robustness_R3_NG([4,16]+1,((1)*n-n)+3 ,[1,2,3])),2);
FEVD_R4 = round(squeeze(FEVD_robustness_R4_NG([4,16]+1,((1)*n-n)+3 ,[1,2,3])),2);
T7FEVD = table(FEVD_baseline,FEVD_R3,FEVD_R4);
T7FEVD = addvars(T7FEVD,{'h=4';'h=16'},'Before','FEVD_baseline');
T7FEVD.Properties.VariableNames={'h','Posterior Baseline','Posterior R3','Posterior R4'};
table2latex(T7FEVD, 'results/Tables/table_F2_panel_C_FEVD.tex');
disp(T7FEVD)

