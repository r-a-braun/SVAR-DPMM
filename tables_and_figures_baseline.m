%% Note: [see readme]
% Run this code only after estimating 
% the baseline models (runme_baseline_models.m) 

%% Add paths and load posterior output
clear; clc;
addpath('auxiliary functions','results')
load('results/output_baseline_NG.mat') ;
load('results/output_baseline_Gauss.mat') ;
color_1 = [255, 0, 0]./255;
color_2 = [0, 102, 255]./255;
color_3 = [245, 197, 66]./255;
color_g = [255, 153, 0]./255;

%% Print Figure 3: posterior predictive densities of the oil market shocks
quants_fig3 = [.05,.5,.95];
FVALP = quantile(output_baseline_NG.pred_mcmc,quants_fig3,3);
facealpha = 0.2;
hFig3 = figure(3);
shocks = {' \varepsilon_{t}^{s}', ' \varepsilon_{t}^{ea}',...
    ' \varepsilon_{t}^{cd}',' \varepsilon_{t}^{id}', ' \varepsilon_{t}^{me}' };
for i =1:4
    subplot(2,2,i)
    horizon = output_baseline_NG.input.pr.domain;
    hold on; grid on;
    xgraph = [horizon',fliplr(horizon')];
    plot(horizon, normpdf(output_baseline_NG.input.pr.domain), 'b','LineWidth', 1.3,'markersize',3)
    plot(horizon, FVALP(:,i ,2), 'r','LineWidth', 1.3,'markersize',3)
    Y_1 = [FVALP( :, i,1)', fliplr(FVALP(:,i,3)')];
    bounds = fill(xgraph,Y_1,color_1,'LineStyle','none');
    set(bounds,'FaceColor',color_1,'EdgeColor',color_1,'FaceAlpha',facealpha,'EdgeAlpha',facealpha);
    xlim([horizon(1),horizon(end)])
    title(strcat('$',shocks{i},'$'),'Interpreter','latex')
    hold off
end
set(hFig3,'PaperPositionMode','auto')
set(hFig3, 'Position', [30 20 800 400])
hFig3 = tightfig(hFig3);
print(hFig3,'results/Figures/Figure_3_NonGaussianity', '-painters' ,'-dpdf')
close(hFig3)

%% Create Table 2: Skewness and Kurtosis
quants_tab2 = [.05,.5,.95];
Spost = round(quantile(output_baseline_NG.shockskew_mcmc,quants_tab2,2),2)';
Kpost = round(quantile(output_baseline_NG.shockkurt_mcmc,quants_tab2,2),2)';
Skewness_table = table(quants_tab2',Spost(:,1),Spost(:,2),Spost(:,3),Spost(:,4));
Skewness_table.Properties.VariableNames={'quantile','supply','ea','cd','id'};
Kurtosis_table = table(quants_tab2',Kpost(:,1),Kpost(:,2),Kpost(:,3),Kpost(:,4));
Kurtosis_table.Properties.VariableNames={'quantile','supply','ea','cd','id'};
disp('Skewness:')
disp(Skewness_table)
table2latex(Skewness_table, 'results\Tables\table_2_skewness.tex');
disp('Kurtosis:')
disp(Kurtosis_table)
table2latex(Kurtosis_table, 'results\Tables\table_2_kurtosis.tex');

%% Figure 4: Posterior distribution of the test statistic proposed by Olea et al. (2022 and Matteson & Tsay (2017)
hFig4 = figure(4);
alpha_fc = 0.3;
subplot(2,2,4)
hold on; grid on;
p1=histogram(output_baseline_Gauss.MontielOlseaStat_mcmc(1,:),50,'Normalization','pdf','FaceColor',color_2,'FaceAlpha',alpha_fc) ;
p2=histogram(output_baseline_Gauss.MontielOlseaStat_mcmc(2,:),50,'Normalization','pdf','FaceColor',color_3,'FaceAlpha',alpha_fc);
xlim([0,0.3])
legend([p1 p2],{'$S(E)$','$S_{0}(E)$'},'interpreter','latex')
title('$S(E)$ (Gaussian)','interpreter','latex')
hold off
subplot(2,2,3)
hold on; grid on;
p1=histogram(output_baseline_Gauss.MattesonTsayStat_mcmc(1,:),50,'Normalization','pdf','FaceColor',color_2,'FaceAlpha',alpha_fc) ;
p2=histogram(output_baseline_Gauss.MattesonTsayStat_mcmc(2,:),50,'Normalization','pdf','FaceColor',color_3,'FaceAlpha',alpha_fc);
legend([p1 p2],{'$U(E)$','$U_{0}(E)$'},'interpreter','latex')
xlim([-0.3,1])
title('$U(E)$ (Gaussian)','interpreter','latex')
hold off
subplot(2,2,2)
hold on; grid on;
p1 = histogram(output_baseline_NG.MontielOlseaStat_mcmc(1,:),50,'Normalization','pdf','FaceColor',color_1,'FaceAlpha',alpha_fc) ;
p2 = histogram(output_baseline_NG.MontielOlseaStat_mcmc(2,:),50,'Normalization','pdf','FaceColor',color_3,'FaceAlpha',alpha_fc);
xlim([0,0.3])
title('$S(E)$ (Non-Gaussian)','interpreter','latex')
legend([p1 p2],{'$S(E)$','$S_{0}(E)$'},'interpreter','latex')
hold off
subplot(2,2,1)
hold on; grid on;
p1 = histogram(output_baseline_NG.MattesonTsayStat_mcmc(1,:),50,'Normalization','pdf','FaceColor',color_1,'FaceAlpha',alpha_fc) ;
p2 = histogram(output_baseline_NG.MattesonTsayStat_mcmc(2,:),50,'Normalization','pdf','FaceColor',color_3,'FaceAlpha',alpha_fc);
legend([p1 p2],{'$U(E)$','$U_{0}(E)$'},'interpreter','latex')
xlim([-0.3,1])
title('$U(E)$ (Non-Gaussian)','interpreter','latex')
hold off
set(hFig4,'PaperPositionMode','auto')
set(hFig4, 'Position', [30 20 800 400])
hFig4 = tightfig(hFig4);
print(hFig4,'results/Figures/Figure_4_Independence', '-painters' ,'-dpdf')
close(hFig4)

%% Figure 5 (main text) and D.2 (appendix): prior vs posterior density of key  parameters
% alpha(qp): short-run price elasticity of oil supply (sign: positive)
c_alpha_qp = 0.1; sigma_alpha_qp = 0.2; nu_alpha_qp = 3;
prior_alphaqp = makedist('tlocationscale',c_alpha_qp,sigma_alpha_qp,nu_alpha_qp);
priors{1} = truncate(prior_alphaqp,0,inf);
% alpha(yp): short-run oil price elasticity of global demand (sign: negative)
c_alpha_yp = -0.05; sigma_alpha_yp = 0.1; nu_alpha_yp = 3;
prior_alphayp = makedist('tlocationscale',c_alpha_yp,sigma_alpha_yp,nu_alpha_yp);
priors{2} = truncate(prior_alphayp,-inf,0);
% beta(qy): income elasticity of oil demand (sign: positive)
c_beta_qy = 0.7; sigma_beta_qy = 0.1; nu_beta_qy = 3;
prior_betaqy = makedist('tlocationscale',c_beta_qy,sigma_beta_qy,nu_beta_qy);
priors{3} = truncate(prior_betaqy,0,inf);
% beta(qp): short-run price elasticity of oil demand (sign: negative)
c_beta_qp = -0.1; sigma_beta_qp = 0.2; nu_beta_qp = 3;
prior_betaqp = makedist('tlocationscale',c_beta_qp,sigma_beta_qp,nu_beta_qp);
priors{4} = truncate(prior_betaqp,-inf,0);
% psi1: short-run production elasticity of inventory demand (sign: unrestricted)
c_psi1 = 0; sigma_psi1 = 0.5; nu_psi1 = 3;
priors{5} = makedist('tlocationscale',c_psi1,sigma_psi1,nu_psi1);
% psi3: short-run price elasticity of inventory demand (sign: unrestricted)
c_psi3 = 0; sigma_psi3 = 0.5; nu_psi3 = 3;
priors{6} = makedist('tlocationscale',c_psi3,sigma_psi3,nu_psi3);
% xi
priors{7} = makedist('Beta', 15, 10 );
priors{8} = makedist('Beta', 3, 9 );
params = {'\alpha_{qp}', '\alpha_{yp}', '\beta_{qy}', '\beta_{qp}', '\psi_1', '\psi_3', '\chi', '\rho^\star '};
xlims = [0,.5;   -.2,0;   0,3;   -2,0;   -1,1;  -1,1 ;   0,1 ; 0,2];
set_1 = 1:4;
set_2 = 5:8;
alpha_fc = 0.3;
hFig5 = figure(5);
for a = 1:size(set_1,2)
    i = set_1(a);
    subplot(2,size(set_1,2),a)
    histogram(output_baseline_Gauss.gamma_A_mcmc(i,:),50,'Normalization','pdf','FaceColor',color_2,'FaceAlpha',alpha_fc)
    hold on; grid on;
    grid_plot = linspace(xlims(i,1),xlims(i,2),1000);
    plot(grid_plot,priors{i}.pdf(grid_plot),'color',color_g,'LineWidth',2)
    xlim([grid_plot(1),grid_plot(end)])
    title(strcat('$',params{i},'$'),'interpreter','latex')
    if a == 1
        ylabel('Gaussian','interpreter','latex')
    end
    hold off
end
for a = 1:size(set_1,2)
    i = set_1(a);
    subplot(2,size(set_1,2),size(set_1,2)+a)
    histogram(output_baseline_NG.gamma_A_mcmc(i,:),'Normalization','pdf','FaceColor',color_1,'FaceAlpha',alpha_fc)
    hold on
    grid on
    grid_plot = linspace(xlims(i,1),xlims(i,2),1000);
    plot(grid_plot,priors{i}.pdf(grid_plot),'color',color_g,'LineWidth',2)
    xlim([grid_plot(1),grid_plot(end)])
    title(strcat('$',params{i},'$'),'interpreter','latex')
    if a == 1
        ylabel('non-Gaussian','interpreter','latex')
    end
    hold off
end
set(gcf,'PaperPositionMode','auto')
set(hFig5, 'Position', [30 50 900 400])
hFig5 = tightfig(hFig5);
print(hFig5,'results/Figures/Figure_5_posteriors', '-painters' ,'-dpdf')
close(hFig5)


hFig15 = figure(15);
for a = 1:size(set_2,2)
    i = set_2(a);
    subplot(2,size(set_2,2),a)
    histogram(output_baseline_Gauss.gamma_A_mcmc(i,:),'Normalization','pdf','FaceColor',color_2,'FaceAlpha',alpha_fc)
    hold on; grid on;
    grid_plot = linspace(xlims(i,1),xlims(i,2),1000);
    plot(grid_plot,priors{i}.pdf(grid_plot),'color',color_g,'LineWidth',2)
    xlim([grid_plot(1),grid_plot(end)])
    title(strcat('$',params{i},'$'),'interpreter','latex')
    if a == 1
        ylabel('Gaussian','interpreter','latex')
    end
    hold off
end
for a = 1:size(set_2,2)
    i = set_2(a);
    subplot(2,size(set_2,2),size(set_2,2)+a)
    histogram(output_baseline_NG.gamma_A_mcmc(i,:),'Normalization','pdf','FaceColor',color_1,'FaceAlpha',alpha_fc)
    hold on; grid on;
    grid_plot = linspace(xlims(i,1),xlims(i,2),1000);
    plot(grid_plot,priors{i}.pdf(grid_plot),'color',color_g,'LineWidth',2)
    xlim([grid_plot(1),grid_plot(end)])
    title(strcat('$',params{i},'$'),'interpreter','latex')
    if a == 1
        ylabel('non-Gaussian','interpreter','latex')
    end
    hold off
end
set(gcf,'PaperPositionMode','auto')
set(hFig15, 'Position', [30 50 900 400])
hFig15 = tightfig(hFig15);
print(hFig15,'results/Figures/Figure_D2_posteriors', '-painters' ,'-dpdf')
close(hFig15)

%% Figure 6: Posterior median IRFs with 90% credible intervals (shaded areas)

quants_IRFsFEVDs = [0.05, 0.16, 0.5, 0.84, 0.95]; % Posterior Quantile Level
shocks = {' \varepsilon_{t}^{s}', ' \varepsilon_{t}^{ad}',' \varepsilon_{t}^{cd}',' \varepsilon_{t}^{id}', ' \varepsilon_{t}^{me}' };
shocks2 = {'Supply','Aggregate Demand','Consumption Demand','Inventory Demand','Measurement Error'};
cumsumdummys = [1, 1 , 1, 1 ];
variables = {' q_t', ' y_t',' p_t' , ' i_t'};
nrep = size(output_baseline_NG.gamma_A_mcmc,2);
% Impulse Response and FEVD Analysis of the Non-Gaussian Model
h = 16; n = size(output_baseline_NG.input.y,2); p = output_baseline_NG.input.p;
cumsumdummys_location = repmat(logical(cumsumdummys),1,3);
FEVD_save = zeros(h+1,n*(n+1),nrep);
irf_save = zeros(h+1,n*(n+1),nrep);
for j=1:nrep
    clc; disp(j/nrep)
    % IRFs
    Aplus_j =  reshape(output_baseline_NG.vAplus_mcmc(:,j),n*p,n)';
    [AA, J] = companion(Aplus_j(:,1:n*p), 0);
    Apoweri = AA;
    Phis = zeros(n,n,h+1);
    Phis(:,:,1) = eye(n);
    for ii = 1:h
        Phis(:,:,ii+1) = J*Apoweri*J';
        Apoweri = Apoweri * AA;
    end
    Aj = reshape(output_baseline_NG.vA_mcmc(:,j),n+1,n+1);
    Sig2j = [output_baseline_NG.shockvar_mcmc(:,j);1];
    Bj = inv(Aj)*diag(sqrt(Sig2j));
    Bsd = Bj./[Bj(3,1),Bj(3,2),Bj(3,3),Bj(3,4),Bj(4,5)];
    FEVD = zeros(n,n+1,h+1);
    Theta = zeros(n,n+1,h+1);
    Theta2 = zeros(n,n+1,h+1);
    for ii = 0:h
        Theta(:,:,ii+1) = Phis(:,:,ii+1)*Bsd(1:n,:);
        Theta2(:,:,ii+1) = Phis(:,:,ii+1)*Bj(1:n,:);
        FEVD(:,:,ii+1) = sum(Theta2(:,:,1:ii+1).^2,3)./sum(sum(Theta2(:,:,1:ii+1).^2,3),2);
    end
    irfs = reshape(Theta,n*(n+1),h+1)';
    irf_save(:,:,j) = cumsum(irfs);
    FEVD_save(:,:,j) = reshape(FEVD,n*(n+1),h+1)';
end
IRFqs_NG = quantile(irf_save,quants_IRFsFEVDs,3);
FEVDqs_NG  = quantile(FEVD_save,quants_IRFsFEVDs,3);
% Impulse Response and FEVD Analysis of the Gaussian Model
nrep = size(output_baseline_Gauss.gamma_A_mcmc,2);
FEVD_save2 = zeros(h+1,n*(n+1),nrep);
irf_save2 = zeros(h+1,n*(n+1),nrep);
for j=1:nrep
    clc; disp(j/nrep)
    % IRFs
    Aplus_j =  reshape(output_baseline_Gauss.vAplus_mcmc(:,j),n*p,n)';
    [AA, J] = companion(Aplus_j(:,1:n*p), 0);
    Apoweri = AA;
    Phis = zeros(n,n,h+1);
    Phis(:,:,1) = eye(n);
    for ii = 1:h
        Phis(:,:,ii+1) = J*Apoweri*J';
        Apoweri = Apoweri * AA;
    end
    Aj = reshape(output_baseline_Gauss.vA_mcmc(:,j),n+1,n+1);
    Sig2j = [output_baseline_Gauss.shockvar_mcmc(:,j);1];
    Bj = inv(Aj)*diag(sqrt(Sig2j));
    Bsd = Bj./[Bj(3,1),Bj(3,2),Bj(3,3),Bj(3,4),Bj(4,5)];
    FEVD = zeros(n,n+1,h+1);
    Theta = zeros(n,n+1,h+1);
    Theta2 = zeros(n,n+1,h+1);
    for ii = 0:h
        Theta(:,:,ii+1) = Phis(:,:,ii+1)*Bsd(1:n,:);
        Theta2(:,:,ii+1) = Phis(:,:,ii+1)*Bj(1:n,:);
        FEVD(:,:,ii+1) = sum(Theta2(:,:,1:ii+1).^2,3)./sum(sum(Theta2(:,:,1:ii+1).^2,3),2);
    end
    irfs = reshape(Theta,n*(n+1),h+1)';
    irf_save2(:,:,j) = cumsum(irfs);
    FEVD_save2(:,:,j) = reshape(FEVD,n*(n+1),h+1)';
end
IRFqs_G = quantile(irf_save2,quants_IRFsFEVDs,3);
FEVDqs_G = quantile(FEVD_save2,quants_IRFsFEVDs,3);

hFig6 = figure(6);
horizon = (0:h)';
xgraph = [horizon',fliplr(horizon')];
a = 1;
facealpha = 0.2;
for ii = 1:n
    for i = 1:n
        subplot(n,n,a)
        hold on; grid on;
        plot(horizon, IRFqs_G(:,(ii*n-n)+i ,3), 'b','LineWidth', 1.3,'markersize',3)
        plot(horizon, IRFqs_NG(:,(ii*n-n)+i ,3), 'r','LineWidth', 1.3,'markersize',3)
        line([horizon(1),horizon(end)],[0,0],'color',[105,105,105]./255 )
        Y_1 = [IRFqs_NG( :, (ii*n-n)+i,1)', fliplr(IRFqs_NG(:,(ii*n-n)+i,5)')];
        bounds = fill(xgraph,Y_1,color_1,'LineStyle','none');
        set(bounds,'FaceColor',color_1,'EdgeColor',color_1,'FaceAlpha',facealpha,'EdgeAlpha',facealpha);
        line([horizon(1),horizon(end)],[0,0],'color',[105,105,105]./255 )
        Y_1 = [IRFqs_G( :, (ii*n-n)+i,1)', fliplr(IRFqs_G(:,(ii*n-n)+i,5)')];
        bounds = fill(xgraph,Y_1,color_2,'LineStyle','none');
        set(bounds,'FaceColor',color_2,'EdgeColor',color_2,'FaceAlpha',facealpha,'EdgeAlpha',facealpha);
        hold off
        xlim([horizon(1),horizon(end)])
        title(strcat('$',shocks{ii},' \rightarrow',variables{i},'$'),'Interpreter','latex')
        xticks([0,4:4:h])
        a = a + 1;
        if i == 1
            ylabel(shocks2{ii},'interpreter','latex')
        end
    end
end
set(gcf,'PaperPositionMode','auto')
set(hFig6, 'Position', [30 50 900 700])
hFig6 = tightfig(hFig6);
print(hFig6,'results/Figures/Figure_6_IRF', '-painters' ,'-dpdf')
close(hFig6)


%% Table 3: Forecast Error Variance Decomposition (FEVD) of the real oil price growth
horizon = [4;16]; idx = (1:n+1)*(n)-n+3;
FEVD_NG_table_50 = round(squeeze(FEVDqs_NG([4,16]+1,idx ,3)),2);
FEVD_G_table_50 = round(squeeze(FEVDqs_G([4,16]+1,idx,3)),2);
FEVD_median = table(horizon,FEVD_NG_table_50,FEVD_G_table_50);
FEVD_median.Properties.VariableNames = {'h','FEVD RPO non-Gaussian model (median)','FEVD RPO Gaussian model (median)'};
table2latex(FEVD_median, 'results/Tables/table_3_fevds_median.tex');
disp(FEVD_median)

FEVD_NG_table_05 = round(squeeze(FEVDqs_NG([4, 16]+1,idx ,1)),2);
FEVD_G_table_05 = round(squeeze(FEVDqs_G([4, 16]+1,idx ,1)),2);
FEVD_05 = table(horizon,FEVD_NG_table_05,FEVD_G_table_05);
FEVD_05.Properties.VariableNames = {'h','FEVD RPO non-Gaussian model (5\%)','FEVD RPO Gaussian model (5\%)'};
table2latex(FEVD_05, 'results/Tables/table_3_fevds_05_quantiles.tex');


FEVD_NG_table_95 = round(squeeze(FEVDqs_NG([4, 16]+1,idx ,5)),2);
FEVD_G_table_95 = round(squeeze(FEVDqs_G([4, 16]+1,idx ,5)),2);
FEVD_95 = table(horizon,FEVD_NG_table_95,FEVD_G_table_95);
FEVD_95.Properties.VariableNames = {'h','FEVD RPO non-Gaussian model (95\%)','FEVD RPO Gaussian model (95\%)'};
table2latex(FEVD_95, 'results/Tables/table_3_fevds_95_quantiles.tex');


%% Figure B.7 (Appendix B): Markov Chain Monte Carlo output of each element of A plus Relative Numerical Efficiency statistics
params = {'\alpha_{qp}', '\alpha_{yp}', '\beta_{qy}', '\beta_{qp}', '\psi_1', '\psi_3', '\chi', '\rho^\star '};
[n, irep] = size(output_baseline_NG.gamma_A_mcmc);
hfig13 = figure(13);
for i = 1:n
    subplot(2,4,i)
    plot(1:irep,output_baseline_NG.gamma_A_mcmc(i,:),'k')
    RNE = round(GewekeRNE(output_baseline_NG.gamma_A_mcmc(i,:)'),2);
    xlim([1,irep])
    title(strcat('$',params{i},'$,','\  RNE: \,$',num2str(RNE),'$'),'interpreter','latex')
end
set(gcf,'PaperPositionMode','auto')
set(hfig13, 'Position', [30 50 1000 500])
hfig13 = tightfig(hfig13);
print(hfig13,'results/Figures/Figure_B7_convergence', '-painters' ,'-dpdf')
close(hfig13)

