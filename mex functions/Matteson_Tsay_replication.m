clear; clc; 
load('freedman.mat') 
A = [log(Freedman.population),Freedman.nonwhite,Freedman.density,Freedman.crime];
A =  A(sum(isnan(A),2)==0,:) ;
A = A - mean(A,1);
A = A./([0.79, 10.08, 1441.95, 983.58]);
alpha = 1; 
%% Raw Data
UhatY = multidcov_steadyICA(A,alpha); 
disp('Statistic for Raw Data: ') 
fprintf('%.2f', UhatY) 
fprintf('\n') 
%% PCA
[~,Z,~] = pca(A);  
UhatZ = multidcov_steadyICA(Z,alpha);  
disp('Statistic for PCA:') 
fprintf('%.2f', UhatZ)
fprintf('\n') 
 
