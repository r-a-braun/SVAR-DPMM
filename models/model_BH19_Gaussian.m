function [output] = model_BH19_Gaussian(input)
%-------------------------------------------------------------------------

% Estimates the oil market model of the paper
% "The importance of supply and demand for oil prices: evidence from non-Gaussianity" 
% (Quantitative Economics)
% using Gaussian errors

% Inputs:
%   - input: Structure containing the input parameters for the model estimation.
%     - input.p: VAR lag order.
%     - input.y: Data matrix of size (T x K), where T is the number of observations and K is the number of variables.
%     - input.nthin: Thinning parameter for MCMC samples.
%     - input.ndraws: Number of MCMC draws.
%     - input.nburn: Number of burn-in iterations.
%     - input.pr: Structure containing prior information for the model.
% Outputs:
%   - output: Structure containing the output results of the model estimation.
%     - output.acc_probs: Acceptance probabilities for each column of A in the MCMC.
%     - output.vA_mcmc: MCMC samples of the vectorized structural matrix A.
%     - output.vAplus_mcmc: MCMC samples of the autoregressive parameters.
%     - output.gamma_A_mcmc: MCMC samples of the structural paramaters
%       underlying A
%     - output.shockvar_mcmc: MCMC samples of the shock variances 
%     - output.MontielOlseaStat_mcmc: MCMC samples of the Montiel-Olsea test statistics.
%     - output.MattesonTsayStat_mcmc: MCMC samples of the Matteson-Tsay test statistics.
%       Note that the second row stores values for resampled shocks under
%       the hypothesis of mutually independence shocks
%     - output.input: Copy of the input parameters.

% AUTHOR:
%   Robin Braun
%   robin.andreas.braun@gmail.com
%   Bank of England
%
% DATE:
%   04/05/2023
%
%-------------------------------------------------------------------------



%% Read input:
p = input.p; 
y = input.y;
nthin = input.nthin; 
ndraws = input.ndraws; 
nburn = input.nburn;     

%% Prepare matrices
[TpP, n] = size(y); T = TpP - p; 
y = detrend(y,0); % Demean the data to omit the intercept (not identified)
Y = y(p+1:end,:);  
X = lagmatrix(y,1:p); 
X = X(p+1:end,:);  

%% Estimate the model by Markov Chain Monte Carlo
% Initialization of the model at prior mode
ipVpm = input.pr.iV_vA*input.pr.m_vA; 
Aplus = (X'*X)\(X'*Y); u = Y - X*Aplus;
a_qp = .1; a_yp = -.1;  b_qy = .7; b_qp = -.3; 
psi_1 = -.1; psi_3 = -.4; chi = 0.6; rhostar = 0.25;   
sig2e = sqrt( rhostar/(1-rhostar)*chi^(-2)*1.3 );
gamma_A = [a_qp, a_yp, b_qy, b_qp, psi_1, psi_3, chi, rhostar ];
A = getA_DPM( gamma_A );  
ustar = [ u, 1./chi.*u(:,end)+sig2e*randn(T,1) ]; 
sigma2 = var((A*ustar')');   
sigma2 = sigma2(1:n); 
% Define auxiliary quantities 
J = eye(n+1); J(end,:) = [];  In = eye(n);
ntil = n + 1; 
[i1, i2] = find(kron(ones(n+1),speye(T))); 
Aconst = NaN(n+1); Aconst(A==1)=1; Aconst(A==0)=0; Aconst(A==-1)=-1;
accs = zeros(1,n+1); 
vA_mcmc = zeros((n+1)^2,ndraws);
vAplus_mcmc = zeros(n*(p*n),ndraws);
gamma_A_mcmc = zeros(length(gamma_A),ndraws); 
shockvar_mcmc = zeros(n,ndraws);  
MattesonTsayStat_mcmc = zeros(2,ndraws); 
MontielOlseaStat_mcmc = zeros(2,ndraws);   
tic;
hwaitbar = waitbar(0,'Please wait...');
for irep = 1:ndraws*nthin+nburn
   waitbar(irep/(ndraws*nthin+nburn),hwaitbar);
   
    %% Step 1: Sample the VAR autoregressive parameters   
    Dshock = diag([sigma2,1]); 
    iSigma = (J*((A\Dshock)/A')*J')\In;   
    iV_p = input.pr.iV_vA + kron(iSigma,X'*X) ;
    alphahat = iV_p\(ipVpm + vec(X'*Y*iSigma));  
    vAplus = alphahat + chol(iV_p,'lower')'\randn(length(alphahat),1);
    Aplus = reshape(vAplus,length(alphahat)/n,n);
    u = Y - X*Aplus;


    %% Step 2: Sample USTAR  
    Precision = kron(A'*diag(1./[sigma2,1])*A,speye(T));   
    Qcc = Precision(T*n+1:T*(n+1),T*n+1:T*(n+1));
    mul =  - (Qcc\Precision(T*n+1:T*(n+1),1:T*n) )*vec( u ) ;
    U5star = mul + chol(Qcc,'lower')'\randn(T,1);
    Ustar = [u , U5star];
    
    %% Step 3: Sample the shock variances
    estar = (A*Ustar')'; 
    for j = 1:n 
        %% sample sigma2 
        sigma2(j) = 1/gamrnd(input.pr.s+T/2,1/(input.pr.S + sum(estar(:,j).^2)/2));
    end    
    sigma2_aug = [sigma2,1]; 

    
    %% Step 3: Sample A via an indepednent MH algorithm using Whaggoner Zha as proposal 
    for i = 1:n+1 % Sample row by row via Gibbs Sampler:
        % A*U = E; A*U = MU + Sig_12 *Etil
        Ai = A(i,:);
        [~,lprior] = input.pr.priorAi(Ai,i);
        Gi = eye(n+1); Gi = Gi(:,isnan(Aconst(i,:)));
        s_a = zeros(n+1,1); s_a(Aconst(i,:)==1)=1; s_a(Aconst(i,:)==-1)=-1;
        if sum( or(Aconst(i,:)==1, Aconst(i,:)==-1) )==0
            zi =  zeros(T,1);
        else
            zi = - Ustar*s_a;
        end 
        Ustar_stand = Ustar'./sigma2_aug(i);
        muphi = (Gi'*Ustar_stand*Ustar*Gi)\(Gi'*Ustar_stand*zi);
        Ri = chol(T*inv(Gi'*Ustar_stand*Ustar*Gi),'lower');
        check = 0; count = 0;
        while and(check == 0, count<=1000)
            Aiprop = propose_Ai( A, i, muphi, Ri, Gi, s_a, T);
            [check, lprior_prop] = input.pr.priorAi(Aiprop,i);
            count = count + 1;
        end
        if check == 1 
            MH_acc = exp(lprior_prop-lprior);
            if rand<MH_acc
                Ai = Aiprop;
                accs(i)=accs(i)+1;
            end
        end
        A(i,:)=Ai;
    end 
    %% Save output
    if and(irep>nburn,mod(irep,nthin)==0)
        sig512 = 1./A(5,4); %  sig5: measurement error standard dev
        chi = -A(5,5)/A(5,4); %  chi: OECD fraction of true oil inventories (about 60-65%)
        rhostar = sig512^2/((1.3)/(chi^2)+sig512^2);
        gamma_A = [-A(1,3),-A(2,3),-A(3,2),-A(3,3),...
            -A(4,1),-A(4,3),chi,rhostar];
        vA_mcmc(:, (irep-nburn)/nthin) = vec(A);
        vAplus_mcmc(:, (irep-nburn)/nthin) = vec(Aplus);
        gamma_A_mcmc(:, (irep-nburn)/nthin) = gamma_A;
        shockvar_mcmc(:, (irep-nburn)/nthin) = sigma2;
        estandard = estar./sqrt([sigma2,1]);
        epseudo = estandard;
        for i = 1:size(estandard,2)
            epseudo(:,i) = estandard(randperm(T),i);
        end
        corr_shocks_sq    = corr(estandard.^2); % Correlation matrix for squared shocks+
        corr_shocks_sq_ps = corr(epseudo.^2); % Correlation matrix for squared shocks
        MontielOlseaStat_mcmc(1,(irep-nburn)/nthin) =  sqrt((sum(corr_shocks_sq(:).^2)-ntil)/(ntil^2-ntil)); % Root mean squared off-diagonal correlation
        MontielOlseaStat_mcmc(2,(irep-nburn)/nthin) =  sqrt((sum(corr_shocks_sq_ps(:).^2)-ntil)/(ntil^2-ntil)); % Root mean squared off-diagonal correlation
        MattesonTsayStat_mcmc(1,(irep-nburn)/nthin) =  multidcov_steadyICA(estandard);
        MattesonTsayStat_mcmc(2,(irep-nburn)/nthin) =  multidcov_steadyICA(epseudo);
    end
    
    
end  
close(hwaitbar) 
%% Save the output: 
output.acc_probs = accs/irep;  
disp('MH acceptance probabilities for each column of A:')
disp(output.acc_probs)
output.vA_mcmc = vA_mcmc;
output.vAplus_mcmc = vAplus_mcmc;
output.gamma_A_mcmc = gamma_A_mcmc;
output.shockvar_mcmc = shockvar_mcmc;  
output.MontielOlseaStat_mcmc = MontielOlseaStat_mcmc;
output.MattesonTsayStat_mcmc = MattesonTsayStat_mcmc;
output.input = input; % Also save the input  

end

        