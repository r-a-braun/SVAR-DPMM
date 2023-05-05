function [output] = model_BH19_NG(input)
%-------------------------------------------------------------------------
% Estimates the oil market model of the paper
% "The importance of supply and demand for oil prices: evidence from non-Gaussianity" 
% (Quantitative Economics) using non-Gaussian shocks modeled by a Dirichlet
% Mixture model 

% Inputs:
%   - input: Structure containing the input parameters for the model estimation.
%     - input.p: VAR lag order.
%     - input.y: Data matrix of size (T x K), where T is the number of observations and K is the number of variables.
%     - input.nthin: Thinning parameter for MCMC samples.
%     - input.ndraws: Number of MCMC draws.
%     - input.nburn: Number of burn-in iterations.
%     - input.pr: Structure containing prior information for the model.
%       - input.pr.alpha: Concentration parameter for the Dirichlet Process Mixture Model (DPMM).
%       - input.pr.iV_vA: Prior inverse covariance matrix for the VAR autoregressive parameters.
%       - input.pr.m_vA: Prior mean vector for the VAR autoregressive parameters.
%       - input.pr.domain: grid of points to evaluate the predictive
%         distributions
%       - input.pr.priorAi: Function handle for the prior distribution of each row of A.

% Outputs:
%   - output: Structure containing the output results of the model estimation.
%     - output.acc_probs: Acceptance probabilities for each column of A in the MCMC.
%     - output.vA_mcmc: MCMC samples of the vectorized AR parameter matrix A.
%     - output.vAplus_mcmc: MCMC samples of the autoregressive parameters.
%     - output.gamma_A_mcmc: MCMC samples of the structural paramaters
%       underlying A
%     - output.shockvar_mcmc: MCMC samples of the shock variances.
%     - output.shockskew_mcmc: MCMC samples of the shock skewnesses.
%     - output.shockkurt_mcmc: MCMC samples of the shock kurtoses.
%     - output.pred_mcmc: MCMC samples of the predicted densities.
%     - output.MontielOlseaStat_mcmc: MCMC samples of the Montiel-Olsea test statistics.
%     - output.MattesonTsayStat_mcmc: MCMC samples of the Matteson-Tsay test statistics
%       Note that the second row stores values for resampled shocks under
%       the hypothesis of mutually independence shocks.
%     - output.input: Copy of the input parameters.
% AUTHOR:
%   Robin Braun
%   robin.andreas.braun@gmail.com
%   Bank of England
%
% DATE:
%   04/05/2023
%-------------------------------------------------------------------------

%% Read input:
p = input.p; 
y = input.y;
nthin = input.nthin; 
ndraws = input.ndraws; 
nburn = input.nburn;    
alpha = input.pr.alpha;
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
% Initialize the DPMM parameters
Kal = zeros(T,n);
for j = 1:n 
    Kal(:,j)= randi(2,T,1); % Some random initializiation 
end
mut =  0.1*randn(T,n); mua = [mut, zeros(T,1)];
vt = sigma2 + 0.05*randn(T,n); vta = [vt,ones(T,1)];
pred_den = zeros(length(input.pr.domain),n); 
invSiga = sparse(1:T*(n+1),1:T*(n+1), 1./vta(:)) ;
Z = kron(speye(n),X); 
skew_pd = zeros(n,1); 
kurt_pd = zeros(n,1); 
% Define auxiliary quantities 
[i1, i2] = find(kron(ones(n+1),speye(T)));
J = eye(n+1); J(end,:) = []; 
J2 = eye(n+1); J2(1:4,:) = [];
Aconst = NaN(n+1); Aconst(A==1)=1; Aconst(A==0)=0; Aconst(A==-1)=-1;
accs = zeros(1,n+1); 
vA_mcmc = zeros((n+1)^2,ndraws);
vAplus_mcmc = zeros(n*(p*n),ndraws);
gamma_A_mcmc = zeros(length(gamma_A),ndraws); 
shockvar_mcmc = zeros(n,ndraws);
shockskew_mcmc = zeros(n,ndraws);
shockkurt_mcmc = zeros(n,ndraws); 
pred_mcmc = zeros(length(input.pr.domain),n,ndraws);
MattesonTsayStat_mcmc = zeros(2,ndraws); 
MontielOlseaStat_mcmc = zeros(2,ndraws);  
ntil = n + 1;
tic;
hwaitbar = waitbar(0,'Please wait...');
for irep = 1:ndraws*nthin+nburn
   waitbar(irep/(ndraws*nthin+nburn),hwaitbar);
    %% Step 1: Sample the VAR autoregressive parameters
    Ya = Y  -  (J*(A\[mut, zeros(T,1)]'))';
    vYa = vec(Ya);
    AAA = sparse(i1,i2,repmat(A,T,1)); 
    Precision  = AAA'*invSiga*AAA ; % Precision matrix of [u,ustar]
    Lam1 = Precision(1:T*n,1:T*n);
    Lam2 = Precision(T*n+1:end,T*n+1:end);
    Lam12 = Precision(T*n+1:end,1:T*n);
    iSigt = Lam1 - Lam12'*(Lam2\Lam12); % Computes the covariance of the observed errors [u]
    ZiS = Z'*iSigt;
    iV_p = ZiS*Z + input.pr.iV_vA;
    alphahat = iV_p\( ZiS*vYa + ipVpm );
    vAplus = alphahat + chol(iV_p,'lower')'\randn(length(alphahat),1);
    Aplus = reshape(vAplus,length(alphahat)/n,n);
    u = Y-X*Aplus; 
    
    %% Step 2: Sample USTAR
    AAA = sparse(i1,i2,repmat(A,T,1));
    Precision  = AAA'*invSiga*AAA ;
    Qcc = Precision(T*n+1:T*(n+1),T*n+1:T*(n+1));
    mul = (J2*(A\mua'))' - (Qcc\Precision(T*n+1:T*(n+1),1:T*n) )*vec(u-(J*(A\mua'))') ;
    U5star = mul + chol(Qcc,'lower')'\randn(T,1);
    Ustar = [u , U5star];
    
    %% Step 3: Sample the DPMM parameters
    estar = (A*Ustar')'; 
    for j = 1:n
        if and(irep>nburn,mod(irep,nthin)==0) 
            [Kal(:,j), alpha(j), mut(:,j) , vt(:,j), ~, sigma2(j), skew_pd(j), kurt_pd(j), pred_den(:,j)] = dpm_sampler_neal(estar(:,j),Kal(:,j),alpha(j),input.pr); 
        else  
            [Kal(:,j), alpha(j), mut(:,j) , vt(:,j) ] = dpm_sampler_neal(estar(:,j),Kal(:,j),alpha(j),input.pr);  
        end
    end   
    vta = [vt,ones(T,1)];
    mua = [mut, zeros(T,1)];
    invSiga = sparse(1:T*(n+1),1:T*(n+1), 1./vta(:)) ; 
    
    
    %% Step 4: Sample A via an independent MH step using Whaggoner Zha as proposal 
    for i = 1:n+1 % Sample row by row via Gibbs Sampler:
        % A*U = E; A*U = MU + Sig_12 *Etil
        Ai = A(i,:);
        [~,lprior] = input.pr.priorAi(Ai,i);
        Gi = eye(n+1); Gi = Gi(:,isnan(Aconst(i,:)));
        s_a = zeros(n+1,1); s_a(Aconst(i,:)==1)=1; s_a(Aconst(i,:)==-1)=-1;
        if sum( or(Aconst(i,:)==1, Aconst(i,:)==-1) )==0
            zi =  mua(:,i);
        else
            zi =  mua(:,i) - Ustar*s_a;
        end
        iSig = sparse(1:T,1:T,1./vta(:,i));
        muphi = (Gi'*Ustar'*iSig*Ustar*Gi)\(Gi'*Ustar'*iSig*zi);
        Ri = chol(T*inv(Gi'*Ustar'*iSig*Ustar*Gi),'lower');
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
        shockskew_mcmc(:, (irep-nburn)/nthin) = skew_pd;
        shockkurt_mcmc(:, (irep-nburn)/nthin) = kurt_pd;   
        pred_mcmc(:,:,(irep-nburn)/nthin) = pred_den;   
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
output.shockskew_mcmc = shockskew_mcmc;
output.shockkurt_mcmc = shockkurt_mcmc;
output.pred_mcmc = pred_mcmc;
output.MontielOlseaStat_mcmc = MontielOlseaStat_mcmc;
output.MattesonTsayStat_mcmc = MattesonTsayStat_mcmc;
output.input = input; % Also save the input  

end

        