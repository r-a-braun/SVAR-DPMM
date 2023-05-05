function [intlike, R] = SIS_DPM(e,alpha,prior,R)
 
%-------------------------------------------------------------------------  
% DESCRIPTION:
%   This function performs Collapsed Sampling Importance Resampling (SIS)
%   to evaluate the likelihood of a Gaussian Dirichlet Process Mixture
%   model using a conjugate base distribution. See Basu & Chib (2003)   
% 
% INPUTS:
%   - e: The input data.
%   - alpha: The concentration parameter of the Dirichlet Process.
%   - prior: The prior distribution parameters.
%   - R: The number of repetitions (optional).
%
% OUTPUTS:
%   - intlike: The integrated likelihood estimate.
%   - R: The updated number of repetitions.
%
% AUTHOR:
%   Robin Braun
%   robin.andreas.braun@gmail.com
%   Bank of England
%
% DATE:
%   04/05/2023
%
%-------------------------------------------------------------------------


 
s = prior.s;
S = prior.S;
V = prior.tau;
m = prior.m; 
T = length(e); 
if nargin < 4
    R = 10; % Start with 10, if variance < 1/N, add more loops
end
Kal = NaN(T,R); u = NaN(T,R); 
% Initialize
logu1 = log_pred_dens_nig(e(1),m,V,s,S); 
u(1,:) = exp(logu1);
Kal(1,:) = 1; 
k = 1*ones(1,R) ; n_i = zeros(T,R); n_i(1,:) = 1;
Vn = V*ones(T,R); mn = m*ones(T,R); an = s*ones(T,R); bn = S*ones(T,R);
Vn(1,:) = 1./( 1./(V) + n_i(1,:) );
mn(1,:)= Vn(1,:).* (m./(V) + e(1) );
an(1,:) = s + n_i(1,:)./2;
bn(1,:) = S + 0.5*( m.^2./V + e(1).^2 - mn(1,:).^2./Vn(1,:));
logint1 = log_pred_dens_nig(e,m,V,s,S); 
int1 = exp(logint1);
Urands = rand(T,R); 
% Loop over the e's
for i = 2:T 
   % Step 1. Compute the predictive probability  
   p1 = alpha/(alpha+i-1)*int1(i); 
   logintk2 = log_pred_dens_nig(e(i),mn(1:max(k),:),Vn(1:max(k),:),an(1:max(k),:),bn(1:max(k),:)) ;
   intk2 = exp(logintk2) ;
   u(i,:) =  p1 + sum((n_i(1:max(k),:)./(alpha+i-1)).*intk2,1); 
   pi_ki = [n_i(1:max(k),:).*intk2; ones(1,R)*alpha*int1(i)];
   pi_ki = pi_ki./sum(pi_ki,1);
   Kali = sum(cumsum(pi_ki) < Urands(i,:)) + 1;
   Kali(Kali>k+1)=k(Kali>k+1)+1; % Those need correction for the artificial 0's;
   Kal(i,:) = Kali; 
   k = k + (Kal(i,:) ==  k+1);
   for j = 1:R 
       n_i(Kal(i,j),j) = n_i(Kal(i,j),j) + 1; 
       Vp = Vn(Kal(i,j),j); mp = mn(Kal(i,j),j); ap = an(Kal(i,j),j); bp = bn(Kal(i,j),j);
       Vn(Kal(i,j),j) = 1./( 1./Vp + 1 );
       mn(Kal(i,j),j) = Vn(Kal(i,j),j).* (mp./Vp + e(i) );
       an(Kal(i,j),j) = ap + 1./2;
       bn(Kal(i,j),j) = bp + 0.5*( mp.^2./Vp + e(i).^2 - mn(Kal(i,j),j).^2./Vn(Kal(i,j),j)); 
   end 
end 
store_llike = sum(log(u),1);   



if nargin < 4 % If R is unkown, choose it adaptively such that variance < 1
    var_llike = var(store_llike)/R;
    if var_llike > 1/prior.n
        R2 = floor(var_llike)*prior.n*R;
        %% Start with the second round:
        Kal = NaN(T,R2); u = NaN(T,R2);
        logu1 = log_pred_dens_nig(e(1),m,V,s,S);
        u(1,:) = exp(logu1);
        Kal(1,:) = 1;
        k = 1*ones(1,R2) ; n_i = zeros(T,R2); n_i(1,:) = 1;
        Vn = V*ones(T,R2); mn = m*ones(T,R2); an = s*ones(T,R2); bn = S*ones(T,R2);
        Vn(1,:) = 1./( 1./(V) + n_i(1,:) );
        mn(1,:)= Vn(1,:).* (m./(V) + e(1) );
        an(1,:) = s + n_i(1,:)./2;
        bn(1,:) = S + 0.5*( m.^2./V + e(1).^2 - mn(1,:).^2./Vn(1,:));
        logint1 = log_pred_dens_nig(e,m,V,s,S);
        int1 = exp(logint1);
        Urands = rand(T,R2);
        % Loop over the e's
        for i = 2:T
            % Step 1. Compute the predictive probability
            p1 = alpha/(alpha+i-1)*int1(i);
            logintk2 = log_pred_dens_nig(e(i),mn(1:max(k),:),Vn(1:max(k),:),an(1:max(k),:),bn(1:max(k),:)) ;
            intk2 = exp(logintk2) ;
            u(i,:) =  p1 + sum((n_i(1:max(k),:)./(alpha+i-1)).*intk2,1);
            pi_ki = [n_i(1:max(k),:).*intk2; ones(1,R2)*alpha*int1(i)];
            pi_ki = pi_ki./sum(pi_ki,1);
            Kali = sum(cumsum(pi_ki) < Urands(i,:)) + 1;
            Kali(Kali>k+1)=k(Kali>k+1)+1; % Those need correction for the artificial 0's;
            Kal(i,:) = Kali;
            k = k + (Kal(i,:) ==  k+1);
            for j = 1:R2
                n_i(Kal(i,j),j) = n_i(Kal(i,j),j) + 1;
                Vp = Vn(Kal(i,j),j); mp = mn(Kal(i,j),j); ap = an(Kal(i,j),j); bp = bn(Kal(i,j),j);
                Vn(Kal(i,j),j) = 1./( 1./Vp + 1 );
                mn(Kal(i,j),j) = Vn(Kal(i,j),j).* (mp./Vp + e(i) );
                an(Kal(i,j),j) = ap + 1./2;
                bn(Kal(i,j),j) = bp + 0.5*( mp.^2./Vp + e(i).^2 - mn(Kal(i,j),j).^2./Vn(Kal(i,j),j));
            end
        end
        store_llike2 = sum(log(u),1);
        store_llike = [store_llike, store_llike2];
        R = R + R2;
    end
end

maxllike = max(store_llike);
intlike = log(mean(exp(store_llike-maxllike))) + maxllike;


 
end


 
