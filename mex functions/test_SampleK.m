clear; clc;
% mex sampleK_mex.cpp
rng(100)

T = 500; n = 5; 
Kal =  randi(2,T,n); % Some random initializiation 
estar = randn(T,n);
psr = rand(T,n);
tau = 1; m = 0; s = 2; S = 2; alpha = 0.3;
nrep = 1000;
tic;
for j = 1:nrep
    for i = 1:n
        [Kal(:,i),SumX,SumX2,n_ii] = sampleK_matlab(Kal(:,i),estar(:,i),psr(:,i),m,tau,s,S,alpha);
    end
end 
toc;
tic;
for j = 1:nrep 
    for i = 1:n
    [Kalout,SumX,SumX2,n_ii] = sampleK_mex(Kal(:,i),estar(:,i),psr(:,i),m,tau,s,S,alpha);
    end
end
toc; 
