function [ Ai, llh1, llh2 ] = propose_Ai( A, i , muphii, Ri, Gi, s_a, T)
%-------------------------------------------------------------------------
% [Ai, llh1, llh2] = propose_Ai(A, i, muphii, Ri, Gi, s_a, T)
%
% DESCRIPTION:
%   This function draws from the distribution of the i-th row of the
%   imapct matrix based on a uniform prior
%   It implements the algorithm described in  Appendix A of the paper
% INPUTS: 
%   - A: The current parameter values.
%   - i: the index of the row of A  
%   - muphii: The value of muphi at index i. (See paper)
%   - Ri: The value of Ri. (See paper)
%   - Gi: The value of Gi. (See paper)
%   - s_a: The value of s_a. (See paper)
%   - T: The value of T. (See paper)
%
% OUTPUTS:
%   - Ai: The proposed parameter values.
%   - llh1: The log-likelihood.
%   - llh2: The two mixture approximation of the log-likelihood
%
% AUTHOR:
%   Robin Braun
%   robin.andreas.braun@gmail.com
%   Bank of England
%
% DATE:
%   04/05/2023
%-------------------------------------------------------------------------
 

Ami = A; Ami(i,:)=[];
v1 = Ri'*Gi'*null(Ami); v1 = v1./sqrt(v1'*v1);
v = [ v1, null(v1') ]; 
xi = zeros(size(Gi,2),1);
xihat = zeros(size(Gi,2),1);
llh1 = 0; llh2 = 0;
for j = 1:size(Gi,2)
    xihat(j) = muphii'*(Ri'\v(:,j));
    if j == 1
        a = det([A(1:i-1,:)', s_a , A(i+1:end,:)'] );
        b = det([A(1:i-1,:)', Gi*(Ri*v(:,1)) , A(i+1:end,:)'] );
        muj = xihat(j); rho = 1/T;
        mus=[1/(2*b)*( (b*muj-a)+sqrt( (a-b*muj)^2 - 4*b*(-a*muj-b)));...
            1/(2*b)*( (b*muj-a)-sqrt( (a-b*muj)^2 - 4*b*(-a*muj-b)))];
        sig2s = -1./[T*(-(b^2)/((a+b*mus(1))^2)-1);...
            T*(-(b^2)/((a+b*mus(2))^2)-1)]; 
        lfd = @(x) (T.*log(abs(a+x.*b)) - T/2*(x-muj).^2)  ;
        dvals = [lfd(mus(1)),lfd(mus(2))]; dvals = dvals-max(dvals);
        dvals = exp(dvals)./sum(exp(dvals));
        w = dvals(1);
        if rand<=w
            xi(j) =  mus(1) + sqrt(sig2s(1))*randn;
        else
            xi(j) =  mus(2) + sqrt(sig2s(2))*randn;
        end
        llh1 = llh1 + T*log(abs(a+xi(1).*b)) - T/2*(xi(1)-muj).^2;
        llh2 = llh2 + log(w.*normpdf(xi(1),mus(1),sqrt(sig2s(1)))+(1-w)*normpdf(xi(1),mus(2),sqrt(sig2s(2)))); 
    else
        xi(j) = xihat(j) + sqrt(1/T)*randn;
        llh1 = llh1 + (- T/2*(xi(j)-xihat(j)).^2);
        llh2 = llh2 + (- T/2*(xi(j)-xihat(j)).^2);
    end
end
Ai = s_a + Gi*Ri*sum(xi'.*v,2);
end

