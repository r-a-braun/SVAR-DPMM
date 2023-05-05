function results = GewekeRNE(draws) 
% EDITED FUNCTION "momentg.m" from James P. LeSage Toolbox.
% Computes the RNE for the gibbs sampler output.
% Spectral Denstiy estimate based on 4% Tapared Window (as in Primiceri, 2005) 
% PURPOSE: computes Gewke's convergence diagnostics NSE and RNE
%          (numerical std error and relative numerical efficiencies)
% -----------------------------------------------------------------
% USAGE: result = momentg(draws)
% where: draws = a matrix of Gibbs draws (ndraws x nvars)
% -----------------------------------------------------------------
% RETURNS: a structure result:
%          result(i).rne1  = rne using 4% autocovariance taper

% SEE ALSO: coda(), apm()
% -----------------------------------------------------------------
% REFERENCES: Geweke (1992), `Evaluating the accuracy of sampling-based
% approaches to the calculation of posterior moments', in J.O. Berger,
% J.M. Bernardo, A.P. Dawid, and A.F.M. Smith (eds.) Proceedings of
% the Fourth Valencia International Meeting on Bayesian Statistics,
% pp. 169-194, Oxford University Press
% Also: `Using simulation methods for Bayesian econometric models:
% Inference, development and communication', at: www.econ.umn.edu/~bacc
% -----------------------------------------------------------------

% written by:
% James P. LeSage, Dept of Economics
% University of Toledo
% 2801 W. Bancroft St,
% Toledo, OH 43606
% jpl@jpl.econ.utoledo.edu

% NOTE: this code draws heavily on MATLAB programs written by
% Siddartha Chib available at: www.econ.umn.edu/~bacc
% I have repackaged it to make it easier to use.

[ndraw , nvar] = size(draws);

NG=100;


ntaper=4;
ns = floor(ndraw/NG);
nuse = ns*NG;

results=zeros(nvar,1);
for jf = 1:nvar 
    cnt = 0;
    cn = zeros(NG);
    cd = zeros(NG);
    cdn = zeros(NG);
    cdd = zeros(NG);
    cnn = zeros(NG);
    cvar = zeros(NG);
    
    % form sufficiency statistics needed below
    td=0; tn=0; tdd=0; tnn=0; tdn=0; tvar=0;
    for ig=1:NG 
        gd=0; gn=0; gdd=0; gdn=0; gnn=0; gvar=0;
        for is = 1:ns 
            cnt = cnt + 1;
            g = draws(cnt,jf);
            ad = 1;
            an = ad*g;
            gd = gd+ad;
            gn = gn+an;
            gdn = gdn + ad*an;
            gdd = gdd + ad*ad;
            gnn = gnn + an*an;
            gvar = gvar + an*g;
        end 
        td = td+gd;
        tn = tn+gn;
        tdn = tdn+gdn;
        tdd = tdd+gdd;
        tnn=tnn+gnn;
        tvar=tvar+gvar;
        
        cn(ig)=gn/ns;
        cd(ig)=gd/ns;
        cdn(ig)=gdn/ns;
        cdd(ig)=gdd/ns;
        cnn(ig)=gnn/ns;
        cvar(ig)=gvar/ns;
    end 
    
    eg = tn/td;
    varg = tvar/td - eg^2;
 

    %get autocovariance of grouped means
    barn=tn/nuse;
    bard=td/nuse;
    for ig=1:NG 
        cn(ig)=cn(ig)-barn;
        cd(ig)=cd(ig)-bard;
    end 
    rnn=zeros(1,NG);
    rdd=zeros(1,NG);
    rnd=zeros(1,NG);
    rdn=zeros(1,NG);
    for lag=0:NG-1 
        ann=0; add=0; and=0; adn=0;
        for ig=lag+1:NG 
            ann=ann+cn(ig)*cn(ig-lag);
            add=add+cd(ig)*cd(ig-lag);
            and=and+cn(ig)*cd(ig-lag);
            adn=adn+cd(ig)*cd(ig-lag);
        end 
        % index 0 not allowed, lag+1 stands for lag
        rnn(lag+1)=ann/NG;
        rdd(lag+1)=add/NG;
        rnd(lag+1)=and/NG;
        rdn(lag+1)=adn/NG;
    end 
    
    % numerical standard error with tapered autocovariance functions
    m=ntaper;
    am=m;
    snn=rnn(1); sdd=rdd(1); snd=rnd(1);
    for lag=1:m-1 
        att=1-lag/am;
        snn=snn+2*att*rnn(lag+1);
        sdd=sdd+2*att*rdd(lag+1);
        snd=snd+att*(rnd(lag+1) + rnd(lag+1));
    end 
    varnum=ns*nuse*(snn-2*eg*snd+sdd*eg^2)/(td^2);

    
    % save results in structure
    results(jf) = varg/(nuse*varnum);
    
    
end 

