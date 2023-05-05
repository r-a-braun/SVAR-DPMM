function A = getA_DPM( x )
%-------------------------------------------------------------------------
% DESCRIPTION:
% This function constructs the A matrix of the Baumeister and Hamilton model. 
% INPUT:
%   - x: Parameter vector of size (8, 1) containing the following elements:
%        - x(1): a_qp
%        - x(2): a_yp
%        - x(3): b_qy
%        - x(4): b_qp
%        - x(5): psi_1
%        - x(6): psi_3
%        - x(7): chi
%        - x(8): rhostar
%
%   AUTHOR:
%   Robin Braun
%   robin.braun@bankofengland.co.uk
%   Bank of England
%
%   DATE:
%   04/05/2023
%-------------------------------------------------------------------------

a_qp  = x(1);
a_yp  = x(2);
b_qy  = x(3);
b_qp  = x(4);
psi_1 = x(5);
psi_3 = x(6);
chi = x(7);
rhostar = x(8);  
a1 = 1./sqrt( rhostar/(1-rhostar)*chi^(-2)*1.3 );
a2 = -chi./sqrt( rhostar/(1-rhostar)*chi^(-2)*1.3 );
A =  [1,     0,    -a_qp,       0,          0;...
    0,      1,    -a_yp,       0,          0;...
    1,  -b_qy,    -b_qp,       0,         -1;...
    -psi_1,     0 ,   -psi_3,       0,          1;...
    0,       0,        0,           a1,        a2]; 
end

