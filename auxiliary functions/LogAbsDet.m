function x = LogAbsDet(X)
%------------------------------------------------------------------------- 
% DESCRIPTION:
%   This function computes the log of the absolute value of the determinant
%   of the square matrix X based on the LU decomposition  
% INPUTS:
%   - X: The square matrix for which the log absolute determinant is computed. 
% OUTPUTS:
%   - x: The log absolute determinant of X.  
%------------------------------------------------------------------------- 
[~,U,~]=lu(X);
n=size(U,1);
x=0.0;
for i=1:n
    if U(i,i) == 0.0
        x=-inf;
        return;
    end
    x=x+log(abs(U(i,i)));
end
    