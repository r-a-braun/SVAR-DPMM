function vA = vec(A) 
% DESCRIPTION:
%   This function vectorizes a matrix by stacking its columns into a single
%   column vector.
%
% INPUTS:
%   - A: The matrix to be vectorized.
%
% OUTPUTS:
%   - vA: The vectorized form of the matrix A.
vA = A(:);
end

