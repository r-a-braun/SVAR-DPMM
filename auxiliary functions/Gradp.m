function g=Gradp(f,x0,varargin)
%-------------------------------------------------------------------------
% DESCRIPTION:
%   This function computes the gradient of function f evaluated at x0 using
%   forward gradients. It adjusts for possible differently scaled x0 by
%   taking percentage increments. The function f should return either a scalar
%   or a column vector. The input x0 should be a column vector of parameters. 
% INPUTS:
%   - f: The function handle for which the gradient is computed.
%   - x0: The column vector of parameters at which the gradient is evaluated.
%   - varargin: Additional arguments to be passed to the function f.
%
% OUTPUTS:
%   - g: The gradient of the function f evaluated at x0.
%-------------------------------------------------------------------------

f0=feval(f,x0,varargin{:}); 
[T,c]=size(f0);

if size(x0,2)>size(x0,1)
    x0=x0';
end
k=size(x0,1); % number of parameters wrt which one should compute gradient

h=0.0000001; %some small number

g=zeros(T,k); %will contain the gradient
e=eye(k); 
for j=1:k 
    if x0(j)>1 
        f1=feval(f,(x0.*( ones(k,1) +  e(:,j) *h )),varargin{:});    
        g(:,j)=(f1-f0)/(x0(j)*h);    
    else
        f1=feval(f, x0 +  e(:,j) *h ,varargin{:});    
        g(:,j)=(f1-f0)/h;    
    
    end
    
end
