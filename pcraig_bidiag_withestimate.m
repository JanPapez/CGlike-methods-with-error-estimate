function [x,results] = pcraig_bidiag_withestimate(A,b,maxit,L)
%% (preconditioned) CRAIG method with adaptive error estimate for
%   || x - x_k ||
%   described in detail in [Papez, Tichy: Estimating the error in CG-like 
%   algorithms for least-squares and least-norm problems, 2023]
%
% Jan Papez, Petr Tichy, May 2023
%       https://github.com/JanPapez/CGlike-methods-with-error-estimate

if nargin < 4
    L = speye(size(A,1));
end

% initialization
x      = zeros(size(A,2),1);

z      = -1;
b      = L\b;
beta   = norm(b);
u      = (1/beta)*b;
v      = A'*(L'\u);
alfa   = norm(v);
v      = (1/alfa)*v;

es = adaptive(0);

% iteration

for k = 1:maxit
    z = -(beta/alfa)*z;
    x =   x + z*v;
    
    u = L\(A*v) - alfa*u;
    beta = norm(u);
    if beta > 0, u = (1/beta)*u; end
    
    v = A'*(L'\u) - beta*v;
    alfa = norm(v);
    if alfa > 0, v  = (1/alfa)*v; end

    % adaptive
    es = adaptive(k, z^2, es);
    
    % use es.estim(end) for stopping criterion;
    % error at length(es.estim) iteration estimated using es.delay(end) 
    % additional iterations
end

%% output

% index of the last CRAIG iteration, associated with the computed approximation x
    results.k = k;
% index of last iteration with accepted error estimate
    results.ell = length(es.estim);
% estimated error at the ell-th iteration   
    results.estim_error_of_xl = es.estim(end);
% adaptively chosen delay for estimating the error at the ell-th iteration 
    results.d = es.delay(end);

% if needed, user can get more information about the convergence:
results.reconstructed_conv_curve = sqrt(es.curve);
results.estim_history   = es.estim;
results.delay_history   = es.delay;

end
