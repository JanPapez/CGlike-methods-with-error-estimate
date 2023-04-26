function [x,results] = pcgne_adaptdd(A,b,maxit,x,L)
%% preconditioned CGNE method (variant of Craig's method) with adaptive 
%   error estimate for   || x - x_k ||
%   described in detail in [Papez, Tichy: Estimating the error in CG-like 
%   algorithms for least-squares and least-norm problems, 2023]
%
% Jan Papez, Petr Tichy, May 2023
%       https://github.com/JanPapez/CGlike-methods-with-error-estimate

if nargin < 5
    L = speye(size(A,2));
    if nargin < 4
        x = zeros(size(A,2),1);
    end
end

% initialization
r = L\(b - A*x);
p = A'*(L'\r);
gamma_new = norm(r)^2;

es = adaptive(0);

% iteration

for k = 1:maxit
    gamma = gamma_new;
    alpha = gamma/(norm(p)^2);
    x = x + alpha*p;
    r = r - alpha*(L\(A*p));
    gamma_new = norm(r)^2;
    beta = gamma_new / gamma;
    p = A'*(L'\r) + beta*p;
    
    % adaptive
    es = adaptive(k, alpha * gamma, es);
    
    % use es.estim(end) for stopping criterion;
    % error at length(es.estim) iteration estimated using es.delay(end) 
    % additional iterations
end

%% output

% index of the last CG iteration, associated with the computed approximation x
    results.k = k;
% index of last CG iteration with accepted error estimate
    results.ell = length(es.estim);
% estimated error at the ell-th CG iteration   
    results.estim_error_of_xl = es.estim(end);
% adaptively chosen delay for estimating the error at the ell-th iteration 
    results.d = es.delay(end);

% if needed, user can get more information about the convergence:
results.reconstructed_conv_curve = es.curve;
results.estim_history   = es.estim;
results.delay_history   = es.delay;

end
