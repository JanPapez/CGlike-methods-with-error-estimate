function [x,results] = plsqr_adaptdd(A,b,maxit,x,L)
%% preconditioned LSQR method with adaptive error estimate for
%   || x - x_k ||_{A^T A}
%   described in detail in [Papez, Tichy: Estimating the error in CG-like 
%   algorithms for least-squares and least-norm problems, 2023]
%
% Jan Papez, Petr Tichy, May 2023
%       https://github.com/JanPapez/CGlike-methods-with-error-estimate

if nargin < 5
    L = speye(size(A,1));
    if nargin < 4
        x = zeros(size(A,2),1);
    end
end

LT = L';

% initialization
u = b - A*x; beta = norm(u);
u = u/beta;

v = L\(A'*u); alpha = norm(v);
v = v/alpha;

w = v;
phibar = beta;
rhobar = alpha;
 
es = adaptive(0);

% iteration

for k = 1:maxit
    
    u = A*(LT\v) - alpha*u;
    beta = norm(u);
    u = u/beta;
    
    v = L\(A'*u) - beta*v;
    alpha = norm(v);
    v = v/alpha;
    
    rho = sqrt(rhobar^2 + beta^2);
    c = rhobar/rho;
    s = beta/rho;
    theta = s*alpha; rhobar = -c*alpha;
    phi = c*phibar; phibar = s*phibar;
    
    x = x + (phi/rho)*w;
    w = v - (theta/rho)*w;
    
    % adaptive part
    es = adaptive(k, phi^2, es);
    
    % use es.estim(end) for stopping criterion;
    % error at length(es.estim) iteration estimated using es.delay(end) 
    % additional iterations
end

x = L'\x;

%% output

% index of the last LSQR iteration, associated with the computed approximation x
    results.k = k;
% index of last iteration with accepted error estimate
    results.ell = length(es.estim);
% estimated error at the ell-th iteration   
    results.estim_error_of_xl = es.estim(end);
% adaptively chosen delay for estimating the error at the ell-th iteration 
    results.d = es.delay(end);

% if needed, user can get more information about the convergence:
results.reconstructed_conv_curve = es.curve;
results.estim_history   = es.estim;
results.delay_history   = es.delay;

end
