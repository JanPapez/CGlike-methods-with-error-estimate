%% How to run scripts in CGlike-methods-with-error-estimate
%   GitHub repository 
%
% Jan Papez, Petr Tichy, May 2023
%       https://github.com/JanPapez/CGlike-methods-with-error-estimate


% Here we run the algorithms on a simple examples with random matrix, 
%   zero initial guess and no preconditioner


%% Least-squares problem
m = 1000; n = 800; maxit = 100;

A = rand(m, n);

x = ones(size(A,2),1);
x(2:2:end) = -2;
x(5:5:end) = 0;

b_consistent = A*x;
b = b_consistent + randn(size(b_consistent))*norm(b_consistent);

disp('*** running CGLS');
[x_CGLS,results] = pcgls_withestimate(A,b,maxit); 
fprintf('*** Summary: error %e estimated at iteration %d with delay %d \n\n', ...
    results.estim_error_of_xl, results.ell, results.d);


disp('*** running LSQR');
[x_LSQR,results] = plsqr_withestimate(A,b,maxit);
fprintf('*** Summary: error %e estimated at iteration %d with delay %d \n\n', ...
    results.estim_error_of_xl, results.ell, results.d);

%% Least-norm problem
m = 800; n = 1000; 

A = rand(m, n);

x = ones(size(A,2),1);
x(2:2:end) = -2;
x(5:5:end) = 0;

b = A*x;

disp('*** running CGNE');
[x_CGNE,results] = pcgne_withestimate(A,b,maxit);
fprintf('*** Summary: error %e estimated at iteration %d with delay %d \n\n', ...
    results.estim_error_of_xl, results.ell, results.d);


disp('*** running CRAIG');
[x_CRAIG,results] = pcraig_bidiag_withestimate(A,b,maxit);
fprintf('*** Summary: error %e estimated at iteration %d with delay %d \n\n', ...
    results.estim_error_of_xl, results.ell, results.d);

%% plotting 

% reconstructed convergence curve is an approximation to real convergence 
%   curve of the method and we plot it together with the error estimate

figure(1);
semilogy(results.reconstructed_conv_curve, 'r-', 'LineWidth', 2), hold on
semilogy(results.estim_history, 'b-', 'LineWidth', 1.5), 
semilogy(results.ell, results.estim_error_of_xl, 'b', 'Marker', 'o'), hold off
legend('Reconstructed convergence curve','Error estimate')
