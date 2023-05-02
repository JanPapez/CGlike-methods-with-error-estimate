%% How to run scripts in CGlike-methods-with-error-estimate
%   GitHub repository 
%
% Jan Papez, Petr Tichy, May 2023
%       https://github.com/JanPapez/CGlike-methods-with-error-estimate


% Here we run the algorithms on a simple example with zero initial guess 
%   and no preconditioner


%% Least-squares problem
m = 1000; n = 800; maxit = 100;
[U,~] = qr(rand(m)); [V,~] = qr(rand(n));
sing_values = [0.99.^(1:30)  1e2 + linspace(0,1e1,n-30)]';
A = U*spdiags(sing_values,0,m,n)*V';

x = ones(size(A,2),1);
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
A = A';

x = ones(size(A,2),1);
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

subplot(3,1,1:2)
semilogy(results.reconstructed_conv_curve, 'r-', 'LineWidth', 2), hold on
semilogy(results.estim_history, 'b-', 'LineWidth', 1.1), 
semilogy(results.ell, results.estim_error_of_xl, 'b', 'Marker', 'o'), hold off
legend('Reconstructed convergence curve','Error estimate')

subplot(3,1,3)
plot(results.delay_history, 'b-', 'LineWidth', 1.1)
legend('adaptive delay')
