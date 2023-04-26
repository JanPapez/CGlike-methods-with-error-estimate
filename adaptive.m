function es = adaptive(k, add, es)
%% function for evaluating the adaptive error estimate in CG and CG-like 
%   methods, in a form independent on the chosen method
%   described in detail in [Papez, Tichy: Estimating the error in CG-like 
%   algorithms for least-squares and least-norm problems, 2023]
%
% Jan Papez, Petr Tichy, May 2023
%       https://github.com/JanPapez/CGlike-methods-with-error-estimate

verbatim = true;   % print info?

% initialization
if k == 0

    es.ell = 1;
    es.d = 0;
    es.tau = 0.25;
    es.TOL = 1e-4;
    
    es.delay = [];
    es.Delta = [];
    es.curve = [];
    es.estim = [];
    es.Deltadd = [];
    
else
    ell = es.ell;
    d = es.d;
    
    es.Delta(k) = add;
    es.Deltadd(k) = 0;
    es.curve(k) = 0;
    es.curve = es.curve + es.Delta(k);
    
    index = max(k,1);
    es.Deltadd(index:k) = es.Deltadd(index:k) + add;
   
    if k > 1                      % ... adaptive choice of d
        S = findS(es.curve,es.Deltadd,ell,es.TOL);
        
        num = S * es.Delta(k);
        den = sum(es.Delta(ell:k-1));
        
        while ((d >= 0) && (num/den <= es.tau))
            es.delay(ell) = d;
            es.estim(ell) = sqrt(den + es.Delta(k));
            if verbatim
                fprintf('Iteration %d: estimated error = %f \n', ell, es.estim(ell));
            end
            ell = ell + 1; d = d - 1;
            den = sum(es.Delta(ell:k-1));
        end
        d = d + 1;
    end
    es.d = d;
    es.ell = ell;
    
end

function [S] = findS(curve,Delta,ell,TOL) 
%% ... find the safety factor S using the tolerance 1e-4 

ind = find((curve(ell)./curve) <= TOL, 1, 'last');
if isempty(ind), ind = 1; end

S = max(curve(ind:end-1)./Delta(ind:end-1));
end

end