function ll = poisson_log_likelihood(lam, y)
    lam = lam + 1e-10; 
    ll = sum(y.*log(lam) - lam - gammaln(y));
end