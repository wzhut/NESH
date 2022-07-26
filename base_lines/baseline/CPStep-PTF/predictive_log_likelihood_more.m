function [ll_mean, ll_std] = predictive_log_likelihood_more(model, data)
    n_test_data = length(data.test);
    ll = zeros(n_test_data,1);
    for t=1:n_test_data
        cur = data.test{t};
        ntest = size(cur.test_ind,1);
        M = ones(ntest, model.R);
        for k=1:model.nmod
            M = M.*model.U{k}(cur.test_ind(:,k),:);
        end
        lam = exp(sum(M,2));
        T = cur.test_T;
        y = cur.test_vals;
        ll(t) = sum(y.*log(lam) - lam*T);
    end
    ll_mean = mean(ll);
    ll_std = std(ll)/sqrt(n_test_data);
end