function ll = predictive_log_likelihood_approx(model, data)
    Xt = zeros(size(data.test_ind,1), sum(model.dim));
    st = 0;
    for k=1:model.nmod
        Xt(:,st + 1:st + model.dim(k)) = model.U{k}(data.test_ind(:,k),:);
        st= st + model.dim(k);
    end
    Ktm = ker_cross(Xt, model.Um, model.ker_param);
    %expt. w.r.t posterior dist. 
    lam = exp(Ktm*(model.Kmm\model.mu));
    T = data.test_T;
    y = data.test_vals;
    ll = sum(y.*log(lam) - lam*T);
end