%---data----
%indices for training
%data.train_subs
%data.y_subs;
%data.tensor_sz;
%data.T, the time period
%---model---
%model.nvec: the latent dimension for each mode
%CP tensor facotrization,each entry is a poisson point process
%use log_evidence_lower_bound_batch_robust to avoid divdied by 0 case
%the last factors are time factors
%testing more than one datasets
function [model, test_ll] = CPMarkovPP_online_robust_more(data, model)
    %initialization    
    model = do_init(model, data);
    %assemble
    nmod = model.nmod;
    x = [];
    for k=1:nmod
         x = [x; vec(model.U{k})];
    end
    logs = log(1);
    x = [x;logs];
    n = size(model.subs,1);
    decay = model.decay;
    batch_size = model.batch_size;
    epsi = 1e-5;
    step_rate = 1;
    sms = zeros(length(x), 1);
    gms = zeros(length(x), 1);
    samples = randperm(size(model.subs,1));
    epoch = model.epoch;
    test_ll = zeros(epoch,2);
    for j=1:epoch
        st = 1;
        while st < n
            to_sel = samples(st:min(st+batch_size-1, n));
            grad = log_evidence_lower_bound_batch_robust(x, model, to_sel);
            gms = decay * gms + (1 - decay) * grad.^2;
            step = sqrt(sms + epsi) ./ sqrt(gms + epsi) .* grad * step_rate;
            x = x + step;
            sms = decay * sms + (1 - decay) * step.^2;
            n_batch = ceil(st/batch_size);
            if mod(n_batch, 5) == 0
                fprintf('epoch = %d, batch %d\n', j, n_batch);
            end
            st = st + batch_size;
        end
        %test performance per epoch
        %deassemble
        st = 0;
        for k=1:nmod
            model.U{k} = reshape(x(st + 1 : st + model.nvec(k)*model.R), model.nvec(k), model.R);
            st = st + model.nvec(k)*model.R;
        end
        model.s = exp(x(end));
        [ll_mean, ll_std] = predictive_log_likelihood_more(model, data);
        test_ll(j,:) = [ll_mean, ll_std];
    end
    
    
    %deassemble
%     st = 0;
%     for k=1:nmod
%         model.U{k} = reshape(x(st + 1 : st + model.nvec(k)*model.R), model.nvec(k), model.R);
%         st = st + model.nvec(k)*model.R;
%     end
    
end