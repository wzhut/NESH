
clear all;
close all;
rng('default');

addpath(genpath('../util'));
addpath(genpath('../tensor_toolbox_2.6'));
addpath(genpath('../lightspeed'));
addpath(genpath('../minFunc_2012'));
% addpath('../');

% load('../mat/crash_2015.mat');

% y_max = max(train_y);
% train_y = train_y / y_max;
% test_y = test_y / y_max;

R_list = [2, 5, 8, 10];
save_file = ['retail_res.mat'];
for i =1:4
    R = R_list(i);
    ll_list = [];
    for f = 1:5
        file = ['../mat/retail_f_', num2str(f),'.mat'];
        load(file);
        data.ind = double(train_ind) + 1;
        data.e = train_y;
        % nvec = double(max([train_ind + 1; test_ind + 1]));
        nvec = max(double([train_ind; test_ind])) + 1;
%         nvec = max(data.ind);
        nmod = size(nvec,2);
        
        data.e_count = sptensor(data.ind(1,:), 1, nvec);
        for n=2:size(data.ind,1)
            sub = data.ind(n,:);
            data.e_count(sub) = data.e_count(sub) + 1;
        end
        data.T = max(data.e);

        data.train_subs = find(data.e_count);
        data.y_subs = data.e_count(data.train_subs);
        data.tensor_sz = nvec;

        % test = load('../data/ufo-test.mat');
        test.ind = double(test_ind) + 1;
        test.e = test_y;
        % test = test.data;
        test.e_count = sptensor(test.ind(1,:), 1, nvec);
        for n=2:size(test.ind,1)
            sub = test.ind(n,:);
            test.e_count(sub) = test.e_count(sub) + 1;
        end
        test.T = max(test.e)-min(test.e);
        data.test_ind = find(test.e_count);
        data.test_vals = test.e_count(data.test_ind);
        data.test_T = test.T;

        %GP tensor poission process regression
        %parameters setting
        model = [];
        %model.nmod = nmod;
        %the length of latent features
        %cp_pp_model = load('../CP-PP/cp-pp-model.mat');
        model.R = 1;
        for k=1:nmod
            model.U{k} = rand(nvec(k), model.R);
            %model.U{k} = 0.1*randn(nvec(k), model.R);
            %model.U{k} = cp_pp_model.model.U{k};
        end
        model.dim = model.R*ones(length(nvec),1);
        %no. of pseudo inputs
        model.np = 100; 
        %decay rate (w.r.t. AdDelta)
        model.decay = 0.95;
        %no. of epoch
        model.epoch = 100;
        %batch of events
        model.batch_size = 100;
        %training time period
        model.T = data.T; 
        %training
        [model,test_ll_app, test_elbo] = GPTensorPP_online(data,model);

        %test
        test_approx_ll = predictive_log_likelihood_approx(model, data);
        
        ll_list(f) = max(test_approx_ll);
%         fprintf('GP-PP Tensor, approx. test log-likelihood: %g\n', test_approx_ll);
%         test_ELBO = predictive_log_ELBO(model, data);
%         fprintf('GP Tensor, test var-ELBO = %g\n', test_ELBO);
%         res = [];
%         res.epoch = 1:model.epoch;
%         res.ELBO = test_elbo;
%         res.LL_approx = test_ll_app;
    end
    res(i, 1) = mean(ll_list);
    res(i, 2) = std(ll_list) / sqrt(5);
end
save(save_file, 'res');


% save('./rfp-pp-crash.mat', 'res');
%save('model-gp-pp.mat', 'model');

