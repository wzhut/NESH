clear all;
close all;
rng('default');

addpath_recurse('./util');
addpath_recurse('./tensor_toolbox_2.6');
addpath_recurse('./lightspeed');
addpath_recurse('./minFunc_2012');

%load('../data/Simu-train.mat');
load('../../RFP-HP/Simu-train-1000.mat');
%load('../data/Simu-train-large.mat');
nvec = max(data.ind);
%data.e = data.e(1:900);
%data.ind = data.ind(1:900,:);

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

data.test_ind = data.train_subs;
data.test_vals = data.y_subs;
data.test_T = data.T;


%GP tensor poission process regression
%parameters setting
model = [];
%model.nmod = nmod;
%the length of latent features
model.R = 2;
for k=1:nmod
    %model.U{k} = randn(nvec(k), model.R);
    model.U{k} = rand(nvec(k), model.R);
end
%decay rate (w.r.t. AdDelta)
model.decay = 0.95;
%no. of epoch
model.epoch = 50;
%batch of events
model.batch_size = 50;
%training time period
model.T = data.T; 
%training
[model,test_ll] = CPTensorPP_online(data, model);
figure;
plot(1:length(test_ll), test_ll);
fprintf('max ll = %g\n', max(test_ll));
%save('cp-pp-model.mat', 'model');
n = nvec(1);
figure;
scatter(model.U{1}(1:n/2,1), model.U{1}(1:n/2,2), 'red');
hold on;
scatter(model.U{1}(n/2+1:n,1), model.U{1}(n/2+1:n,2), 'blue');

figure;
scatter(model.U{2}(1:n/2,1), model.U{2}(1:n/2,2), 'red');
hold on;
scatter(model.U{2}(n/2+1:n,1), model.U{2}(n/2+1:n,2), 'blue');


figure;
scatter(model.U{3}(1:n/2,1), model.U{3}(1:n/2,2), 'red');
hold on;
scatter(model.U{3}(n/2+1:n,1), model.U{3}(n/2+1:n,2), 'blue');

U = model.U;
save('Simu-U-CP-PP.mat', 'U');
