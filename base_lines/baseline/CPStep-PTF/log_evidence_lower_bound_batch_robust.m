%deal with "divided by 0" case
function [g] =  log_evidence_lower_bound_batch_robust(x, model, sel)
    x = vec(x);
    nmod = length(model.nvec);
    nvec = model.nvec;
    U = cell(nmod,1);
    st = 0;
    for k=1:nmod
        U{k} = reshape(x(st + 1 : st + nvec(k)*model.R), nvec(k), model.R);
        st = st + nvec(k)*model.R;
    end
    s = exp(x(end));
    %# of entries
    n = size(model.subs,1);
    n_batch = length(sel);
    y = model.y(sel)*n/n_batch;
    T = model.T *n/n_batch;
    nmod = model.nmod;    
    %ind_batch = model.subs(sel,:);
    U_batch = cell(nmod, 1);
    M = ones(n_batch, model.R);
    inner_held_out = cell(nmod,1);
    for k=1:nmod
        inner_held_out{k} = M;
    end
    for k=1:nmod
        U_batch{k} = U{k}(model.subs(sel, k),:);
        M = M.*U_batch{k};
        for j=1:nmod
            if j~=k
                inner_held_out{j} = inner_held_out{j}.*U_batch{k};
            end
        end
    end
%     lam_batch = exp(sum(M,2));
    M_sum = sum(M, 2)+1e-6;
    g_batch = cell(nmod,1);
    for k=1:nmod
%         g_batch{k} = diag(-T*lam_batch + y)*inner_held_out{k};
        g_batch{k} = diag(-T*2.*M_sum + y * 2 ./ M_sum)*inner_held_out{k};
    end
    gU = cell(nmod, 1);
    %log prior term
    %treat time factors as usually factors, no markov
    for k=1:nmod
        gU{k} = -U{k};
    end
%     for k=1:nmod-1
%         gU{k} = -U{k};        
%     end
    
%     %the last mode is time factor    
%     UD = zeros(size(U{nmod}));
%     UD(2:end,:) = U{nmod}(1:end-1,:);
%     UD = U{nmod} - UD;
%     UD(1,:) = zeros(1,model.R);
%     gU{nmod} = zeros(size(U{nmod}));
%     gU{nmod}(1,:) = -U{nmod}(1,:) + s*UD(2,:);
%     for k=2:nvec(nmod)-1
%         gU{nmod}(k,:) = -s*UD(k,:) + s*UD(k+1,:);
%     end
%     gU{nmod}(end,:)= -s*UD(end,:);
%     gs = (model.a0 - 1 + 0.5*model.R*(nvec(nmod)-1))/s ...
%         -(model.b0 + 0.5*sum(vec(UD.^2)));
%     %the grad. in log-domain
%     gs = gs*s;
    gs = 0;
    
    
    g = [];
    for k=1:nmod
        gU{k} = gU{k} + model.ind2entry{k}(:,sel)*g_batch{k};
        g = [g(:); vec(gU{k})];
    end   
    g=[g(:);gs];
    if sum(isinf(g)) || sum(isnan(g))
        fprintf('inf!\n');
    end
end