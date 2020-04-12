function [M,R] = constructMap(ns,nt,Ys,Yt0,n,C,X)

e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
M  = e*e'*C;
Ms = zeros(n, n);
if ~isempty(Yt0) && length(Yt0)==nt
    for c = reshape(unique(Ys),1,C)
        
        idx_s = find(Ys==c);
        ns_c = length(idx_s);
        
        idx_t = ns + find(Yt0==c);
        nt_c = length(idx_t);
        
        Ms(idx_s, idx_s) =   1/(ns_c * ns_c) * ones(ns_c, ns_c);
        Ms(idx_s, idx_t) = - 1/(ns_c * nt_c) * ones(ns_c, nt_c);
        Ms(idx_t, idx_s) = - 1/(nt_c * ns_c) * ones(nt_c, ns_c);
        Ms(idx_t, idx_t) =   1/(nt_c * nt_c) * ones(nt_c, nt_c);
    end
end
M = M + Ms;
M = M/norm(M,'fro');

R = CovDiff(1:ns,ns+1:ns+nt,X,C);

if ~isempty(Yt0) && length(Yt0)==nt
    for c = reshape(unique(Ys),1,C)
        idx_s = find(Ys==c);
        idx_t = ns + find(Yt0==c);
        R = R + CovDiff(idx_s,idx_t,X,1);
    end
end

end

function R = CovDiff(idx_s,idx_t,X,w)
% equivalent form of XZcX'XZcX'
R = Cov(idx_s,X,w) -  Cov(idx_t,X,w);
R = R*R';
end

function cov = Cov(idx,X,w)
Xs = X(:,idx);
n = length(idx);
H = w * ( 1/n*eye(n) - 1/n^2*ones(n, n) );
cov = Xs*H*Xs';
end