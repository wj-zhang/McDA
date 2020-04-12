function [Z, A] = McDA(Xs, Xt, Ys, Yt0, options)

% Maximum Mean and Covariance Discrepancy for Unsupervised Domain Adaptation.
% Neural Processing Letters, 2019.
% Wenju Zhang (zhangwenju13@nudt.edu.cn), Xiang Zhang, Long Lan, Zhigang Luo.

k = options.k;
lambda = options.lambda;
ker = options.ker;
beta = options.beta;

X = [Xs,Xt];
X = X*diag(sparse(1./sqrt(sum(X.^2))));
[m,n] = size(X);
ns = size(Xs,2);
nt = size(Xt,2);
C = length(unique(Ys));

H = eye(n)-1/(n)*ones(n,n);

if ~strcmp(ker,'primal')
    K = kernel(ker,X,[],1);
end

if ~strcmp(ker,'primal')
    [M,R] = constructMap(ns,nt,Ys,Yt0,n,C,K);
    XMX = K*M*K';
    XHX = K*H*K';
else
    [M,R] = constructMap(ns,nt,Ys,Yt0,n,C,X);
    XMX = X*M*X';
    XHX = X*H*X';
end

[A,~] = eigs((XMX+XMX.')/2+lambda*eye(size(XMX,1))+beta*R,XHX,k,'SM');

if strcmp(ker,'primal')
    Z = A'*X;
else
    Z = A'*K;
end

end
