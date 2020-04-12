% Maximum Mean and Covariance Discrepancy for Unsupervised Domain Adaptation.
% Neural Processing Letters, 2019.
% Wenju Zhang (zhangwenju13@nudt.edu.cn), Xiang Zhang, Long Lan, Zhigang Luo.

warning off;
options.k = 20;
options.ker = 'linear';
options.lambda = 1;
options.beta = 1;

T = 10;

src_name = {'Caltech10','Caltech10','Caltech10','amazon','amazon','amazon','webcam','webcam','webcam','dslr','dslr','dslr'};
tgt_name = {'amazon','webcam','dslr','Caltech10','webcam','dslr','Caltech10','amazon','dslr','Caltech10','amazon','webcam'};
result = [];

for idx = 1:length(src_name)

    src = char(src_name{idx});
    tgt = char(tgt_name{idx});
    task_name = strcat(src,'_vs_',tgt);
    
    load(['./data/' src '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
    Xs = zscore(fts,1);
    Xs = Xs';
    Ys = labels;
    
    load(['./data/' tgt '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
    Xt = zscore(fts,1);
    Xt = Xt';
    Yt = labels;
    
    Cls = [];
    Acc = []; 
    for t = 1:T
        [Z,A] = McDA(Xs,Xt,Ys,Cls,options);
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(Xs,2));
        Zt = Z(:,size(Xs,2)+1:end);

        Cls = knnclassify(Zt',Zs',Ys,1);
        acc = length(find(Cls==Yt))/length(Yt);
        Acc = [Acc;acc];
    end
    fprintf('%s: acc=%0.4f\n', task_name, Acc(end));
    result = [result,Acc(end)];
end

fid = fopen(strcat('./result_office.txt'),'wt');
fprintf(fid,'%0.4f\n',result);
fclose(fid);
