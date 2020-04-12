% Maximum Mean and Covariance Discrepancy for Unsupervised Domain Adaptation.
% Neural Processing Letters, 2019.
% Wenju Zhang (zhangwenju13@nudt.edu.cn), Xiang Zhang, Long Lan, Zhigang Luo.

warning off;
options.k = 90;
options.ker = 'primal';
options.lambda = 0.001;
options.beta = 1;

T = 10;

src_name = {'PIE05','PIE05','PIE05','PIE05','PIE07','PIE07','PIE07','PIE07','PIE09','PIE09','PIE09','PIE09','PIE27','PIE27','PIE27','PIE27','PIE29','PIE29','PIE29','PIE29'};
tgt_name = {'PIE07','PIE09','PIE27','PIE29','PIE05','PIE09','PIE27','PIE29','PIE05','PIE07','PIE27','PIE29','PIE05','PIE07','PIE09','PIE29','PIE05','PIE07','PIE09','PIE27'};

result = [];

for idx = 1:length(src_name)

    src = char(src_name{idx});
    tgt = char(tgt_name{idx});
    task_name = strcat(src,'_vs_',tgt);
    
    load(strcat('./data/',src));
    Xs = fea';
    Xs = Xs*diag(sparse(1./sqrt(sum(Xs.^2))));
    Ys = gnd;
    load(strcat('./data/',tgt));
    Xt = fea';
    Xt = Xt*diag(sparse(1./sqrt(sum(Xt.^2))));
    Yt = gnd;
    
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

fid = fopen(strcat('./result_PIE.txt'),'wt');
fprintf(fid,'%0.4f\n',result);
fclose(fid);