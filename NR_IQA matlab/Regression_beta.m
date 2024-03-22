%% Regression
addpath(genpath('dataset'));addpath(genpath('OutputData'));
%%
baselines = [0.973635, 0.854091, 0.943860;
             0.931052, 0.774274, 0.898428;
             0.964643, 0.845629, 0.821987;
             0.884902, 0.703035, 0.885035;
             0.919559, 0.750622, 0.856003];
%%
no = 7;
fprintf("k = %d\n",no);
% 根据 TID2013 的数据计算回归方程
Layerscore_Mos = load('OutputData\layer score + mos matrix\imagenet-vgg-f + TID2013.txt');
Mssim = Layerscore_Mos(:,1:end-1);
mos = Layerscore_Mos(:,end);
% 读取 index + sw 到 Line
line = load(['OutputData\max sw_', num2str(no), '\imagenet-vgg-f.txt']);
% 用函数集扩充 Mssim
Mssim = Expand(Mssim);
clear Layerscore_Mos;
% 回归
mat = zeros(size(line,1),2*no+1);
for co = 1:size(line,1)
    if mod(co,1000) == 0
        fprintf("第 %d 行 index \n",co);
    end
    index = line(co,1:no)+1;
    Mssim_s = [];
    for k=1:no
        Mssim_s = [Mssim_s, Mssim(:,index(k))];
    end
    x0 = Mssim_s - ones(size(Mssim,1),1)*mean(Mssim_s);
    y = mos;
    beta = inv(x0'*x0)*x0'*y;
    % id(1-k) beta sw
    mat(co,:) = [line(co,1:no), beta',line(co,no+1)];
end
%% 根据 CSIQ/IVC/LIVE/TID2013/KADID10K 测试,并根据 baseline 进一步筛选
line = mat;
dsLs = {'CSIQ','IVC','LIVE','TID2013','KADID10K'};
mat = [];
evLine = reshape(baselines',[1, 15]);
for co = 1:size(line,1)
    if mod(co,100) == 0
        fprintf("LINE %d\n",co);
    end
    index = line(co,1:no)+1;
    beta = line(co,no+1:2*no);
    Evalue = zeros(1,length(dsLs)*3);
    for i = 1:length(dsLs)
        Layerscore_Mos = load(['OutputData\layer score + mos matrix\imagenet-vgg-f + ',dsLs{i},'.txt']);
        Mssim = Layerscore_Mos(:,1:end-1);
        mos = Layerscore_Mos(:,end);
        % 用函数集扩充 Mssim
        Mssim = Expand(Mssim);
        clear Layerscore_Mos;
        Mssim_s = [];
        for k=1:no
            id = index(k);
            Mssim_s = [Mssim_s, Mssim(:,id)];
        end
        y = mos;
        yhat = Mssim_s*beta';
        begin = (i-1)*3+1;
        final = (i-1)*3+3;
        Evalue(begin:final) = CalculateSKP(y,yhat);
    end
    Delta = Evalue-evLine;
    if(all(Delta >= 0))
        mat = [mat;line(co,:),Evalue];
        a = 1
    end
end