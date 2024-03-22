% 4.层分数回归并筛选最优指标 针对数据集分别适应的，没有统一的beta，容易超越baseline
clear;clc;close all;
%%
% MaD-DLS:(CSIQ IVC LIVE TID2013 KADID10K):(SROCC KROCC PLCC)
baselines = [0.973635, 0.854091, 0.943860;
             0.931052, 0.774274, 0.898428;
             0.964643, 0.845629, 0.821987;
             0.884902, 0.703035, 0.885035;
             0.919559, 0.750622, 0.856003];
% Max(MaD-DLS,AHIQ):(CSIQ IVC LIVE TID2013 KADID10K):(SROCC KROCC PLCC)
% baselines = [0.973635, 0.854091, 0.955000;
%              0.931052, 0.774274, 0.898428;
%              0.970000, 0.845629, 0.952000;
%              0.901000, 0.703035, 0.899000;
%              0.919559, 0.750622, 0.856003];
% Max(MaD-DLS,DISTS,LPIPS,DeepSim,BPRI):(CSIQ IVC LIVE TID2013 KADID10K):(SROCC KROCC PLCC)
% baselines = [0.973635, 0.854091, 0.943860;
%              0.931052, 0.774274, 0.911200;
%              0.974000, 0.845629, 0.821987;
%              0.899100, 0.703035, 0.891700;
%              0.919559, 0.750622, 0.856003];
% k的取值
no = 7;
fprintf("k = %d\n",no);
%% 根据 CSIQ/IVC/LIVE/TID2013/KADID10K 测试,根据 baseline 进一步筛选
% 读取 index + sw 到 Line
Line = load(['OutputData\max sw_', num2str(no), '_new\imagenet-vgg-f.txt']);
dsLs = {'CSIQ','IVC','LIVE','TID2013','KADID10K'};
MAT = [];
evLine = reshape(baselines',[1, 15]);
for co = 1:size(Line,1)
    if mod(co,100) == 0
        fprintf("LINE %d\n",co);
    end
    index = Line(co,1:no)+1;
    Evalue = zeros(1,length(dsLs)*3);
    for i = 1:length(dsLs)
        Layerscore_Mos = load(['OutputData\layer score + mos matrix\imagenet-vgg-f + ',dsLs{i},'.txt']);
        Mssim0 = Layerscore_Mos(:,1:end-1);
        Mssim = [Mssim0 Mssim0.^2 Mssim0.^(1/2) Mssim0.^3 Mssim0.^(1/3) log(Mssim0) 2.^Mssim0 exp(Mssim0)];
        Mssim_add = [];
        for x_i = 1:size(Mssim0,2)
            for y_i = (x_i+1):size(Mssim0,2)
                Mssim_add = [Mssim_add, Mssim0(:,x_i).*Mssim0(:,y_i)];
            end
        end
        Mssim = [Mssim, Mssim_add];
        mos = Layerscore_Mos(:,end);
        clear Mssim0 Layerscore_Mos Mssim_add;
        Mssim_s = [];
        for k=1:no
            id = index(k);
            Mssim_s = [Mssim_s, Mssim(:,id)];
        end
        X0 = Mssim_s - ones(size(Mssim,1),1)*mean(Mssim_s);
        y = mos;
        yhat = X0*inv(X0'*X0)*X0'*y + mean(y);
        begin = (i-1)*3+1;
        final = (i-1)*3+3;
        [SROCC, KROCC, PLCC]  = calculate_srocc_krocc_plcc(y,yhat);
        Evalue(begin:final) = [SROCC, KROCC, PLCC];
    end
    Delta = Evalue-evLine;
    if(all(Delta >= 0))
        MAT = [MAT;Line(co,:),Evalue];
    end
end
save(['MAT',num2str(no),'_new.mat'],"MAT");