%% Regression
clear;clc;close all;
addpath(genpath('outputs'));
%%
no = 7;
fprintf("k = %d\n",no);
% 根据 TID2013 的数据计算回归方程
dataset = 'live';
Layerscore_Mos = load(['..\outputs\', dataset, '.txt']);
Mssim = Layerscore_Mos(:,1:end-1);
mos = Layerscore_Mos(:,end);
clear Layerscore_Mos;
% 读取 index + sw 到 Line
line = load(['outputs\sw_', num2str(no), '_', dataset, '.txt']);
% 用函数集扩充 Mssim
expand = Expand;
Mssim = expand.Expand_base(Mssim);
% 回归
mat = zeros(size(line,1),2*no+4);
for co = 1:size(line,1)
    if mod(co,1000) == 0
        fprintf("No.%d index line of 50000 lines\n",co);
    end
    index = line(co,1:no)+1;
    Mssim_s = [];
    for k=1:no
        Mssim_s = [Mssim_s, Mssim(:,index(k))];
    end
    x0 = Mssim_s - ones(size(Mssim,1),1)*mean(Mssim_s);
    y = mos;
    beta = inv(x0'*x0)*x0'*y;
    yhat = x0*inv(x0'*x0)*x0'*y + mean(y);
    [s, k, p] = CalculateSKP(y,yhat);
    % id(1-k) beta sw
    mat(co,:) = [line(co,1:no), beta',line(co,no+1),s, k, p];
end
