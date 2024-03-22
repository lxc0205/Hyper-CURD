function [SROCC, KROCC, PLCC] = CalculateSKP(y,yhat)
    %y yhat需要为列向量
    SROCC = corr(y, yhat, 'type','spearman');
    KROCC = corr(y, yhat, 'type','kendall');
    PLCC = corr(y, yhat, 'type','pearson');
end