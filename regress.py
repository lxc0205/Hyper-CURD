import argparse
import numpy as np
from utils import calculate_sp, loaddata, sort
from tqdm import tqdm
def savedata(file, mat, no):
    for i in range(len(mat)):
        if i < no:
            file.write(str(int(mat[i])))
        else:
            file.write(str(mat[i]))
        file.write('\t')
    file.write('\n')
def main(config, no = 7):
    Mssim, mos = loaddata(f'./outputs/eval outputs/{config.dataset}.txt', config.dataset, config.pretrained_dataset)

    # 读取 index + sw 到 Line
    if config.dataset == config.pretrained_dataset:
        line = np.loadtxt(f'./outputs/sort outputs/sw_{config.dataset}_sorted.txt')
    else:
        line = np.loadtxt(f'./outputs/sort outputs/sw_{config.dataset}_{config.pretrained_dataset}_sorted.txt')

    # 初始化矩阵
    mat = np.zeros((line.shape[0], 2 * no + 2))

    for co, row in tqdm(enumerate(line)):

        index = row[:no].astype(int) # 索引从0开始
        Mssim_s = Mssim[:, index]

        # 计算线性回归
        x0 = Mssim_s
        # x0 = Mssim_s - np.mean(Mssim_s, axis=0)
        y = mos

        beta = np.linalg.inv(x0.T @ x0) @ (x0.T @ y)
        # yhat = x0 @ beta + np.mean(y)
        yhat = x0 @ beta
 
        s, p = calculate_sp(y, yhat)

        # 保存结果到矩阵
        mat[co] = np.concatenate((row[:no], beta, [s, p]))
    
    # 按照srcc排序
    # mat = sort(mat, order=False, row=14)

    # 保存结果到文件
    if config.dataset == config.pretrained_dataset:
        output_file = f"./outputs/regress outputs/regress_{config.dataset}.txt" 
    else:
        output_file = f'./outputs/regress outputs/regress_{config.dataset}_{config.pretrained_dataset}.txt'
    with open(output_file, 'w') as file:
        for i in range(mat.shape[0]):
            savedata(file, mat[i,:], no)
        print("output to "+output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='tid2013', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--pretrained_dataset', dest='pretrained_dataset', type=str, default='koniq-10k', help='Support datasets: koniq-10k|live|csiq|tid2013')
    config = parser.parse_args()
    main(config)
