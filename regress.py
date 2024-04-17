import argparse
import numpy as np
from utils import expand, calculate_sp, loadtxt, normalize


def main(config):
    # 设置参数
    no = 7
    print(f"k = {no}")

    Layerscore_Mos = loadtxt(f'./utputs/eval outputs/{config.dataset}.txt')
    mos = Layerscore_Mos[:, -1]
    Mssim = Layerscore_Mos[:, :-1]

    # 读取 index + sw 到 Line
    if config.dataset == config.pretrained_dataset:
        line = np.loadtxt(f'./outputs/sort outputs/sw_{config.dataset}.txt')
    else:
        line = np.loadtxt(f'./outputs/sort outputs/sw_{config.dataset}_{config.pretrained_dataset}.txt')

    # 用函数集扩充 Mssim
    Mssim = expand(Mssim)
    mos = normalize(mos, config.dataset)

    # 初始化矩阵
    mat = np.zeros((line.shape[0], 2 * no + 2))

    for co, row in enumerate(line):
        if (co + 1) % 1000 == 0:
            print(f"Processed {co + 1} index lines out of 50000 lines")

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

    sorted_indices = np.argsort(mat[:, 14]+mat[:, 15], axis=0, kind='mergesort')[::-1]
    sorted_indices = sorted_indices.reshape(-1, 1)
    sorted_indices = np.tile(sorted_indices, (1, mat.shape[1]))
    sorted_matrix = np.take_along_axis(mat, sorted_indices, axis=0)
    mat = sorted_matrix

    # 保存结果到文件
    if config.dataset == config.pretrained_dataset:
        output_file = f"./outputs/regress outputs/regress_{config.dataset}.txt" 
    else:
        output_file = f'./outputs/regress outputs/regress_{config.dataset}_{config.pretrained_dataset}.txt'
    np.savetxt(output_file, mat, delimiter=',')
    print("output to "+output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='tid2013', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--pretrained_dataset', dest='pretrained_dataset', type=str, default='koniq-10k', help='Support datasets: koniq-10k|live|csiq|tid2013')
    config = parser.parse_args()
    main(config)
