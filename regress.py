import argparse
import numpy as np
from scipy.stats import spearmanr, pearsonr

# Expand函数的Python实现
def expand(Mssim):
    Mssim_expand = np.hstack((
        Mssim, 
        Mssim**2, 
        np.sqrt(Mssim), 
        Mssim**3, 
        Mssim**(1/3), 
        np.log(Mssim), 
        np.power(2, Mssim), 
        np.exp(Mssim)
    ))
    return Mssim_expand

def expand_cross(Mssim):
    # 计算 Mssim 的各种变换
    Mssim0 = np.hstack((
        Mssim,
        Mssim**2,
        np.sqrt(Mssim),
        Mssim**3,
        Mssim**(1/3),
        np.log(Mssim),
        2**Mssim,
        np.exp(Mssim)
    ))

    # 初始化 Mssim_add 为空数组
    Mssim_add = []

    # 计算 Mssim0 中每一列的乘积
    for x_i in range(Mssim0.shape[1]):
        for y_i in range(x_i + 1, Mssim0.shape[1]):
            Mssim_add.append(Mssim0[:, x_i] * Mssim0[:, y_i])

    # 将计算结果转换为 NumPy 数组并添加到 Mssim_expand 中
    Mssim_expand = np.hstack((Mssim0, np.array(Mssim_add).T))

    return Mssim_expand

# CalculateSP函数的Python实现
def calculate_sp(y, yhat):
    SROCC, _ = spearmanr(y, yhat)
    PLCC, _ = pearsonr(y, yhat)
    return SROCC, PLCC

def loadtxt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        fields = line.split('\t')[:-1]
        float_fields = [float(field) for field in fields]
        data.append(float_fields)
    return np.array(data)


def main(config):
    # 设置参数
    no = 7
    print(f"k = {no}")

    Layerscore_Mos = loadtxt(f'.\outputs\{config.dataset}.txt')
    mos = Layerscore_Mos[:, -1]
    Mssim = Layerscore_Mos[:, :-1]

    # 读取 index + sw 到 Line
    if config.dataset == config.pretrained_dataset:
        line = np.loadtxt(f'.\outputs\sw_{no}_{config.dataset}.txt')
    else:
        line = np.loadtxt(f'.\outputs\sw_{no}_{config.dataset}_{config.pretrained_dataset}.txt')

    # 用函数集扩充 Mssim
    Mssim = expand(Mssim)

    # 初始化矩阵
    mat = np.zeros((line.shape[0], 2 * no + 2))

    for co, row in enumerate(line):
        if (co + 1) % 1000 == 0:
            print(f"Processed {co + 1} index lines out of 50000 lines")

        index = row[:no].astype(int) # 索引从0开始
        Mssim_s = Mssim[:, index]

        # 计算线性回归
        x0 = Mssim_s - np.mean(Mssim_s, axis=0)
        y = mos
        beta = np.linalg.inv(x0.T @ x0) @ (x0.T @ y)
        yhat = x0 @ beta + np.mean(y)
        s, p = calculate_sp(y, yhat)

        # 保存结果到矩阵
        mat[co] = np.concatenate((row[:no], beta, [s, p]))

    sorted_indices = np.argsort(mat[:, 14], axis=0, kind='mergesort')[::-1]
    sorted_indices = sorted_indices.reshape(-1, 1)
    sorted_indices = np.tile(sorted_indices, (1, mat.shape[1]))
    sorted_matrix = np.take_along_axis(mat, sorted_indices, axis=0)
    mat = sorted_matrix

    # 保存结果到文件
    if config.dataset == config.pretrained_dataset:
        output_file = f'outputs\\results_{config.dataset}_no{no}.txt' 
    else:
        output_file = f'outputs\\results_{config.dataset}_{config.pretrained_dataset}_no{no}.txt'
    np.savetxt(output_file, mat, delimiter=',')
    print("output to "+output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='tid2013', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--pretrained_dataset', dest='pretrained_dataset', type=str, default='koniq-10k', help='Support datasets: koniq-10k|live|csiq|tid2013')
    config = parser.parse_args()
    main(config)
