import argparse
import numpy as np
from tqdm import tqdm
from utils import calculate_sp, loadMssimMos, savedata_intfloat, sort

def main(config, no = 7):
    Mssim, mos = loadMssimMos(f'./outputs/eval outputs/{config.dataset}_{config.pretrained_dataset}.txt', config.dataset, config.pretrained_dataset)

    line = np.loadtxt(f'./outputs/sort outputs/sw_{config.dataset}_{config.pretrained_dataset}_sorted.txt')

    mat = np.zeros((line.shape[0], 2 * no + 2))
    for co, row in tqdm(enumerate(line), total=len(line)):
        index = row[:no].astype(int) # 索引从0开始
        Mssim_s = Mssim[:, index]

        # 线性回归
        # Mssim_s = Mssim_s - np.mean(Mssim_s, axis=0)
        beta = np.linalg.inv(Mssim_s.T @ Mssim_s) @ (Mssim_s.T @ mos)
        # yhat = Mssim_s @ beta + np.mean(mos)
        yhat = Mssim_s @ beta

        s, p = calculate_sp(mos.squeeze(), yhat.squeeze())

        # 保存结果到矩阵
        mat[co] = np.concatenate((row[:no], beta.squeeze(), [s, p]))

    # 按照srcc排序
    mat = sort(mat, order=False, row=14)

    # 保存结果到文件
    output_file = f'./outputs/regress outputs/regress_{config.dataset}_{config.pretrained_dataset}.txt'
    with open(output_file, 'w') as file:
        for i in range(mat.shape[0]):
            savedata_intfloat(file, mat[i,:], no)
        print("output to "+output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='csiq', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--pretrained_dataset', dest='pretrained_dataset', type=str, default='csiq', help='Support datasets: koniq-10k|live|csiq|tid2013')
    config = parser.parse_args()
    main(config)
