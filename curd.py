import os
import math
import argparse
import numpy as np
from tqdm import tqdm
from itertools import combinations
from scipy.linalg import det
from utils import loaddata, loadMssimMos, savedata_withlabel, savedata, sort, calculate_sp, savedata_intfloat

def main(config, no = 7, threshold = 0.9999):

    print('curd: %s_%s' % (config.dataset, config.pretrained_dataset))

    Mssim, mos = loadMssimMos(f"./outputs/hyperIQA outputs/{config.dataset}_{config.pretrained_dataset}.txt", config.dataset, config.pretrained_dataset)
    data = np.transpose(np.concatenate((Mssim, mos), axis=1))
    R = np.abs(np.corrcoef(data))

    with open(f"./outputs/curd outputs/sw_{config.dataset}_{config.pretrained_dataset}.txt", 'w') as file:
        total_combos = math.comb(R.shape[0] - 1, no)
        for combo in tqdm( combinations(range(R.shape[0]-1), no), total=total_combos):
            # 考虑互相关列
            indices = combo + (R.shape[0]-1, )

            submatrix_den = R[combo, :][:, combo] # 计算square_omega的分母
            if (submatrix_den > threshold).sum() > no: # 大于 threahold 的元素个数,不超过no ( 除了 no 个对角线元素 )
                continue

            submatrix_num = R[indices, :][:, indices] # 计算square_omega的分子

            if det(submatrix_den) == 0: # 避免分母为0
                square_omega = 1
                print('Division by zero, square_omega is set to 1')
            else:
                square_omega = det(submatrix_num) / det(submatrix_den) # 计算条件不相关

            if square_omega < 0:
                print("The square_omega is not non-negative.")

            savedata_withlabel(file, combo, square_omega) # 写入当前行

    # 排序
    dir = f"./outputs/curd outputs/sw_{config.dataset}_{config.pretrained_dataset}.txt"
    data = loaddata(dir)
    os.remove(dir)
    sorted_matrix = sort(data, config.order, row=7)

    with open(dir, 'w') as file:
        for i in range(config.save_num):
            savedata(file, sorted_matrix[i,:])
    print(f"Curd finished!\n")

    # 拟合
    line = np.loadtxt(f'./outputs/curd outputs/sw_{config.dataset}_{config.pretrained_dataset}.txt')
    a = sum(sum(line - sorted_matrix))

    mat = np.zeros((sorted_matrix.shape[0], 2 * no + 2))
    for co, row in tqdm(enumerate(sorted_matrix), total=len(sorted_matrix)):
        index = row[:no].astype(int)
        Mssim_s = Mssim[:, index]

        beta = np.linalg.inv(Mssim_s.T @ Mssim_s) @ (Mssim_s.T @ mos)
        yhat = Mssim_s @ beta

        s, p = calculate_sp(mos.squeeze(), yhat.squeeze())
        mat[co] = np.concatenate((row[:no], beta.squeeze(), [s, p]))

    # 按照srcc排序
    mat = sort(mat, order=False, row=14)

    # 保存结果到文件
    output_file = f'./outputs/curd outputs/fitting_{config.dataset}_{config.pretrained_dataset}.txt'
    with open(output_file, 'w') as file:
        for i in range(mat.shape[0]):
            savedata_intfloat(file, mat[i,:], no)
        print("output to "+output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='live', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--pretrained_dataset', dest='pretrained_dataset', type=str, default='koniq-10k', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--order', dest='order', type=bool, default = True, help='Ascending(True) or descending(False)') 
    parser.add_argument('--save_num', dest='save_num', type=int, default=50000, help='Save numbers.') 
    config = parser.parse_args()
    main(config)



