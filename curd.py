import os
import math
import random
import argparse
import numpy as np
from tqdm import tqdm
from scipy.linalg import det
from itertools import combinations
from utils import loadMssimMos, loaddata, savedata_intfloat, savedata_withlabel, sort, calculate_sp 
def main(config, no = 7, threshold = 0.9999):

    print('curd: %s_%s' % (config.dataset, config.pretrained_dataset))

    Mssim, mos = loadMssimMos(f"./outputs/hyperIQA outputs/{config.dataset}_{config.pretrained_dataset}.txt", config.dataset, config.pretrained_dataset)
    data = np.transpose(np.concatenate((Mssim, mos), axis=1))
    R = np.abs(np.corrcoef(data))

    temp_dir = f"./outputs/curd outputs/temp_" + str(random.randint(100000, 999999)) + ".txt"
    with open(temp_dir, 'w') as file:
        total_combos = math.comb(R.shape[0] - 1, no)
        for combo in tqdm( combinations(range(R.shape[0]-1), no), total=total_combos):
            # 考虑互相关列
            indices = combo + (R.shape[0]-1, )

            submatrix_den = R[combo, :][:, combo] # 计算square_omega的分母
            if (submatrix_den > threshold).sum() > no: continue # 大于threahold的元素个数,不超过no (除了 no 个对角线元素)

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
    data = loaddata(temp_dir)
    os.remove(temp_dir)
    sorted_matrix = sort(data, config.order, row=7)
    sorted_matrix = sorted_matrix[:config.save_num, :]

    # 拟合
    mat = np.zeros((sorted_matrix.shape[0], 2 * no + 3))
    for co, row in tqdm(enumerate(sorted_matrix), total=len(sorted_matrix)):
        index = row[:no].astype(int)
        Mssim_s = Mssim[:, index]

        beta = np.linalg.inv(Mssim_s.T @ Mssim_s) @ (Mssim_s.T @ mos)
        yhat = Mssim_s @ beta

        s, p = calculate_sp(mos.squeeze(), yhat.squeeze())
        mat[co] = np.concatenate((row[:no+1], beta.squeeze(), [s, p]))

    # 按照srcc排序
    mat = sort(mat, order=False, row=15)

    # 保存结果到文件
    output_file = f'./outputs/curd outputs/fitting_{config.dataset}_{config.pretrained_dataset}.txt'
    with open(output_file, 'w') as file:
        for i in range(mat.shape[0]):
            savedata_intfloat(file, mat[i,:], no)
        print("output to "+output_file)
    print(f"Curd finished!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='live', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--pretrained_dataset', dest='pretrained_dataset', type=str, default='koniq-10k', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--order', dest='order', type=bool, default = True, help='Ascending(True) or descending(False)') 
    parser.add_argument('--save_num', dest='save_num', type=int, default=50000, help='Save numbers.') 
    config = parser.parse_args()
    main(config)



