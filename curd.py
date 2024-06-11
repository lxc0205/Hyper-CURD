import math
import argparse
import numpy as np
from tqdm import tqdm
from scipy.linalg import det
from itertools import combinations
from utils import loadMssimMos, savedata_intfloat_comma, calculate_sp, sort
def main(config, no = 7, threshold = 0.9999):
    Mssim, mos = loadMssimMos(f"./outputs/hyperIQA outputs/{config.dataset}_{config.pretrained_dataset}.txt", config.dataset, config.pretrained_dataset)
    R = np.abs(np.corrcoef(np.transpose(np.concatenate((Mssim, mos), axis=1))))

    mat = np.zeros((math.comb(R.shape[0] - 1, no), no + 1))
    epoch = 0
    for combo in tqdm(combinations(range(R.shape[0]-1), no), total=math.comb(R.shape[0] - 1, no)):
        submatrix_den = R[combo, :][:, combo] # 计算square_omega的分母
        if (submatrix_den > threshold).sum() > no: continue # 大于threahold的元素个数不超过对角线元素个数 no

        indices = combo + (R.shape[0]-1, )
        submatrix_num = R[indices, :][:, indices] # 计算square_omega的分子

        square_omega = 1 if det(submatrix_den) == 0 else det(submatrix_num) / det(submatrix_den)
        if square_omega < 0: continue
        mat[epoch] = np.concatenate((combo, [square_omega]))
        epoch += 1

    # 排序
    mat = mat[:epoch, :]
    sorted_matrix = sort(mat, config.order, row = 7)
    print(f'Number of data: {epoch}\n')
    
    # regression evaluation
    mat = np.zeros((sorted_matrix.shape[0], 2*no + 4))
    epoch = 0
    for row in tqdm(sorted_matrix, total=len(sorted_matrix)):
        try:
            index = row[:no].astype(int)
            Mssim_s = Mssim[:, index]

            beta = np.linalg.inv(Mssim_s.T @ Mssim_s) @ (Mssim_s.T @ mos)
            yhat = Mssim_s @ beta

            srcc, plcc = calculate_sp(mos.squeeze(), yhat.squeeze())
            # 0 1 2 3 4 5 6    7    8 9 10 11 12 13 14     15     16             17
            # ----index----   sw    -------beta-------    srcc   plcc     (srcc + plcc)/2
            # ----index----   sw    -------beta-------    srcc   plcc    srcc+plcc - 0.3*sw
            mat[epoch] = np.concatenate((row[:no+1], beta.squeeze(), [srcc, plcc, (srcc + plcc)/2]))
            # mat[epoch] = np.concatenate((row[:no+1], beta.squeeze(), [srcc, plcc, srcc + plcc - 0.3*row[no]]))
            epoch += 1
        except:
            continue
    
    print(f'Number of data: {epoch}\n')

    # 排序,结果保存到文件
    mat = sort(mat, order=False, row=17)
    mat = mat[:5000000, :]
    with open(f'./outputs/curd outputs/fitting_{config.dataset}_{config.pretrained_dataset}.txt', 'w') as file:
        for i in range(mat.shape[0]):
            savedata_intfloat_comma(file, mat[i,:], no)
    print("Curd finished!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='csiq', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--pretrained_dataset', dest='pretrained_dataset', type=str, default='koniq-10k', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--order', dest='order', type=bool, default = True, help='Ascending(True) or descending(False)') 
    parser.add_argument('--save_num', dest='save_num', type=int, default=50000, help='Save numbers.')
    config = parser.parse_args()
    print(f'curd: {config.dataset}_{config.pretrained_dataset}')
    main(config)



