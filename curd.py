import time
import argparse
import numpy as np
from tqdm import tqdm
from itertools import combinations
from scipy.linalg import det
from utils import loadtxt, savedata, sort

def savesortdata(file, mat):
    for i in range(len(mat)):
        file.write(str(mat[i]))
        file.write('\t')
    file.write('\n')
def main(config, no = 7):

    print('curd: %s, %s' % (config.dataset, config.pretrained_dataset))

    # 输入输出文件夹
    if config.dataset == config.pretrained_dataset:
        file_name = f"./outputs/eval outputs/{config.dataset}.txt"
        outputs_file_name = f"./outputs/curd outputs/sw_{config.dataset}.txt"
    else:
        file_name = f"./outputs/eval outputs/{config.dataset}_{config.pretrained_dataset}.txt"
        outputs_file_name = f"./outputs/curd outputs/sw_{config.dataset}_{config.pretrained_dataset}.txt"
    
    # 读取数据
    Mssim, mos = loadtxt(file_name, config.dataset, config.pretrained_dataset)

    # 重新拼接
    data = np.transpose(np.concatenate((Mssim, mos), axis=1))
  
    # Pearson 相关矩阵
    R = np.corrcoef(data)

    # 记录开始时间
    start_time = time.time()

    threshold = 0.9999
    with open(outputs_file_name, 'w') as file:
        for combo in tqdm(combinations(range(R.shape[0]-1), no)):
            # 考虑互相关列
            indices = combo + (R.shape[0]-1, )

            submatrix_den = R[combo, :][:, combo] # 计算square_omega的分母
            if (submatrix_den > threshold).sum() > no: # 大于 threahold 的元素个数,不超过no ( 除了 no 个对角线元素 )
                continue

            submatrix_num = R[indices, :][:, indices] # 计算square_omega的分子

            if det(submatrix_den) == 0: # 避免分母为0
                square_omega = 1
                print('division by zero')
            else:
                square_omega = det(submatrix_num) / det(submatrix_den) # 计算条件不相关

            try:
                assert square_omega >= 0, "The square_omega is not non-negative."
            except AssertionError as e:
                print(e)

            savedata(file, combo, square_omega) # 写入当前行

    # 记录结束时间
    end_time = time.time()

    # 计算并打印运行时间
    run_time = end_time - start_time
    print(f"The curd code took {run_time} seconds to run.\n\n")

    # 排序
    print(f"Start sorting....")
    if config.dataset == config.pretrained_dataset:
        input_file = f"./outputs/curd outputs/sw_{config.dataset}.txt"
        output_file = f"./outputs/sort outputs/sw_{config.dataset}_sorted.txt"
    else:
        input_file = f"./outputs/curd outputs/sw_{config.dataset}_{config.pretrained_dataset}.txt"
        output_file = f"./outputs/sort outputs/sw_{config.dataset}_{config.pretrained_dataset}_sorted.txt"

    data = loadtxt(input_file)
    with open(output_file, 'w') as file:
        sorted_matrix = sort(data, config.order, row=7)
        for i in range(config.save_num):
            savesortdata(file, sorted_matrix[i,:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='tid2013', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--pretrained_dataset', dest='pretrained_dataset', type=str, default='koniq-10k', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--order', dest='order', type=bool, default = True, help='Ascending(True) or descending(False)') 
    parser.add_argument('--save_num', dest='save_num', type=int, default=50000, help='Save numbers.') 
    config = parser.parse_args()
    main(config)



