import time
import argparse
from tqdm import tqdm
import numpy as np
from itertools import combinations
from scipy.linalg import det
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

def loadtxt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        fields = line.split('\t')[:-1]
        float_fields = [float(field) for field in fields]
        data.append(float_fields)
    return np.array(data)

def conditional_uncorrelation(submatrix_num, submatrix_den):
    det_num= det(submatrix_num)
    det_den = det(submatrix_den)
    square_omega = det_num / det_den
    return square_omega

def normalize(scores, datasets, new_min=0, new_max=1):
    if datasets == 'csiq':
        old_min = 0 
        old_max = 1
        dmos = True
    elif datasets == 'live':
        old_min = 0
        old_max = 100
        dmos = True
    elif datasets == 'tid2013':
        old_min = 0
        old_max = 9
        dmos = False
    elif datasets == 'koniq-10k':
        old_min = 0
        old_max = 100
        dmos = False
    else:
        print('wrong dataset name!')
        return 0

    # 计算归一化后的分数
    if dmos:
        output_scores = [(1-((new_max - new_min) * (score - old_min) / (old_max - old_min) + new_min)) for score in scores]    # 如果是 Dmos，则将分数取反
    else:
        output_scores = [((new_max - new_min) * (score - old_min) / (old_max - old_min) + new_min) for score in scores]
    return output_scores

def saveSquareOmega(file, combo, square_omega):
    for i in range(len(combo)):
        file.write(str(combo[i]))
        file.write('\t')
    file.write(str(square_omega))
    file.write('\t')
    file.write('\n')

def main(config):
    k = 7
    # 输入输出文件夹
    file_name = f"./outputs/eval outputs/{config.file_name}.txt"
    outputs_file_name = f"./outputs/curd outputs/sw_{config.file_name}.txt"
    
    # 读取数据
    data = loadtxt(file_name)
    mos = data[:, -1]
    Mssim = data[:, :-1]

    # 用函数集扩充 Mssim
    Mssim = expand(Mssim)
    # mos = normalize(mos, config.dataset)
    mos = mos[:, np.newaxis]
    data = np.concatenate((Mssim, mos), axis=1)
    data = np.transpose(data)
  
    # Pearson 相关矩阵
    R = np.corrcoef(data)

    # 记录开始时间
    start_time = time.time()

    threshold = 0.9999
    with open(outputs_file_name, 'w') as file:
        for combo in tqdm(combinations(range(R.shape[0]-1), k)):
            # 考虑互相关列
            indices = combo + (R.shape[0]-1, )

            # 计算square_omega的分母
            submatrix_den = R[combo, :][:, combo] # 分母
            # 统计大于 0.9999 的元素个数, 除了k个对角线元素, 所有元素应当小于threahold
            if (submatrix_den > threshold).sum() > k:
                continue

            # 计算square_omega的分子
            submatrix_num = R[indices, :][:, indices] # 分子

            # 计算条件不相关
            square_omega = conditional_uncorrelation(submatrix_num, submatrix_den)

            try:
                assert square_omega >= 0, "The square_omega is not non-negative."
            except AssertionError as e:
                print(e)

            # 写入当前行，'\n'代表换行符
            saveSquareOmega(file, combo, square_omega)

    # 记录结束时间
    end_time = time.time()

    # 计算并打印运行时间
    run_time = end_time - start_time
    print(f"The code took {run_time} seconds to run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', dest='file_name', type=str, default='koniq-10k', help='The name of the layer scores in eval output directory')
    config = parser.parse_args()
    main(config)