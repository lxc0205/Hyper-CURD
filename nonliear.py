import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_sp, loaddata


def main(config):
    Mssim, mos = loaddata(f'./outputs/eval outputs/{config.dataset}.txt', config.dataset, config.pretrained_dataset)

    beta = [float(x.strip()) for x in config.beta.split()]
    index = [int(x.strip()) for x in config.index.split()]

    Mssim_s = Mssim[:, index]

    # 计算线性回归
    # x0 = Mssim_s - np.mean(Mssim_s, axis=0)
    x0 = Mssim_s
    y = mos
    yhat = x0 @ beta

    if config.figure:
        sorted_indices = np.argsort(y)
        y_sorted = y[sorted_indices]
        yhat_sorted = yhat[sorted_indices]
        
        # 使用matplotlib绘制曲线图
        x = range(len(y))
        plt.figure(figsize=(10, 5))  # 可以设置图形的大小
        plt.plot(x, y_sorted, 'b-', label='Vector y')  # 绘制第一个向量，使用蓝色实线
        plt.plot(x, 0.07*yhat_sorted, 'r-', label='Vector yhat')  # 绘制第二个向量，使用红色实线
        plt.xlabel('X-axis')  # 设置x轴标签
        plt.ylabel('Y-axis')  # 设置y轴标签
        plt.title('Comparison of Two Vectors')  # 设置图形标题
        plt.legend()  # 显示图例
        plt.grid(True)  # 显示网格
        plt.show()  # 显示图形

    s, p = calculate_sp(y, yhat)
    print(f"SRCC {s},\tPLCC {p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='tid2013', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--index', dest='index', type=str, default='0 1 2 3 4 5 6', help='Index of selected layers')
    parser.add_argument('--beta', dest='beta', type=str, default='0.0 1.0 2.0 3.0 4.0 5.0 6.0', help='Coefficients of linear regression')
    parser.add_argument('--figure', dest='figure', type=bool, default=False, help='Figure show flag')
    config = parser.parse_args()
    main(config)
