import argparse
from utils import loadMssimMos, calculate_sp
def main(config):
    Mssim, mos = loadMssimMos(f'./outputs/hyperIQA outputs/{config.dataset}_{config.predataset}.txt', config.dataset, config.predataset)
    
    # ----index----   sw    -------beta-------    srcc   plcc     (srcc + plcc)/2

    # beta = [15969.875, 4326.625, 883.921875, 1.0262349247932434, -278.994140625, -30607.75, 9705.1875]
    # index = [1, 7, 19, 30, 31, 37, 43]

    beta = [3.324539993639519,13.125284559431748,0.8595986471534047,-12.530279544116638,-68.34548125360743,-8.317460779137036,28.50518866092898]
    index = [7,10,18,21,23,26,41]

    Mssim_s = Mssim[:, index]
    yhat = Mssim_s @ beta

    srcc, plcc = calculate_sp(mos.squeeze(), yhat.squeeze())
    print('Testing SRCC %4.4f,\tPLCC %4.4f' % (srcc, plcc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='csiq', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--predataset', dest='predataset', type=str, default='koniq-10k', help='Support datasets: koniq-10k|live|csiq|tid2013')
    config = parser.parse_args()
    print(f'regression functions.')
    main(config)
