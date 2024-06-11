import argparse
from utils import loadMssimMos, calculate_sp
def main(config):
    Mssim, mos = loadMssimMos(f'./outputs/hyperIQA outputs/{config.dataset}_{config.pretrained_dataset}.txt', config.dataset, config.pretrained_dataset)
    
    beta = [15969.875, 4326.625, 883.921875, 1.0262349247932434, -278.994140625, -30607.75, 9705.1875]
    index = [1, 7, 19, 30, 31, 37, 43]

    Mssim_s = Mssim[:, index]
    yhat = Mssim_s @ beta

    srcc, plcc = calculate_sp(mos.squeeze(), yhat.squeeze())
    print('Testing SRCC %4.4f,\tPLCC %4.4f' % (srcc, plcc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='csiq', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--pretrained_dataset', dest='pretrained_dataset', type=str, default='koniq-10k', help='Support datasets: koniq-10k|live|csiq|tid2013')
    config = parser.parse_args()
    print(f'regression functions.')
    main(config)
