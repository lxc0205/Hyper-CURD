import os
import random
import argparse
import numpy as np
from HyerIQASolver import HyperIQASolver
from utils import folder_path, img_num
def main(config):
    sel_num = img_num[config.predataset]

    srcc_all = np.zeros(config.train_test_num, dtype=np.float)
    plcc_all = np.zeros(config.train_test_num, dtype=np.float)

    for i in range(config.train_test_num):
        print('Round %d' % (i+1))
        # Randomly select 80% images for training and the rest for testing
        random.shuffle(sel_num)
        train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]

        solver = HyperIQASolver(config, folder_path[config.predataset], train_index, test_index)
        srcc_all[i], plcc_all[i] = solver.train()
        solver.save(config.predataset)

    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)

    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predataset', dest='predataset', type=str, default='koniq-10k', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=25, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=25, help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10, help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=96, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=16, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=10, help='Train-test times')
    config = parser.parse_args()
    print(f'Train HyperIQA on {config.predataset} dataset for {config.train_test_num} rounds...')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main(config)

