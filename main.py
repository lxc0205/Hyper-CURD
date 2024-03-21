import os
import argparse
import data_loader
import numpy as np
from tqdm import tqdm
from scipy import stats
from IQA import UIC_IQA, Hyper_IQA

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

folder_path = {
    'live': './Database/databaserelease2/',
    'csiq': './Database/CSIQ/',
    'tid2013': './Database/TID2013/',
    'livec': './Database/ChallengeDB_release/ChallengeDB_release/',
    'koniq-10k': './Database/koniq-10k/',
    'bid': './Database/BID/',
}

img_num = {
    'live': list(range(0, 29)),
    'csiq': list(range(0, 30)),
    'tid2013': list(range(0, 25)),
    'livec': list(range(0, 1162)),
    'koniq-10k': list(range(0, 10073)),
    'bid': list(range(0, 586)),
}

# IQA方法
U = UIC_IQA()
H = Hyper_IQA()

def saveLayerScore(file, layer_scores, label):
    for i in range(len(layer_scores)):
        file.write(str(layer_scores[i]))
        file.write('\t')
    file.write(str(float(label.numpy())))
    file.write('\t')
    file.write('\n')

def main(config):
    print('Testing on %s dataset' % (config.dataset))
    
    idx = img_num[config.dataset]

    dataLoader = data_loader.DataLoader(config.dataset, folder_path[config.dataset], idx, config.patch_size, config.patch_num, istrain=False)
    data = dataLoader.get_data()

    # 原方案
    # pred_scores = []
    # gt_scores = []
    # for img, label in tqdm(data): # FloatTensor [1, 3, 224, 224],  FloatTensor [1]

    #     score = H.model(img)

    #     pred_scores.append(float(score.item()))
    #     gt_scores = gt_scores + label.tolist()

    # pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, config.patch_num)), axis=1)
    # gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, config.patch_num)), axis=1)
    # srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    # plcc, _ = stats.pearsonr(pred_scores, gt_scores)

    # print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc, plcc))

            

    # 新方案
    pred_scores = []
    gt_scores = []
    with open("./outputs/" + config.dataset + ".txt", "w") as file:
        for img, label in tqdm(data):# FloatTensor [1, 3, 224, 224],  FloatTensor [1]
            # img = img.squeeze(0).cpu().numpy() # numpy (224, 224, 3) 

            layer_scores, score = U.model(img)
            saveLayerScore(file, layer_scores, label) # 保存层分数

            pred_scores.append(float(score.item()))
            gt_scores = gt_scores + label.tolist()

    pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, config.patch_num)), axis=1)
    gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, config.patch_num)), axis=1)
    srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    plcc, _ = stats.pearsonr(pred_scores, gt_scores)

    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc, plcc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='tid2013', help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--patch_num', dest='patch_num', type=int, default=1, help='Number of sample patches from testing image')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Crop size for training & testing image patches')

    config = parser.parse_args()
    main(config)

