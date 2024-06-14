import os
import argparse
import data_loader
import numpy as np
from tqdm import tqdm
from iqa import IQA
from utils import calculate_sp, savedata_withlabel, folder_path, img_num

def main(config):
    dataLoader = data_loader.DataLoader(config.dataset, folder_path[config.dataset], img_num[config.dataset], patch_size = 224, patch_num = 1, istrain=False)
    data = dataLoader.get_data()
    
    method = IQA(config.predataset)

    if not config.curd:
        # 原方案
        pred_scores = []
        gt_scores = []
        for img, label in tqdm(data):
            score = method.Hyper_IQA(img)

            pred_scores.append(float(score.item()))
            gt_scores = gt_scores + label.tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, 1)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, 1)), axis=1)
        srcc, plcc = calculate_sp(pred_scores, gt_scores)
        print(f'Testing median SRCC {srcc},\tmedian PLCC {plcc}')
    else:
        # 新方案
        with open(f"./outputs/hyperIQA outputs/{config.dataset}_{config.predataset}.txt", "w") as file:
            for img, label in tqdm(data):
                layer_scores, _ = method.UIC_IQA(img)
                savedata_withlabel(file, layer_scores, float(label.numpy())) # 保存层分数
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='tid2013', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--predataset', dest='predataset', type=str, default='koniq-10k', help='Support datasets: koniq-10k|live|csiq|tid2013')
    parser.add_argument('--curd', action='store_true', help='The flag of using curd')
    config = parser.parse_args()
    print(f'Testing on {config.dataset} dataset, based on the model pretrained on {config.predataset}, CURD switch: {config.curd}.')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main(config)
