import torch
import models
import random               
import numpy as np
import torchvision
from torchvision.transforms import ToPILImage 

# model
model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
model_hyper.train(False)
# load our pre-trained model on the koniq-10k dataset
model_hyper.load_state_dict((torch.load('./pretrained/tid2013_pretrained.pkl')))

# Load VGG model
net = torchvision.models.vgg16(pretrained=True).cuda().features.eval()
# Feature Layers ID
convlayer_id = [4, 9, 16, 23, 30]
# Sample Rate
sr = np.array([64, 128, 256, 512, 512]) # 减少了特征图数量
# sr = np.array([32, 64, 256, 512, 512])

# 特征图用
transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 384)),
            torchvision.transforms.RandomCrop(size=224),
            torchvision.transforms.ToTensor()])

to_pil = ToPILImage()

class Hyper_IQA():
    def __init__(self):
        super(Hyper_IQA, self).__init__()

    def model(self, img):
        img = torch.tensor(img).cuda()
        # 随机计算十次计算平均scores
        pred_scores = []
        for i in range(10):
            paras = model_hyper(img)  # 'paras' contains the network weights conveyed to target network

            # Building target network
            model_target = models.TargetNet(paras).cuda()
            for param in model_target.parameters():
                param.requires_grad = False

            # Quality prediction
            pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
            pred_scores.append(float(pred.item()))
        score = np.mean(pred_scores)

        # print('Final quality score: %.2f' % final_score) 
        return score


class UIC_IQA():
    def __init__(self):
        super(UIC_IQA, self).__init__()

    def extractFeature(self, img):
        img = torch.tensor(img).cuda()
        feat_map = [img]
        cnt = 0
        for i, layer in enumerate(net.children()):
            img = layer(img)
            if i in convlayer_id:
                img0 = img
                for j in range(img0.shape[1]):
                    if j % sr[cnt] == 0:
                        # 假设img0的通道数为3，即最大索引为2
                        num_channels = img0.size(1) - 1
                        # 生成随机的三个通道索引（0到num_channels之间，包括0和num_channels）
                        random_channels = [random.randint(0, num_channels) for _ in range(3)]
                        temp = torch.cat([torch.tensor(transform(to_pil(img0[:, c,:,:]))).unsqueeze(1) for c in random_channels], dim=1).cuda()
                        feat_map.append(temp)
                cnt = cnt + 1
        return feat_map

    
    def model(self, img):
        # Extract feature map
        feat_map = self.extractFeature(img)
        pred_scores = []
        layer_scores = []
        # random crop 10 patches and calculate mean quality score
        for feat in feat_map:
            for i in range(10):
                paras = model_hyper(feat)  # 'paras' contains the network weights conveyed to target network

                # Building target network
                model_target = models.TargetNet(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
                pred_scores.append(float(pred.item()))
            score = np.mean(pred_scores)
            layer_scores.append(score)
            # quality score ranges from 0-100, a higher score indicates a better quality
            # print('Predicted quality score: %.2f' % score)
        final_score = np.mean(layer_scores)
        # print('Final quality score: %.2f' % final_score) 
        return layer_scores, final_score

