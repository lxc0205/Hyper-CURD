import torch
import models
import random               
import numpy as np
from torchvision import transforms
import torchvision

# Load VGG model
# net = models.vgg16(pretrained=True).cuda().features.eval()
net = torchvision.models.vgg16(weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1).cuda().features.eval()
# Feature Layers ID
# convlayer_id = [4, 9, 16, 23, 30]
convlayer_id = [0, 2, 5, 7, 10]
# Sample Rate
sr = np.array([64, 128, 256, 512, 512])

# 特征图用
transform = transforms.Compose([
            transforms.Resize((512, 384)),
            transforms.RandomCrop(size=224),
            transforms.ToTensor()])

to_pil = transforms.ToPILImage()

class IQA():
    def __init__(self, dataset):
        super(IQA, self).__init__()
        # load our pre-trained model on the pretrained dataset
        self.model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        self.model_hyper.train(False)
        self.model_hyper.load_state_dict((torch.load('./outputs/pretrained/' + dataset + '_pretrained.pkl')))

    def HyperIQA(self, img):
        paras = self.model_hyper(img)  # 'paras' contains the network weights conveyed to target network

        # Building target network
        model_target = models.TargetNet(paras).cuda()
        for param in model_target.parameters():
            param.requires_grad = False

        # Quality prediction
        pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
        return pred

    def extractFeature(self, img):
        img = torch.as_tensor(img).cuda()
        feat_map = [img]
        cnt = 0
        for i, layer in enumerate(net.children()):
            img = layer(img)
            if i in convlayer_id:
                img0 = img
                for j in range(img0.shape[1]):
                    if j % sr[cnt] == 0:
                        random_channels = [random.randint(0, img0.size(1) - 1) for _ in range(3)] # 生成随机的三个通道索引（0到num_channels之间，包括0和num_channels）
                        temp = torch.cat([torch.as_tensor(transform(to_pil(img0[:, c,:,:]))).unsqueeze(1) for c in random_channels], dim=1).cuda()
                        feat_map.append(temp)
                cnt = cnt + 1
        return feat_map
    
    def Hyper_IQA(self, img):
        img = torch.tensor(img).cuda()
        pred_scores = []
        for _ in range(10):
            pred = self.HyperIQA(img)
            pred_scores.append(float(pred.item()))
        return np.mean(pred_scores)

    def UIC_IQA(self, img):
        feat_map = self.extractFeature(img) # Extract feature map
        pred_scores = []
        layer_scores = []
        for feat in feat_map:
            for _ in range(10):
                pred = self.HyperIQA(feat)
                pred_scores.append(float(pred.item()))
            score = np.mean(pred_scores)
            layer_scores.append(score)
        final_score = np.mean(layer_scores)
        return layer_scores, final_score

