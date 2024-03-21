import torch
import torchvision
import models
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

# feature map net
net = torchvision.models.vgg16(pretrained=True).cuda().features.eval()

# evalue method net
model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
model_hyper.train(False)
model_hyper.load_state_dict((torch.load('./pretrained/tid2013_pretrained.pkl')))

# image transforms
transform1 = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 384)),
            torchvision.transforms.RandomCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                std=(0.229, 0.224, 0.225))
            ])

transform2 = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 384)),
            torchvision.transforms.RandomCrop(size=224),
            torchvision.transforms.ToTensor()])

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def extractFeature(imgx): 
    to_pil = ToPILImage()
    layerId = [4, 9, 16, 23, 30]
    sr = np.array([32, 64, 256, 512, 512])
    img = transforms.ToTensor()(imgx).unsqueeze(0).cuda() # B C H W = 1 3 1536 2048
    imgx = transform1(imgx)
    imgx = torch.tensor(imgx.cuda()).unsqueeze(0)
    feat_map = [imgx]
    cnt = 0
    for i, layer in enumerate(net.children()):
        img = layer(img)
        if i in layerId:
            img0 = img
            for j in range(img0.shape[1]):
                if j % sr[cnt] == 0:
                    temp  = img0[:,0,:,:].unsqueeze(1)
                    temp = to_pil(temp.squeeze(1))
                    temp = transform2(temp)
                    temp = torch.tensor(temp).unsqueeze(1)
                    temp = torch.cat([temp, temp, temp], dim = 1).cuda()
                    feat_map.append(temp)
            cnt = cnt + 1
    return feat_map

im_path = './data/D_01.jpg'
img = pil_loader(im_path) # PIL
feat_map = extractFeature(img) # Tensor [1 1 h w]
pred_scores = []
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
    pred_scores.append(score)
    # quality score ranges from 0-100, a higher score indicates a better quality
    print('Predicted quality score: %.2f' % score)
final_score = np.mean(pred_scores)
print('Final quality score: %.2f' % final_score)



