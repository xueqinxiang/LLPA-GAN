import numpy as np
import cv2

import torchvision.models as models
import torchvision.transforms as transforms

import pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image

######################################################################
import networks.generator
import clip
import torch

if __name__ == "__main__":
    # 0.定义device
    torch.cuda.manual_seed_all(100)
    torch.cuda.set_device(1)
    device = torch.device("cuda")

    # 1.定义模型结构，选取要可视化的层
    #resnet18 = models.resnet18(pretrained=True)
    #resnet18.eval()
    #traget_layers = [resnet18.layer4[1].bn2]

    #1.1 clip model
    clip_info = {'src': "clip", 'type': 'ViT-B/32'}
    clip_model = clip.load(clip_info['type'], device=device)[0]
    clip_model = clip_model.eval()

    # 2.读取图片，将图片转为RGB
    origin_img = cv2.imread('./bird.jpg')
    rgb_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)

    # 3.图片预处理：resize、裁剪、归一化
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.CenterCrop(224)
    ])
    crop_img = trans(rgb_img)
    net_input = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(crop_img).unsqueeze(0)

    # 4.将裁剪后的Tensor格式的图像转为numpy格式，便于可视化
    canvas_img = (crop_img*255).byte().numpy().transpose(1, 2, 0)
    canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_RGB2BGR)

    # 5.实例化cam
    cam = pytorch_grad_cam.GradCAMPlusPlus(model=resnet18, target_layers=traget_layers, use_cuda=False)
    grayscale_cam = cam(net_input)
    grayscale_cam = grayscale_cam[0, :]

    # 6.将feature map与原图叠加并可视化
    src_img = np.float32(canvas_img) / 255
    visualization_img = show_cam_on_image(src_img, grayscale_cam, use_rgb=False)
    cv2.imshow('feature map', visualization_img)
    cv2.waitKey(0)