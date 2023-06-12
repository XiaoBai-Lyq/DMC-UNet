import torch, torchvision
from load_LIDC_data import LIDC_IDRI
from torch.utils.data import DataLoader,Dataset
import numpy as np
import UNett_batcnnorm
from torch import nn
from skimage import io
import os
from tqdm import tqdm
from utils import AverageMeter, str2bool
from collections import OrderedDict




from metrics1 import iou_score,precision_coef,recall_coef
from Swin_Unet_main.networks import vision_transformer
from UNetVit import MU_Vit_cam
import  torch.nn.functional as F
from unety import resunet_aspp_up_vit4
from unety import resunet
from unety.SAR_UNet import Se_PPP_ResUNet
import UNet_CA
from UNet3P.models import UNet_3Plus
from Swin_Unet_main.networks import vision_transformer
from UNetVit import U_Net_ASPP_SAM_skip_batchnormal
from UCTransNet.nets import UCTransNet
from unety import attention_unet
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from unety import CBAM_UNet
from unety import multiscale_unet
from unety import resunet
from CA_Net_master.Models.networks import network
from unety import resunet_attention
from unety import resunet_attentionunet2
from unety import resunet_fpn
from unety import resunet_sa1
from unety import resunet_ca2
from unety import resunet_aspp_bottom
from unety import resunet_hdc
from unety import resunet_aspp_up
from unety import resunet_aspp_Tranconv
from unety import resunet_aspp_up_vit
from ResUNet_family import res_unet_plus
from UNet3P.models import UNet_2Plus
from unety import resunet_aspp_up_vit2
from unety import resunet_aspp_up_vit3
from unety import resunet_aspp_up_vit4
from unety import resunet_vit2
from unety import resunet_aspp2_up_vit4
import DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change2

class dice_loss(nn.Module):  # dice损失，做反向传播
    def __init__(self,c_num=2):  # 格式需要
        super(dice_loss, self).__init__()
    def forward(self,data,label):  # 格式需要
        n = data.size(0)  # data.size(0)指 batch_size 的值，也就是一个批次几个
        dice_list = []  # 用来放本批次中的每一个图的dice
        all_dice = 0.  # 一会 算本批次的平均dice 用
        for i in range(n):  # 本批次内，拿一个图出来
            my_label11 = label[i]  # my_label11为取得的对应label，也可以说是前景为结节的分割图
            my_label1 = torch.abs(1 - my_label11)  # my_label1为 前景为非结节的分割图   label-label=0，label-0=label，这样就互换了

            my_data1 = data[i][0]  # my_data1为我的模型预测出的 前景为非结节的分割图
            my_data11 = data[i][1]  # my_data11为我的模型预测出的 前景为结节的分割图

            m1 = my_data1.view(-1)  # 把my_data1拉成一维       ps：前景为非结节的分割图
            m2 = my_label1.view(-1)  # 把my_label1拉成一维     ps：前景为非结节的分割图

            m11 = my_data11.view(-1)  # 把my_data1拉成一维     ps：前景为结节的分割图
            m22 = my_label11.view(-1)  # 把my_label1拉成一维   ps：前景为结节的分割图

            dice = 0  # dice初始化为0
            dice += (1-(( 2. * (m1 * m2).sum() +1 ) / (m1.sum() + m2.sum() +1)))  # dice loss = label-DSC的公式，比较的是 前景为非结节的分割图
            dice += (1-(( 2. * (m11 * m22).sum() + 1) / ( m11.sum()+m22.sum()+ 1)))  # dice loss = label-DSC的公式，比较的是 前景为结节的分割图
            dice_list.append(dice)  # 里面放本批次中的所有图的dice，每张图的dice为 前景结节 和 前景非结节 两图的dice loss 求和

        for i in range(n):  # 遍历本批次所有图
            all_dice += dice_list[i]  # 求和
        dice_loss = all_dice/n
        return dice_loss  # 返回本批次所有图的平均dice loss

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.image_path = os.listdir(self.root_dir)
        self.label_path = os.listdir(self.label_dir)
    def __getitem__(self, idx):  #如果想通过item去获取图片，就要先创建图片地址的一个列表
        img_name = self.image_path[idx]
        label_name = self.label_path[idx]
        img_item_path = os.path.join(self.root_dir, img_name)  # 每个图片的位置
        label_item_path = os.path.join(self.label_dir, label_name)
        image = np.load(img_item_path)
        image = torch.from_numpy(image)
        label = io.imread(label_item_path)
        label = torch.from_numpy(label)
        return image,label
    def __len__(self):
        return len(self.image_path)

# colormap = [[0,0,0], [1,1,1]]
# def label2image(prelabel,colormap):
#     #预测的标签转化为图像，针对一个标签图
#     _,h,w = prelabel.shape
#     prelabel = prelabel.reshape(h*w,-1)
#     image = np.zeros((h*w,3),dtype="int32")
#     for i in range(len(colormap)):
#         index = np.where(prelabel == i)
#         image[index,:] = colormap[i]
#     return image.reshape(h, w, 3)

pre_dir = 'testdata/DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change2'
img_dir = 'testdata/img_npy'
label_di = 'testdata/label'
dataset = MyData(img_dir, label_di)
test_loader = DataLoader(dataset, batch_size=1,shuffle=False)
# net = resunet_aspp_up_vit4.Unet(1,1)
# net = resunet.resunet(1,1)
# net = UNett_batcnnorm.Unet(1,1)
# net = attention_unet.AttU_Net(1, 1)
# config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
# net = ViT_seg(config_vit, img_size=128, num_classes=1)
# net = UNet_3Plus.UNet_3Plus(in_channels=1,n_classes=1)
# net = Se_PPP_ResUNet(1, 1, deep_supervision=False)
# net = vision_transformer.SwinUnet(img_size=128, num_classes=1)
#net = resunet_vit2.Unet(1,1)
# net= res_unet_plus.ResUnetPlusPlus(channel=1,out_channel=1)
# net = UNett_batcnnorm.Unet(1,1)
net = DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change2.resunet(1, 1)
net.load_state_dict(torch.load('checpoint/A_LIDC/DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change2_base/bestmodel_LIDC.pth'))
net.cuda()

predimg = []
labelimg = []
test_dice_need = []
acc = 0.0
def testdate(test_loader, net):
    avg_meters = {#'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'precision':AverageMeter(),
                      'recall':AverageMeter(),
                      'f1-score':AverageMeter()
        }
    net.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for step, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.float().cuda()
            labels = labels.long().cuda()
            inputs = torch.unsqueeze(inputs, dim=1)
            labels = torch.squeeze(labels).cpu()
            outputs = net(inputs)
            iou = iou_score(outputs, labels)
            precision = precision_coef(outputs, labels)
            recall = recall_coef(outputs, labels)
            f1_score = 2 * (precision * recall) / (precision + recall)

            preds = torch.sigmoid(outputs).data.cpu()
            preds = torch.squeeze(preds)
            predimg.append(preds)

            avg_meters['iou'].update(iou, inputs.size(0))
            avg_meters['precision'].update(precision, inputs.size(0))
            avg_meters['recall'].update(recall, inputs.size(0))
            avg_meters['f1-score'].update(f1_score, inputs.size(0))
            postfix = OrderedDict([
                # ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('precision', avg_meters['precision'].avg),
                ('recall', avg_meters['recall'].avg),
                ('f1-score', avg_meters['f1-score'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    return OrderedDict([  # ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('precision', avg_meters['precision'].avg),
            ('recall', avg_meters['recall'].avg),
            ('f1-score', avg_meters['f1-score'].avg)])


if __name__ == "__main__":
    test_log = testdate(test_loader,net)
    print('testdata IOU:{:.4f}, testdata precision:{:.4f}, testdata recall:{:.4f}, testdata f1-score:{:.4f}'.format(test_log['iou'], test_log['precision'],test_log['recall'],test_log['f1-score']))
    for i in range(len(predimg)):
        pre =predimg[i]
        if i < 10:
            test_pre_name = "000{}.png".format(i)
            #test_pre_np = "000{}.pth".format(i)
            clip_image_path = os.path.join(pre_dir, test_pre_name)
            io.imsave(clip_image_path, pre)
            # clip_label_path = os.path.join(label_dir, test_pre_name)
            # io.imsave(clip_label_path, label)
            # pre_np = os.path.join(pre_np_dir, test_pre_np)
            # torch.save(pre,pre_np)
        if i >= 10 and i < 100:
            test_pre_name = "00{}.png".format(i)
            #test_pre_np = "00{}.pth".format(i)
            clip_image_path = os.path.join(pre_dir, test_pre_name)
            io.imsave(clip_image_path,pre)
            # clip_label_path = os.path.join(label_dir, test_pre_name)
            # io.imsave(clip_label_path, label)
            # pre_np = os.path.join(pre_np_dir, test_pre_np)
            # torch.save(pre,pre_np)
        if i>=100 and i < 1000:
            test_pre_name = "0{}.png".format(i)
            #test_pre_np = "0{}.pth".format(i)
            clip_image_path = os.path.join(pre_dir, test_pre_name)
            io.imsave(clip_image_path,pre)
            # clip_label_path = os.path.join(label_dir, test_pre_name)
            # io.imsave(clip_label_path, label)
            # pre_np = os.path.join(pre_np_dir, test_pre_np)
            # torch.save(pre,pre_np)
        if i>=1000:
            test_pre_name = "{}.png".format(i)
            #test_pre_np = "{}.pth".format(i)
            clip_image_path = os.path.join(pre_dir, test_pre_name)
            io.imsave(clip_image_path,pre)
            # clip_label_path = os.path.join(label_dir, test_pre_name)
            # io.imsave(clip_label_path, label)
            # pre_np = os.path.join(pre_np_dir, test_pre_np)
            # torch.save(pre,pre_np)



