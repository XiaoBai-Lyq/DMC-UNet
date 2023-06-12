import pandas as pd
import argparse
import os
from collections import OrderedDict
import yaml
from load_LIDC_data import LIDC_IDRI
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import albumentations as albu
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from losses import BCEDiceLoss
from metrics1 import iou_score,dice_coef
from utils import AverageMeter, str2bool
from unety import UNett_batcnnorm
from torch.utils.data import DataLoader
from UNetVit import MU_Vit_cam
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
from unety import resunet_aspp_up_vit  #最好的
from ResUNet_family import res_unet_plus
from UNet3P.models import UNet_2Plus
from unety import resunet_aspp_up_vit2
from unety import resunet_aspp_up_vit3
from unety import resunet_aspp_up_vit4
from unety import resunet_vit2
from unety import resunet_aspp2_up_vit4
import DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change2

def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--name', default="TransUNet",
                        help='model name: UNET',choices=['UNET', 'MU_Vit_cam', 'SAR_UNet', 'UNet_CA_bottole', 'UNet3P', 'swinunet', 'SA_UNet', 'UCTransNet',
                                                         'attention_unet', 'TransUNet', 'CBAM_UNet', 'multiscale_unet', 'resunet',
                                                         'CA_Net','resunet_attention2','resunet_bifpn','resunet_sa1','resunet_ca2','resunet_aspp_bottom','resunet_hdc',
                                                         'resunet_aspp_up','resunet_aspp_tranconv','resunet_aspp_up_vit','resunet_plus','unet2p','resunet_aspp_up_vit2',
                                                         'resunet_aspp_up_vit3', 'resunet_aspp_up_vit4','resunet_vit2','resunet_aspp2_up_vit4','DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change2'])
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 6)')
    parser.add_argument('--early_stopping', default=50, type=int,
                        metavar='N', help='early stopping (default: 50)')
    parser.add_argument('--num_workers', default=8, type=int)
    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    # data
    parser.add_argument('--augmentation',type=str2bool,default=False,choices=[True,False])
    config = parser.parse_args()

    return config

def train(train_loader, model, criterion, optimizer,scheduler):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}
    model.train()
    pbar = tqdm(total=len(train_loader))
    for input, target,_ in train_loader:
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        output = torch.squeeze(output)
        target = torch.squeeze(target)
        loss = criterion(output, target)
        iou = iou_score(output, target)
        dice = dice_coef(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice',avg_meters['dice'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    # scheduler.step()
    # lr_exp = scheduler.get_last_lr()[0]
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice',avg_meters['dice'].avg)])

def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            output = torch.squeeze(output)
            target = torch.squeeze(target)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice',avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice',avg_meters['dice'].avg)])

def main():
    # Get configuration
    config = vars(parse_args())
    # Make Model output directory
    if config['augmentation']== True:
        file_name= config['name'] + '_with_augmentation'
    else:
        file_name = config['name'] +'_base'
    os.makedirs('checpoint/A_LIDC/{}'.format(file_name),exist_ok=True)
    print("Creating directory called",file_name)

    print('-' * 20)
    print("Configuration Setting as follow")
    for key in config:
        print('{}: {}'.format(key, config[key]))
    print('-' * 20)

    #save configuration
    with open('checpoint/A_LIDC/{}/config.yml'.format(file_name), 'w') as f:
        yaml.dump(config, f)

    #criterion = nn.BCEWithLogitsLoss().cuda()
    criterion = BCEDiceLoss().cuda()
    cudnn.benchmark = True

    # create model
    print("=> creating model" )
    if config['name']=='MU_Vit_cam':
        model = MU_Vit_cam.mu_vit(in_ch=1, out_ch=1,channel_num=[64,128, 256, 512], patchSize=[8, 4, 2, 1],img_size=[128,64,32,16])
    elif config['name']=='SAR_UNet':
        model = Se_PPP_ResUNet(1, 1, deep_supervision=False)
    elif config['name'] == 'UNet_CA_bottole':
        model = UNet_CA.Unet(1,1)
    elif config['name'] == 'UNet3P':
        model = UNet_3Plus.UNet_3Plus(in_channels=1,n_classes=1)
    elif config['name'] == 'unet2p':
        model = UNet_2Plus.UNet_2Plus(in_channels=1,n_classes=1)
    elif config['name'] == 'swinunet':
        model = vision_transformer.SwinUnet(img_size=128, num_classes=1)
    elif config['name'] == 'SA_UNet':
        model = U_Net_ASPP_SAM_skip_batchnormal.Unet(1,1)
    elif config ['name'] =='UCTransNet':
        model = UCTransNet.UCTransNet(n_channels=1, n_classes=1, img_size=128, vis=False)
    elif config['name'] == 'attention_unet':
        model = attention_unet.AttU_Net(1, 1)
    elif config ['name'] == 'TransUNet':
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        model = ViT_seg(config_vit, img_size=128, num_classes=1)
    elif config['name'] == 'CBAM_UNet':
        model = CBAM_UNet.Unet(1, 1, channel_num=[64,128,256,512])
    elif config['name'] == 'UNET':
        model = UNett_batcnnorm.Unet(1,1)
    elif config['name'] == 'multiscale_unet':
        model = multiscale_unet.Unet(1,1)
    elif config['name'] == 'resunet':
        model = resunet.resunet(1,1)
    elif config['name'] == 'CA_Net':
        model = network.Comprehensive_Atten_Unet(in_ch=1, n_classes=1)
    elif config['name'] == 'resunet_attention2':
        model = resunet_attentionunet2.AttU_Net(1,1)
    elif config['name'] == 'resunet_bifpn':
        model = resunet_fpn.resunet(1, 1)
    elif config['name'] == 'resunet_sa1':
        model = resunet_sa1.resunet_sa1(1, 1)
    elif config['name'] == 'resunet_ca2':
        model = resunet_ca2.resunet_ca2(1,1)
    elif config['name'] == 'resunet_aspp_bottom':
        model = resunet_aspp_bottom.resunet(1,1)
    elif config['name']  == 'resunet_hdc':
        model = resunet_hdc.resunet(1,1)
    elif config['name']  == 'resunet_aspp_up':
        model = resunet_aspp_up.resunet(1,1)
    elif config['name'] == 'resunet_aspp_tranconv':
        model = resunet_aspp_Tranconv.resunet(1,1)
    elif config['name'] == 'resunet_aspp_up_vit':
        model = resunet_aspp_up_vit.Unet(1,1)
    elif config['name'] == 'resunet_plus':
        model = res_unet_plus.ResUnetPlusPlus(channel=1,out_channel=1)
    elif config['name'] == 'resunet_aspp_up_vit2':
        model = resunet_aspp_up_vit2.Unet(1,1)
    elif config['name'] == 'resunet_aspp_up_vit3':
        model = resunet_aspp_up_vit3.Unet(1,1)
    elif config['name'] == 'resunet_aspp_up_vit4':
        model = resunet_aspp_up_vit4.Unet(1,1)
    elif config['name'] == 'resunet_vit2':
        model = resunet_vit2.Unet(1,1)
    elif config['name'] == 'resunet_aspp2_up_vit4':
        model = resunet_aspp2_up_vit4.Unet(1, 1)
    elif config['name']=="DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change2":
        model  = DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change2.resunet(1,1)
    else:
        raise ValueError("Wrong Parameters")
    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    dataset = LIDC_IDRI(dataset_location='H:\\ProbabilisticUnetPytorchmaster\\data\\')
    train_sampler = torch.load('sampler/bingji/train_sampler.pth')
    test_sampler = torch.load('sampler/bingji/test_sampler.pth')
    val_sampler = torch.load('sampler/bingji/val_sampler.pth')
    train_loader = DataLoader(dataset, batch_size=16, sampler= train_sampler,shuffle=False)
    test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler,shuffle=False)
    val_loader = DataLoader(dataset, batch_size=16, sampler=val_sampler,shuffle=False)

    log= pd.DataFrame(index=[],columns= ['epoch','lr','loss','iou','dice','val_loss','val_iou'])

    best_dice = 0
    trigger = 0

    for epoch in range(config['epochs']):

        # train for one epoch
        train_log = train(train_loader, model, criterion, optimizer,scheduler=exp_lr_scheduler)
        # evaluate on validation set
        val_log = validate(val_loader, model, criterion)

        print('Training epoch [{}/{}], Training BCE loss:{:.4f}, Training DICE:{:.4f}, Training IOU:{:.4f}, Validation BCE loss:{:.4f}, Validation Dice:{:.4f}, Validation IOU:{:.4f}'.format(
            epoch + 1, config['epochs'], train_log['loss'], train_log['dice'], train_log['iou'], val_log['loss'], val_log['dice'],val_log['iou']))

        tmp = pd.Series([
            epoch,
            config['lr'],
            #train_log['lr_exp'],
            train_log['loss'],
            train_log['iou'],
            train_log['dice'],
            val_log['loss'],
            val_log['iou'],
            val_log['dice']
        ], index=['epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou','val_dice'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('checpoint/A_LIDC/{}/log.csv'.format(file_name), index=False)

        trigger += 1

        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), 'checpoint/A_LIDC/{}/bestmodel_LIDC.pth'.format(file_name))
            best_dice = val_log['dice']
            print("=> saved best model as validation DICE is greater than previous best DICE")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
