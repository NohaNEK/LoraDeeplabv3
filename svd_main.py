from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from synthia_dataloader import SynthiaDataSet
from torchviz import make_dot

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, GTA, SYNTHESIA,BDD , MP
from utils import ext_transforms as et
from metrics import StreamSegMetrics
import pandas as pd
from peft import LoHaModel, LoHaConfig
import torch
import torch.nn as nn
from utils.visualizer import Visualizer
from svdiff_pytorch.layers import SVDConv2d
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import cv2
from tensorboardX import SummaryWriter
import pandas as pd
from torch.cuda.amp import autocast 
import time

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='/media/fahad/Crucial X81/datasets/cityscapes/',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=100e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=6,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=6,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=768)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=10,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=1000,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            et.ExtResize(size= (1914,1052) ),
            et.ExtRandomCrop(size=(768,768)),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
        ])
     
        val_transform = et.ExtCompose([
            et.ExtResize( (768,768)  ),
            et.ExtToTensor(),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        # train_dst = Cityscapes(root='/media/fahad/DATA_2/cityscapes dataset/',
        #                 split='train', transform=train_transform)
        # img_names = [file for file in os.listdir("/home/fahad/Documents/synthia/RGB/") if file.endswith('.png')]
        # train_dst =SynthiaDataSet(root="/home/fahad/Documents/synthia/",list_path=img_names,img_size=(768,768))
        val_dst = Cityscapes(root='/media/fahad/Crucial X81/datasets/cityscapes/',
                        split='val', transform=val_transform)
        # val_dst = GTA(root=opts.data_root,
        #                        split='val', transform=train_transform)
    return train_dst, val_dst

def add_gta_infos_in_tensorboard(writer,imgs,labels,outputs,cur_itrs,denorm,train_loader):
        # img=imgs[0].detach().cpu().numpy()
        # img=(img*255).astype(np.uint8)
        # writer.add_image('gta_image',img,cur_itrs,dataformats='CHW')

        i=0
        img=imgs[i].detach().cpu().numpy()
        print(img.shape)
        img=(denorm(img)*255).astype(np.uint8)
        print(img.shape)
        writer.add_image('SYN_image',img,cur_itrs,dataformats='CHW')
      

        lbs=labels[i].detach().cpu().numpy()
        print(lbs.shape)
        lbs=train_loader.dataset.decode_target(lbs).astype('uint8')
        writer.add_image('SYN_ground_truth',lbs,cur_itrs,dataformats='HWC')

        pred=outputs.detach().max(1)[1].cpu().numpy()
        pred = train_loader.dataset.decode_target(pred[0]).astype('uint8')
        writer.add_image('SYN_pred',pred,cur_itrs,dataformats='HWC')
        
def add_cs_in_tensorboard(writer,imgs,labels,outputs,cur_itrs,denorm,train_loader,i):
    if imgs[i] == None :
        print("img none", i)
    # print(imgs[i])

    img=imgs[i].detach().cpu().numpy()
    print(img.shape)

    img=(denorm(img)*255).astype(np.uint8)
    lbs=labels[i].detach().cpu().numpy()
    lbs=train_loader.dataset.decode_target(lbs).astype('uint8')
    pred=outputs.detach().max(1)[1].cpu().numpy()
    pred=train_loader.dataset.decode_target(pred[i]).astype('uint8')


    res_grid=[img,np.transpose(lbs,(2,0,1)),np.transpose(pred,(2,0,1))]
    
    writer.add_images('test sample cityscapes '+str(i),res_grid,cur_itrs,dataformats='CHW')
def create_colormap(feat):
    # cmap= plt.get_cmap('viridis')
    cmap= plt.get_cmap('jet')

    feat_map = cmap(feat)

    feat_map=(feat_map*255).astype(np.uint8)
    return feat_map
def validate(opts, model, loader, device, metrics,denorm=None,writer=None, cur_itrs=0,ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)


            outputs,_ = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            # if i <4 :
            #     add_cs_in_tensorboard(writer,images,labels,outputs,cur_itrs,denorm,loader,i)
#            add_cs_in_tensorboard(writer,images,labels,outputs,cur_itrs,denorm,loader,0)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples

def writer_add_features(writer, name, tensor_feat, iterations):
    feat_img = tensor_feat[0].detach().cpu().numpy()
    # img_grid = self.make_grid(feat_img)
    feat_img = np.sum(feat_img,axis=0)
    feat_img = feat_img -np.min(feat_img)
    img_grid = 255*feat_img/np.max(feat_img)
    img_grid = cv2.applyColorMap(np.array(img_grid, dtype=np.uint8), cv2.COLORMAP_JET)
    writer.add_image(name, img_grid, iterations, dataformats='HWC')


def collect_target_modules(model, include_bn=False):
    target_modules = []
    saved_modules=[]
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            target_modules.append(name )
        elif include_bn and isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            saved_modules.append(name )
            saved_modules.append(name )
    return target_modules,saved_modules

def replace_all_conv_layers(module, scale=1.0):
    for name, layer in module.named_children():
        # Recursively apply to nested modules (like layers in resnet)
        if isinstance(layer, nn.Conv2d):
            # Replace Conv2d layer with SVDConv2d
            svd_conv = SVDConv2d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size[0],
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=layer.bias is not None,
                scale=scale
            )
            with torch.no_grad():  # Disable gradient tracking
                    svd_conv.weight.data.copy_(layer.weight.data)
                    if layer.bias is not None:
                        svd_conv.bias.copy_(layer.bias.data)
            setattr(module, name, svd_conv)
        else:
            replace_all_conv_layers(layer, scale)

def replace_first_conv_of_each_layer(module, scale=1.0):
    for name, layer in module.named_children():
        # Flag to check if the first Conv2d has been replaced in this layer/block
        first_conv_replaced = False
        
        # If the current layer has nested modules, recursively apply
        for sub_name, sub_layer in layer.named_children():
            if isinstance(sub_layer, nn.Conv2d) and not first_conv_replaced:
                # Replace only the first Conv2d layer in the module
                svd_conv = SVDConv2d(
                    in_channels=sub_layer.in_channels,
                    out_channels=sub_layer.out_channels,
                    kernel_size=sub_layer.kernel_size[0],
                    stride=sub_layer.stride,
                    padding=sub_layer.padding,
                    dilation=sub_layer.dilation,
                    groups=sub_layer.groups,
                    bias=sub_layer.bias is not None,
                    scale=scale
                )
                with torch.no_grad():  # Disable gradient tracking
                    svd_conv.weight.data.copy_(sub_layer.weight.data)
                    if sub_layer.bias is not None:
                        svd_conv.bias.copy_(sub_layer.bias.data)
                setattr(layer, sub_name, svd_conv)
                first_conv_replaced = True
            else:
                # Recurse into submodules
                replace_first_conv_of_each_layer(sub_layer, scale)


def freeze_all_except_delta(model):
    for name, param in model.named_parameters():
        # Check if the parameter is 'delta' in an SVDConv2d layer
        if 'delta' in name:
            param.requires_grad = True  # Keep gradient updates for delta
        else:
            param.requires_grad = False

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19 #19

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    writer = SummaryWriter("/media/fahad/DATA_2/ckpt_sd_________ddd/R101_CS_svd_all_conv2d")

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    ckpt_p="/media/fahad/DATA_2/checkpoints/ckpts/checkpoints_baseline/latest_deeplabv3plus_resnet101_cityscapes_os16.pth"
    checkpoint = torch.load(ckpt_p, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"before svd Total trainable parameters: {trainable_params}")

    
    
    replace_all_conv_layers(model.backbone,scale=1.0)
    freeze_all_except_delta(model.backbone)


  

    # for name, param in model.backbone.named_parameters():
    #     # param.requires_grad = True
    #     for name, param in model.named_parameters():
    #         if '.base_layer' in name:
    #             param.requires_grad = False  # Freeze this layer
    #             print("non trainable", name)
    #         else:
    #             param.requires_grad = True
    for name, param in model.backbone.named_parameters():
        # param.requires_grad = True
        for name, param in model.backbone.named_parameters():
            if param.requires_grad == True:
              
                print(" trainable", name)
           

            
        
    trainable_params2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"after svd Total trainable parameters: {trainable_params2}")
    print(f"percentage :{trainable_params2/trainable_params*100} %")

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)
    m_cls ={
        'road':0,
        'sidewalk':1,
        'building':2,
        'wall':3,
        'fence':4,
        'pole':5,
        'traffic light':6,
        'traffic sign':7,
        'vegetation':8,
        'terrain':9,
        'sky':10,
        'person':11,
        'rider':12,
        'car':13,
        'truck':14,
        'bus':15,
        'train':16,
        'motorcycle':17,
        'bicycle':18,
    }
    # Reverse the mapping to create a new dictionary
    index_to_class = {v: k for k, v in m_cls.items()}

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints_r101_cs_svd')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
       #writer = SummaryWriter("/media/fahad/Crucial X8/deeplabv3plus/original_baseline/logs/R101")

        # model.eval()
        print('val len',len(val_loader))
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id,writer=writer)
        print(metrics.to_str(val_score))
        print(val_score['Class IoU'])
        return

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)

            labels = labels.to(device, dtype=torch.long)
            with autocast():
                # start_time = time.time() 
                optimizer.zero_grad()
                outputs,_ = model(images)
                # dot = make_dot(outputs, params=dict(model.module.named_parameters()))
                # dot.render("forward_graph_"+str(cur_itrs), format="png")
                 
                loss = criterion(outputs, labels)
                loss.backward()
                # backward_dot = make_dot(loss, params=dict(model.module.named_parameters()))
                # backward_dot.render("backward_graph_"+str(cur_itrs), format="png")
                optimizer.step()
                # end_time = time.time()
                # execution_time = end_time - start_time
                # print(f"Execution time: {execution_time:.6f} seconds")
                # for name, param in model.module.backbone.named_parameters():
                #     if param.grad is None:
                #         print(f"Parameter {name} has no gradient after backward pass")

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                writer.add_scalar('train_image_loss', interval_loss, cur_itrs)
                interval_loss = 0.0
                
            if (cur_itrs) % 100 == 0: 
                # interval_loss=interval_loss/100
                # writer.add_scalar('train_image_loss', interval_loss, cur_itrs)
                # interval_loss = 0.0
                add_gta_infos_in_tensorboard(writer,images,labels,outputs,cur_itrs,denorm,train_loader)
              
               
            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints_r101_cs_svd/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                # model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,denorm=denorm,writer=writer,cur_itrs=cur_itrs,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints_r101_cs_svd/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))
                writer.add_scalar('mIoU_cs', val_score['Mean IoU'], cur_itrs)
                writer.add_scalar('overall_acc_cs',val_score['Overall Acc'],cur_itrs)
                print(val_score['Class IoU'])

                writer.add_text("Class IoU (cs )",str(val_score['Class IoU']),cur_itrs)
                for cl in range(len(val_score['Class IoU'])):
                    #print(cl)
                    cl_name=index_to_class[cl]
                    writer.add_scalar('class IoU (cs)/'+cl_name,val_score['Class IoU'][cl],cur_itrs)

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    main()
