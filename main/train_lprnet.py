# -*- coding: utf-8 -*-
import os
import sys

try:
    root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("ALPR")+1])
    sys.path.append(root)
    os.chdir(root)
except:
    pass 

from dataset.load_data import CHARS, CHARS_DICT, LPRDataLoader
from model.LPRNet import build_lprnet
from torch.autograd import Variable
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import matplotlib.pyplot as plt 
from torch.optim.lr_scheduler import CosineAnnealingLR
import swanlab

swanlab.login(api_key="你的swanlab api_key", save=True)

def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', type=int, default=60, help='epoch to train the network')
    parser.add_argument('--img_size', default=[188,48], help='the image size')
    
    parser.add_argument('--train_img_dirs', type=str, default="./dataset/ccpd_lprnet_balanced/train", help='the train images path')
    parser.add_argument('--test_img_dirs', type=str, default="./dataset/ccpd_lprnet_balanced/val", help='the test images path')
    
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='base value of learning rate.') # 建议改回 0.001，0.005可能有点大
    parser.add_argument('--lpr_max_len', type=int, default=8, help='license plate number max length.')
    parser.add_argument('--train_batch_size', type=int, default=64, help='training batch size.')
    parser.add_argument('--test_batch_size', type=int, default=64, help='testing batch size.')
    parser.add_argument('--phase_train', default=True, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_interval', default=5000, type=int, help='interval for save model state dict')
    parser.add_argument('--test_interval', default=1000, type=int, help='interval for evaluate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    parser.add_argument('--lr_schedule', default=[20, 40], help='schedule for learning rate.')
    parser.add_argument('--save_folder', default='./weight/', help='Location to save checkpoint models')
    parser.add_argument('--result_folder', default='./result/', help='Location to save visualize results')
    parser.add_argument('--pretrained_model', default='', help='pretrained base model')

    args = parser.parse_args()
    return args

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.int64)
    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def train():
    args = get_parser()

    swanlab.init(
        project="LPRNet-Training",
        experiment_name="LPRNet_CTC_Training_FixLength",
        config={
            "learning_rate": args.learning_rate,
            "max_epoch": args.max_epoch,
            "batch_size": args.train_batch_size,
            "img_size": args.img_size,
            "optimizer": "Adam",
        }
    )

    epoch = 0 + args.resume_epoch
    loss_val = 0

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    lprnet.to(device)
    print("Successful to build network!")

    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        print("load pretrained model successful!")
    else:
        def xavier(param):
            nn.init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = xavier(1)
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0.01

        lprnet.backbone.apply(weights_init)
        lprnet.container.apply(weights_init)
        print("initial net weights successful!")

    train_img_dirs = args.train_img_dirs.split(',')
    test_img_dirs = args.test_img_dirs.split(',')
    
    train_dataset = LPRDataLoader(train_img_dirs, args.img_size, args.lpr_max_len)
    test_dataset = LPRDataLoader(test_img_dirs, args.img_size, args.lpr_max_len)

    epoch_size = len(train_dataset) // args.train_batch_size
    max_iter = args.max_epoch * epoch_size

    optimizer = optim.Adam(lprnet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=1e-6)
    
    # reduction='mean'
    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') 

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    history = {'loss': [], 'acc': [], 'iter': []}

    print("Start Training...")
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            batch_iterator = iter(DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
            loss_val = 0
            epoch += 1

        if iteration != 0 and iteration % args.save_interval == 0:
            torch.save(lprnet.state_dict(), args.save_folder + 'LPRNet_' + '_iteration_' + repr(iteration) + '.pth')

        if (iteration + 1) % args.test_interval == 0:
            acc = Greedy_Decode_Eval(lprnet, test_dataset, args, device)
            lprnet.train()
            swanlab.log({"val/accuracy": acc}, step=iteration)
            history['acc'].append(acc)
            history['iter'].append(iteration)
            torch.save(lprnet.state_dict(), args.save_folder + 'current_lprnet.pth')

        start_time = time.time()
        
        try:
            images, labels, lengths = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
            images, labels, lengths = next(batch_iterator)

        images = Variable(images, requires_grad=False).to(device)
        labels = Variable(labels, requires_grad=False).to(device)

        # Forward
        logits = lprnet(images) # Shape: [Batch, Class, Time]
        
        # === 核心修改：动态获取时间步长 ===
        # 不要硬编码 36 或 18，直接问 logits 它是多少
        current_T_length = logits.shape[2] 
        input_lengths, target_lengths = sparse_tuple_for_ctc(current_T_length, lengths)
        # =================================

        log_probs = logits.permute(2, 0, 1) 
        log_probs = log_probs.log_softmax(2).requires_grad_()
        
        optimizer.zero_grad()
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        
        if loss.item() == np.inf:
            continue
            
        loss.backward()
        nn.utils.clip_grad_norm_(lprnet.parameters(), max_norm=5)
        optimizer.step()
        scheduler.step()
        
        loss_val += loss.item()
        end_time = time.time()
        
        current_lr = optimizer.param_groups[0]['lr']
        swanlab.log({
            "train/loss": loss.item(),
            "train/learning_rate": current_lr,
            "train/epoch": epoch
        }, step=iteration)
        
        if iteration % 20 == 0:
            print('Epoch:' + repr(epoch) + ' || iter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + ' || Loss: %.4f ||' % (loss.item()) + ' || Time: %.4f' % (end_time - start_time) + ' || T_len: %d' % current_T_length)
            
            history['loss'].append(loss.item())

    # Final logic...
    print("Final test Accuracy:")
    final_acc = Greedy_Decode_Eval(lprnet, test_dataset, args, device)
    swanlab.log({"final/accuracy": final_acc})
    torch.save(lprnet.state_dict(), args.save_folder + 'Final_LPRNet_model.pth')
    
    plt.figure()
    plt.plot(history['loss'])
    plt.title('LPRNet Training Loss')
    plt.savefig(os.path.join(args.result_folder, 'lprnet_loss_visualize.png'))
    
    if len(history['acc']) > 0:
        plt.figure()
        plt.plot(history['iter'], history['acc'])
        plt.title('LPRNet Validation Accuracy')
        plt.savefig(os.path.join(args.result_folder, 'lprnet_acc_visualize.png'))
    
    swanlab.finish()
    print("Training finished.")

def Greedy_Decode_Eval(Net, datasets, args, device):
    Net = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    for i in range(epoch_size):
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets], dtype=object)

        images = Variable(images.to(device))

        prebs = Net(images)
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: 
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        
        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    return Acc

if __name__ == "__main__":
    train()