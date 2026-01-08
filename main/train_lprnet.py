# -*- coding: utf-8 -*-
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
import os
import matplotlib.pyplot as plt # 引入绘图库

def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)

def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    lr = 0
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    if lr == 0:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    # === 关键修改区域 ===
    parser.add_argument('--max_epoch', default=50, help='epoch to train the network') # 建议由15改为50
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    
    # 指向你的数据集路径
    parser.add_argument('--train_img_dirs', default="./dataset/ccpd_lprnet/train", help='the train images path')
    parser.add_argument('--test_img_dirs', default="./dataset/ccpd_lprnet/val", help='the test images path')
    
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--learning_rate', default=0.01, help='base value of learning rate.') # 降低一点初始LR
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--train_batch_size', default=64, help='training batch size.') # 显存如果不够改小
    parser.add_argument('--test_batch_size', default=64, help='testing batch size.')
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
    labels = np.asarray(labels).flatten().astype(np.int)
    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def train():
    args = get_parser()

    # T_length = 18 
    # 注意：LPRNet的输出宽度取决于输入宽度。输入94时，输出大约是18。
    # 如果遇到 CTC Loss 报错，可能需要微调这个值，但标准 LPRNet 94x24 输出就是 18
    T_length = 18 
    
    epoch = 0 + args.resume_epoch
    loss_val = 0

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)

    # 初始化模型
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    lprnet.to(device)
    print("Successful to build network!")
    print(f"Class Num: {len(CHARS)} (Includes blank)")

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

    optimizer = optim.RMSprop(lprnet.parameters(), lr=args.learning_rate, alpha = 0.9, eps=1e-08,
                         momentum=args.momentum, weight_decay=args.weight_decay)
    
    # 路径处理
    train_img_dirs = args.train_img_dirs.split(',')
    test_img_dirs = args.test_img_dirs.split(',')
    
    train_dataset = LPRDataLoader(train_img_dirs, args.img_size, args.lpr_max_len)
    test_dataset = LPRDataLoader(test_img_dirs, args.img_size, args.lpr_max_len)

    epoch_size = len(train_dataset) // args.train_batch_size
    max_iter = args.max_epoch * epoch_size

    # reduction='mean'
    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') 

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    # === 用于记录可视化数据 ===
    history = {'loss': [], 'acc': [], 'iter': []}

    print("Start Training...")
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            batch_iterator = iter(DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
            loss_val = 0
            epoch += 1

        if iteration !=0 and iteration % args.save_interval == 0:
            torch.save(lprnet.state_dict(), args.save_folder + 'LPRNet_' + '_iteration_' + repr(iteration) + '.pth')

        # 测试与评估
        if (iteration + 1) % args.test_interval == 0:
            acc = Greedy_Decode_Eval(lprnet, test_dataset, args, device)
            lprnet.train() # 切换回训练模式
            
            # 记录数据用于画图
            history['acc'].append(acc)
            history['iter'].append(iteration)
            # 保存当前模型
            torch.save(lprnet.state_dict(), args.save_folder + 'current_lprnet.pth')

        start_time = time.time()
        
        try:
            images, labels, lengths = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
            images, labels, lengths = next(batch_iterator)

        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
        lr = adjust_learning_rate(optimizer, epoch, args.learning_rate, args.lr_schedule)

        images = Variable(images, requires_grad=False).to(device)
        labels = Variable(labels, requires_grad=False).to(device)

        logits = lprnet(images)
        log_probs = logits.permute(2, 0, 1) 
        log_probs = log_probs.log_softmax(2).requires_grad_()
        
        optimizer.zero_grad()
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        
        if loss.item() == np.inf:
            continue
            
        loss.backward()
        optimizer.step()
        loss_val += loss.item()
        end_time = time.time()
        
        if iteration % 20 == 0:
            print('Epoch:' + repr(epoch) + ' || iter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + ' || Loss: %.4f ||' % (loss.item()) + ' Time: %.4f sec ||' % (end_time - start_time) + ' LR: %.6f' % (lr))
            
            # 记录 Loss
            history['loss'].append(loss.item())

    # === 训练结束，保存模型与画图 ===
    print("Final test Accuracy:")
    Greedy_Decode_Eval(lprnet, test_dataset, args, device)
    torch.save(lprnet.state_dict(), args.save_folder + 'Final_LPRNet_model.pth')
    
    # 绘制 Loss 曲线
    plt.figure()
    plt.plot(history['loss'])
    plt.title('LPRNet Training Loss')
    plt.xlabel('Steps (x20)')
    plt.ylabel('CTC Loss')
    plt.savefig(os.path.join(args.result_folder, 'lprnet_loss_visualize.png'))
    
    # 绘制 Acc 曲线
    if len(history['acc']) > 0:
        plt.figure()
        plt.plot(history['iter'], history['acc'])
        plt.title('LPRNet Validation Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join(args.result_folder, 'lprnet_acc_visualize.png'))
    
    print("Training finished. Results saved.")

def Greedy_Decode_Eval(Net, datasets, args, device):
    Net = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
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
    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))
    return Acc

if __name__ == "__main__":
    train()