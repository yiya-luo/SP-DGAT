import argparse
import time
import gc
import random,os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from src.net import GNNStack 
from utlis import AverageMeter, accuracy, log_msg, get_default_train_val_test_loader1, evaluate
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch UEA Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='tf_GAT_p1')#dyGIN2d
parser.add_argument('-d', '--dataset', metavar='DATASET', default='mesag')#PPGflow3_2_5
parser.add_argument('--num_layers', type=int, default=3, help='the number of GNN layers')
parser.add_argument('--groups', type=int, default=6, help='the number of time series groups (num_graphs)')
parser.add_argument('--pool_ratio', type=float, default=0.2, help='the ratio of pooling for nodes')
parser.add_argument('--dropout', type=float, default=0.5, help='dropour')
parser.add_argument('--kern_size', type=str, default="9,5,3", help='list of time conv kernel size for each layer')
parser.add_argument('--in_dim', type=int, default=64, help='input dimensions of GNN stacks')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimensions of GNN stacks')
parser.add_argument('--out_dim', type=int, default=256, help='output dimensions of GNN stacks')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--val-batch-size', default=128, type=int, metavar='V',
                    help='validation batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-3
                    , type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--use_benchmark', dest='use_benchmark', action='store_true',
                    default=True, help='use benchmark')
parser.add_argument('--tag', default='date', type=str,
                    help='the tag for identifying the log and model files. Just a string.')


os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
device_ids = [1,2]
args = parser.parse_args()
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

save_dir = '/home/yiya/code_Project_yiya/osa_GNN/result/'
def main():
    # args = parser.parse_args()
    
    args.kern_size = [ int(l) for l in args.kern_size.split(",") ]
    os.makedirs(f'{save_dir}/adj/{args.arch}/') if not os.path.exists(f'{save_dir}/adj/{args.arch}/') else print('1')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    main_work(args)


def main_work(args):
    # init acc
    best_acc1, best_recall, best_precision, best_f1 = 0,0,0,0
    
    if args.tag == 'date':
        local_date = time.strftime('%m.%d', time.localtime(time.time()))
        args.tag = local_date

    log_file = '/home/yiya/code_Project_yiya/osa_GNN/log/{}_gpu{}_{}_{}_exp.txt'.format(args.tag, args.gpu, args.arch, args.dataset)

    # if args.gpu is not None:
    #     print("Use GPU: {} for training".format(args.gpu))   
    #     # 检查并设置可见的 GPU 设备
    #     device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

    # dataset
    train_loader, val_loader, num_nodes, seq_length, num_classes = get_default_train_val_test_loader1(args)
    
    # training model from net.py
    model = GNNStack(gnn_model_type=args.arch, num_layers=args.num_layers, dropout=args.dropout,
                     groups=args.groups, pool_ratio=args.pool_ratio, kern_size=args.kern_size, 
                     in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim, 
                     seq_len=seq_length, num_nodes=num_nodes, num_classes=num_classes, batch_size=args.batch_size)
        # print & log
    log_msg('pooling_ratio {}, dropout{}, hidden_dim {}, outdim {}'.format(args.pool_ratio, args.dropout, args.hidden_dim,args.out_dim), log_file)

    # print & log
    log_msg('epochs {}, lr {}, weight_decay {}'.format(args.epochs, args.lr, args.weight_decay), log_file)


    # determine whether GPU or not
    if not torch.cuda.is_available():
        print("Warning! Using CPU!!!")
    elif args.gpu is not None:
        # torch.cuda.set_device(args.gpu)

        # collect cache
        gc.collect()
        torch.cuda.empty_cache()

        # model = model.cuda(args.gpu)   
        model = model.to(device)
        model = nn.DataParallel(model)
        
        if args.use_benchmark:
            cudnn.benchmark = True
        print('Using cudnn.benchmark.')
    else:
        print("Error! We only have one gpu!!!")


    # define loss function(criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    # criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,momentum=0.5)
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                              patience=50, verbose=True)


    # validation
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return


    # train & valid
    print('****************************************************')
    print(args.dataset)

    dataset_time = AverageMeter('Time', ':6.3f')

    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []
    epoches = []

    end = time.time()
    for epoch in tqdm(range(args.epochs)):
        epoches += [epoch]

        # train for one epoch
        # acc_train_per, loss_train_per = train(train_loader, model, criterion, optimizer, lr_scheduler, args)
        acc_train_per, loss_train_per, eval_para = train(train_loader, model, criterion, optimizer, lr_scheduler, args)
        t_list = []
        for ti in eval_para[4]:
            if type(ti) is int:
                t_list.append(ti)
            else:
                t_list.append(list(ti))
        acc_train += [acc_train_per]
        loss_train += [loss_train_per]

       
        # evaluate on validation set
        # acc_val_per, loss_val_per = validate(val_loader, model, criterion, args)
        acc_val_per, loss_val_per, eval_val_para, adj = validate(val_loader, model, criterion, args)
        t_val_list = []
        for ti in eval_val_para[4]:
            if type(ti) is int:
                t_val_list.append(ti)
            else:
                t_val_list.append(list(ti))
        acc_val += [acc_val_per]
        loss_val += [loss_val_per]
        if eval_val_para[1]+eval_val_para[2] > 0:
            f1score = 2*eval_val_para[1]*eval_val_para[2]/(eval_val_para[1]+eval_val_para[2])
        else:
            f1score = 0 
        


        # if (acc_val_per>best_acc1) or eval_val_para[1]>best_recall or eval_val_para[2]>best_precision or f1score>best_f1:
             # msg = f'TRAIN, epoch {epoch}, loss {loss_train_per}, acc {acc_train_per}'
        msg = f'TRAIN, epoch {epoch}, loss {loss_train_per}, acc {acc_train_per},recall {eval_para[1]},precision {eval_para[2]},specificity {eval_para[3]},[tp,tn,fp,fn]: {t_list}'
        log_msg(msg, log_file)

        # msg = f'VAL, loss {loss_val_per}, acc {acc_val_per}'
        msg = f'VAL, loss {loss_val_per}, acc {acc_val_per},f1 {f1score}, recall {eval_val_para[1]},precision {eval_val_para[2]},specificity {eval_val_para[3]},[tp,tn,fp,fn]: {t_val_list}'
        log_msg(msg + '\n', log_file)

        # remember best acc
        if best_acc1!= max(acc_val_per, best_acc1):
            torch.save(model.state_dict(), f'{save_dir}/best_model_{args.arch}.pth')
        best_acc1, best_recall, best_precision, best_f1 = max(acc_val_per, best_acc1),max(eval_val_para[1], best_recall),max(eval_val_para[2], best_precision), max(f1score, best_f1)


        # 保存adj
        if (epoch+1)%10 == 0 or epoch==0:
            plt.clf()
            plt.imshow(adj.cpu().detach().numpy()[:64,:], cmap='Blues')
            plt.colorbar()            
            plt.savefig(f'{save_dir}/adj/{args.arch}/adj_{epoch}.png')
            torch.save(adj, f'{save_dir}/adj/{args.arch}/adj_matrix_{epoch}.pth')
    
    # 绘制训练损失和验证损失的折线图
    plt.clf()
    plt.plot(loss_train, 'b', label='Training loss')
    plt.plot(loss_val, 'r', label='Validation loss')

    # 添加标题和标签
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 添加图例
    plt.legend()
    plt.savefig(f'{save_dir}/loss/loss_{args.tag}_{args.arch}.png')        


    # measure elapsed time
    dataset_time.update(time.time() - end)

    # log & print the best_acc
    msg = f'\n\n * BEST_ACC: {best_acc1}\n * TIME: {dataset_time}\n'
    log_msg(msg, log_file)

    print(f' * best_acc1: {best_acc1}')
    print(f' * time: {dataset_time}')
    print('****************************************************')


    # collect cache
    gc.collect()
    torch.cuda.empty_cache()


def train(train_loader, model, criterion, optimizer, lr_scheduler, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc', ':6.2f')

    # switch to train mode
    model.train()

    for count, (data, label) in enumerate(train_loader):

        # data in cuda
        # data = data.cuda(args.gpu).type(torch.float)
        # label = label.cuda(args.gpu).type(torch.long)
        data = data.to(device).type(torch.float)
        label = label.to(device).type(torch.long)

        # compute output
        output,adj = model(data)
    
        loss = criterion(output, label)

        output, acc, [tp, tn, fp, fn] = accuracy(output, label, topk=(1, 1))
        # loss = criterion(output.view(-1, 1).float(), label.view(-1, 1).float())

        losses.update(loss.item(), data.size(0))
        top1.update_val(tp, tn, fp, fn, data.size(0))
        top1.update(acc, data.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    lr_scheduler.step(top1.avg)
    recall, precision, specificity = evaluate(top1.tp, top1.tn, top1.fp, top1.fn)

    return top1.avg, losses.avg, [acc, recall, precision, specificity, [top1.tp, top1.tn, top1.fp, top1.fn]]


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for count, (data, label) in enumerate(val_loader):
            # if args.gpu is not None:
            #     data = data.cuda(args.gpu, non_blocking=True).type(torch.float)
            # if torch.cuda.is_available():
            #     label = label.cuda(args.gpu, non_blocking=True).type(torch.long)

            data = data.to(device).type(torch.float)
            label = label.to(device).type(torch.long)
            # compute output
            output,adj = model(data)

            loss = criterion(output, label)

            output, acc, [tp, tn, fp, fn] = accuracy(output, label, topk=(1, 1))
            # loss = criterion(output.view(-1, 1).float(), label.view(-1, 1).float())
            losses.update(loss.item(), data.size(0))
            top1.update_val(tp, tn, fp, fn, data.size(0))
            top1.update(acc, data.size(0))

        recall, precision, specificity = evaluate(top1.tp, top1.tn, top1.fp, top1.fn)
        return top1.avg, losses.avg, [acc, recall, precision, specificity, [top1.tp, top1.tn, top1.fp, top1.fn]], adj
        # return top1.avg, losses.avg


if __name__ == '__main__':
    main()
