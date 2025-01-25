import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def max_min_zscore(data):
    min_values, _ = data.min(dim=1, keepdim=True)  # 计算每行的最小值
    max_values, _ = data.max(dim=1, keepdim=True)  # 计算每行的最大值
    # 进行最大最小归一化
    normalized_data = (data - min_values) / (max_values - min_values)
    return normalized_data



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

        self.recall = 0
        self.specifity = 0
        self.pre = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def update_val(self, tp, tn, fp, fn, n=1):
        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # pred = output.view(-1,1)
        correct = pred.eq(target.view(1, -1).expand_as(pred))   # 

        # res = []
        # for k in topk:
        #     correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        #     res.append(correct_k.mul_(100.0 / batch_size))
        #     return res
        # 计算准确率
        acc = (correct == True).sum().item() / correct.size(1)

        class_numm = 2
        if class_numm == 2:
            # 计算真正例、真负例、假正例和假负例
            tp = ((pred == 1) & (target == 1)).sum().item()
            tn = ((pred == 0) & (target == 0)).sum().item()
            fp = ((pred == 1) & (target == 0)).sum().item()
            fn = ((pred == 0) & (target == 1)).sum().item()

            return pred, acc, [tp, tn, fp, fn]

        else:
            # 计算混淆矩阵
            confusion_matrix = torch.zeros(class_numm, class_numm).cuda()
            for t, p in zip(target, pred):
                confusion_matrix[t, p] += 1

            # 计算真正例、真负例、假正例和假负例
            tp = confusion_matrix.diag()
            tn = confusion_matrix.sum() - (confusion_matrix.sum(dim=0) - tp)
            fp = confusion_matrix.sum(dim=1) - tp
            fn = confusion_matrix.sum(dim=0) - tp
       
        return pred, acc, [tp.cpu().numpy(),
                    tn.cpu().numpy(),
                    fp.cpu().numpy(),
                    fn.cpu().numpy()]


def evaluate(tp, tn, fp, fn):
    recall=tp/(tp+fn)
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0 
    specificity=tn / (tn + fp)
    return recall, precision, specificity


def log_msg(message, log_file):
    with open(log_file, 'a') as f:
        print(message, file=f)


def get_default_train_val_test_loader(args):
    # ppg_fe = torch.load(
    #     '/data/luojingliu/gnn_result/cluster/result/cluster10_20_nobaseline_A_result/file/dedcode_79.pt')[:20*1000]
    # baseline_fe = torch.load(
    #     '/data/luojingliu/gnn_result/cluster/result/cluster10_20_baseline_A_result/file/endcode_76.pt')[:20*1000]

    data_dir = '/data/luojingliu/gnn_result/' + 'baseline_60'
    ppg_fe = torch.load(f'{data_dir}/20W/Baseline.pt').unsqueeze(1)
    baseline_fe = torch.load(f'{data_dir}/20W/ppg_nonBaseline.pt').unsqueeze(1)


    data = torch.stack((ppg_fe,baseline_fe), dim=2)
    labels = torch.load('/data/mesa/result/train/y.pt')[:len(ppg_fe)]
    data = data.to(torch.device('cpu'))
    labels = labels.to(torch.device('cpu'))



    # init [num_variables, seq_length, num_classes]
    num_nodes = data.size(-2)

    seq_length = data.size(-1)

    num_classes = len(torch.bincount(labels.type(torch.int)))

    test_size = 0.3
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size,
                                                                        random_state=42)

    train_dataset = TensorDataset(data_train, labels_train)
    val_dataset = TensorDataset(data_test, labels_test)

    # data_loader
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.workers,
                                                    shuffle=True,
                                                    pin_memory=True)

    test_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=True,
                                                   pin_memory=True)


    return train_data_loader, test_data_loader, num_nodes, seq_length, num_classes


def get_default_train_val_test_loader1(args):
    # mesa_data_dir = '/data/mesa/result/'
    mesa_data_dir = '/data2/yiya/mesa/result/data1/dataset/mesa/'
    if 'flow' in args.dataset :
        load_dir = '/data2/yiya/mesa/result/ppg_spo2_rp1'
        data_train=torch.load(f'{load_dir}/train2.pt').unsqueeze(1)
        labels_train=torch.load(f'{load_dir}/trainy2.pt')
        data_test=torch.load(f'{load_dir}/test2.pt').unsqueeze(1)
        labels_test=torch.load(f'{load_dir}/testy2.pt')
    else:
        ppg = torch.load(f'{mesa_data_dir}/train/ppg.pt')
        spo2 = torch.load(f'{mesa_data_dir}/train/spo2.pt')
        data_train = torch.stack((ppg,spo2), dim=1) #.unsqueeze(1)
        labels_train = torch.load(f'{mesa_data_dir}/train/y.pt')
        
        ppg_test = torch.load(f'{mesa_data_dir}/test/ppg.pt')
        spo2_test = torch.load(f'{mesa_data_dir}/test/spo2.pt')
        data_test = torch.stack((ppg_test, spo2_test), dim=1) #.unsqueeze(1)
        labels_test = torch.load(f'{mesa_data_dir}/test/y.pt')

    # init [num_variables, seq_length, num_classes]
    num_nodes = data_train.size(-2)
    seq_length = data_train.size(-1)
    num_classes = len(torch.bincount(labels_train.type(torch.int)))

    train_dataset = TensorDataset(data_train, labels_train)
    val_dataset = TensorDataset(data_test, labels_test)

    # data_loader
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.workers,
                                                    shuffle=False,
                                                    pin_memory=True)

    test_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=False,
                                                   pin_memory=True)


    return train_data_loader, test_data_loader, num_nodes, seq_length, num_classes
