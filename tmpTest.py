import torchvision
import torch.utils.data as data
import numpy as np
from sklearn.model_selection import train_test_split
import os, sys
import torch
import torch.nn as nn
from torch import tensor
import time
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import argparse


class MyDataSet(data.Dataset):
    def __init__(self, path, label):
        super(MyDataSet, self).__init__()
        self.x, self.y = generate_datapath(path, label)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        filepath, label = self.x[index].strip(), self.y[index]
        img_array = np.load(filepath)
        imObj = Image.fromarray(img_array).resize((224, 224))
        img_array = np.array(imObj).swapaxes(0, 2) / 255.
        return img_array.astype('float32'), label


def generate_datapath(path_li, path_label):
    filepath = []
    img_label = []
    for i, root in enumerate(path_li):
        label = path_label[i]
        length = 0
        for rootname, _, filename in os.walk(root):
            for s in filename:
                filepath.append(rootname + '/' + s)
                length += 1
        img_label += [label] * length
    return filepath, img_label


def get_img_array(path, _range=None):
    if _range is None:
        length = len(path)
    else:
        length = _range

    img_set = []
    for i in range(length):
        filepath = path[i]
        img = np.load(filepath)
        imObj = Image.fromarray(img).resize((224, 224))
        img = np.array(imObj).swapaxes(0, 2) / 255.
        img_set.append(img.astype('float32'))

    return tensor(img_set)


def get_predict_label(output):
    return torch.max(output, 1)[1].data.numpy().squeeze()


def scoring_acc(target, output):
    s = 0
    for i in range(len(target)):
        if int(target[i]) == output[i]:
            s += 1
    return s / len(target)


def confusion_matrix(target, output):
    cm = np.zeros((2, 2), dtype=np.int)
    for i in range(len(target)):
        if target[i] == 1 and output[i] == 1:  # TP
            cm[0, 0] += 1
        elif target[i] == 0 and output[i] == 1:  # FP
            cm[0, 1] += 1
        elif target[i] == 1 and output[i] == 0:  # FN
            cm[1, 0] += 1
        else:  # TN
            cm[1, 1] += 1
    return cm


def sensitivity(cm_arr):
    return cm_arr[0, 0] / (cm_arr[0, 0] + cm_arr[1, 0])


def specificity(cm_arr):
    return cm_arr[1, 1] / (cm_arr[1, 1] + cm_arr[0, 1])


def get_writer_dir(now_time):
    dir_name = '-'.join(now_time.split(':'))
    dir_name = '_'.join(dir_name.split(' ')[1:])
    return dir_name


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epoch", default=80, type=int, help="Number of the training epochs")
ap.add_argument("-w", "--warmup", default=5, type=int, help="Number of the warmup epochs")
ap.add_argument("-l", "--lr", default=0.001, type=float, help="Number of the training epochs")
ap.add_argument("-bs", "--batch", default=512, type=int, help="Number of the training epochs")
ap.add_argument("-p", "--pretrained", type=bool, default=False, help="Use ImageNet pre-train image.")
ap.add_argument("-c", "--comment", default="None", help="Execute comment")
args = vars(ap.parse_args())

EPOCH = args['epoch']
WARMUP = args['warmup']
LR = args['lr']
BATCH_SIZE = args['batch']

torch.manual_seed(2020)

if __name__ == '__main__':
    now_time = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())  # Sat Mar 28 22:24:24 2016
    print('---------- Start at %s ---------' % now_time)
    print('Execute: %s' % (sys.argv[0]))
    root = 'TestRuns'
    save_dir = os.path.join(root, get_writer_dir(now_time))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    writer = SummaryWriter(save_dir)
    print('Result save in %s' % save_dir)

    neg_len = None
    with open('dataPathFiles/cut_neg.txt', 'r', encoding='utf8') as f:
        neg_path = f.read().strip().split('\n')
        while '' in neg_path:
            neg_path.remove('')

        # 正例尝试部分数据
        # neg_len = len(neg_path)

        neg_label = [0] * len(neg_path)

    with open('dataPathFiles/cut_pos.txt', 'r', encoding='utf8') as f:
        pos_path = f.read().strip().split('\n')
        while '' in pos_path:
            pos_path.remove('')

        # 先尝试一部分数据
        # pos_path = pos_path_all[:neg_len]
        # other_path = pos_path_all[neg_len:]

        pos_label = [1] * len(pos_path)
        # other_label = [1] * len(other_path)
    total_path = pos_path + neg_path
    total_label = pos_label + neg_label
    # del pos_path
    # del neg_path

    path_train_va, path_test, label_train_va, label_test = train_test_split(total_path, total_label,
                                                                            test_size=0.3,
                                                                            random_state=42,
                                                                            stratify=total_label,
                                                                            shuffle=True)

    # 将剩下所有未使用的数据放入测试集
    # path_test = path_test + other_path
    # label_test = label_test + other_label

    # 验证集训练集
    path_train, path_dev, label_train, label_dev = train_test_split(path_train_va, label_train_va,
                                                                    test_size=0.1,
                                                                    random_state=38,
                                                                    stratify=label_train_va,
                                                                    shuffle=True)

    path_test, label_test = generate_datapath(path_test, label_test)
    path_dev, label_dev = generate_datapath(path_dev, label_dev)

    train_data = MyDataSet(path_train, label_train)
    # train_data = MyDataSet(path_train_va, label_train_va)
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    print('Data loaded.')

    # if use ImageNet pre-train
    if args['pretrained']:
        resnet18 = torchvision.models.resnet18(pretrained=args['pretrained'])
        del resnet18.fc
        resnet18.fc = nn.Linear(512, 2)
    else:
        resnet18 = torchvision.models.resnet18(num_classes=2)
    # resnet18.load_state_dict(torch.load('models/test_model.pth'))
    # print('Pre-trained loaded.')
    optimizer = torch.optim.Adam(resnet18.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
    loss_func = nn.CrossEntropyLoss()

    resnet18.to('cuda')

    Eva = 0
    n_iter = 0
    for epoch in range(EPOCH + WARMUP):
        # train
        resnet18.train()
        for i, (x, y) in enumerate(train_loader):
            y = y.long().to('cuda')
            x = x.to('cuda')
            output = resnet18(x)
            loss = loss_func(output, y)
            print('Epoch %d Iteration %d Loss: %s' % (epoch + 1, i + 1, loss))
            writer.add_scalar('train loss', float(loss.data), global_step=n_iter)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_iter += 1
        # evaluate
        # resnet18.eval()
        # with torch.no_grad():
        #     va_out = resnet18(get_img_array(path_dev))
        #     pred_va = get_predict_label(va_out)
        #     acc_va = scoring_acc(label_dev, pred_va)
        #     writer.add_scalar('val acc', float(acc_va), global_step=n_iter)
        #     val_loss = loss_func(va_out, tensor(label_dev).long())
        #     writer.add_scalar('validate loss', float(val_loss.data.numpy()), global_step=n_iter)
        #     Eva = (Eva * epoch + float(val_loss.data.numpy())) / (epoch + 1)
        #     writer.add_scalar('avg val loss', float(Eva), global_step=n_iter)

        # print('Epoch %s now loss %s' % (epoch, val_loss))

        if epoch >= WARMUP:
            scheduler.step()
    print('Training complete.')
    print('Train for %s epochs' % EPOCH)

    print('Saving model ... at %s' % save_dir)
    state_dict = {'model': resnet18.state_dict(), 'optimizer': optimizer.state_dict(), 'Epoch': EPOCH}
    model_save_pth = os.path.join(save_dir, 'checkpoint.dic')
    torch.save(state_dict, model_save_pth)
    print('Model is saved.')

    resnet18.eval()
    resnet18.to('cpu')
    with torch.no_grad():
        test_out = resnet18(get_img_array(path_test))
        test_pred = get_predict_label(test_out)
        acc = scoring_acc(label_test, test_pred)

    print('Acc on all test:', acc)
    conf_matrix = confusion_matrix(label_test, test_pred)
    sen = sensitivity(conf_matrix)
    spe = specificity(conf_matrix)
    print('灵敏度', sen)
    print('特异度', spe)
    print('混淆矩阵', conf_matrix)
