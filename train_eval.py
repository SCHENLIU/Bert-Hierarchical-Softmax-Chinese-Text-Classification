# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from pytorch_pretrained.optimization import BertAdam
from utils import build_dataset, build_iterator, get_time_dif
from models.bert import Config,HierarchicalModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


class process(object):

    def __init__(self, config, model, train_iter, dev_iter, test_iter):

        self.train(config, model, train_iter, dev_iter, test_iter)

    def _get_loss(self, predicts, label):
        path, _ = config.value_to_path_and_nodes_dict[int(label.data[1])]
        criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available:
            criterion = criterion.cuda()

        def f(predict, p):
            p = torch.LongTensor([p])
            # convert to cuda tensors if cuda flag is true
            # if torch.cuda.is_available:
            #     p = p.cuda()
            # p = Variable(p)
            return criterion(predict.unsqueeze(0), p)

        loss = list(map(f, predicts, path))
        return torch.sum(torch.stack(loss))


    def train(self, config, model, train_iter, dev_iter, test_iter):
        start_time = time.time()
        model.train()
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=config.learning_rate,
                             warmup=0.05,
                             t_total=len(train_iter) * config.num_epochs)
        total_batch = 0  # 记录进行到多少batch
        dev_best_loss = float('inf')
        last_improve = 0  # 记录上次验证集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升
        model.train()
        for epoch in range(config.num_epochs):
            print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
            for i, (trains, labels) in enumerate(train_iter):
                predicts, predicts_list = model(trains, labels, mode="train")
                losses = list(map(self._get_loss, predicts, labels))
                loss = torch.mean(torch.stack(losses))
                model.zero_grad()

                #loss = F.cross_entropy(losses, labels)
                loss.backward()
                optimizer.step()
                if total_batch % 1000 == 0:
                    dev_acc_lev1, dev_acc_lev2, dev_loss = self.evaluate(config, model, dev_iter)
                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        torch.save(model.state_dict(), config.save_path)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Val Loss: {2:>5.2},  Val_lev1 Acc: {3:>6.2%},  Val_lev2 Acc: {4:>6.2%},  Time: {5} {6}'
                    print(msg.format(total_batch, loss.item(), dev_loss, dev_acc_lev1, dev_acc_lev2, time_dif, improve))
                    model.train()
                    if total_batch > 10 and total_batch % 90000 == 0:
                        self.test(config, model, train_iter)
                total_batch += 1
                if total_batch - last_improve > config.require_improvement:
                    # 验证集loss超过1000batch没下降，结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
            if flag:
                break
        self.test(config, model, test_iter)


    def test(self, config, model, test_iter):
        # test
        model.load_state_dict(torch.load(config.save_path))
        model.eval()
        start_time = time.time()
        test_acc_lev1, test_acc_lev2, test_loss, test_report, test_confusion = self.evaluate(config, model, test_iter, test=True)
        msg = 'Test Loss: {0:>5.2},  Test Acc_lev1: {1:>6.2%},  Test Acc_lev2: {2:>6.2%}'
        print(msg.format(test_loss, test_acc_lev1, test_acc_lev2))
        print("Precision, Recall and F1-Score...")
        print(test_report)
        print("Confusion Matrix...")
        print(test_confusion)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)


    def evaluate(self,config, model, data_iter, test=False):
        model.eval()
        loss_total = 0
        predict_all_lev1 = np.array([], dtype=int)
        labels_all_lev1 = np.array([], dtype=int)
        predict_all_lev2 = np.array([], dtype=int)
        labels_all_lev2 = np.array([], dtype=int)

        with torch.no_grad():
            for texts, labels in data_iter:
                predicts, predicts_list = model(texts, labels, mode="evl")
                losses = list(map(self._get_loss, predicts, labels))
                loss = torch.mean(torch.stack(losses))

                loss_total += loss.item()

                true = labels.data.cpu()

                predicts_lev1 = [r[0].detach().cpu().numpy() for r in predicts_list]
                predic_lev1_index = [torch.max(torch.tensor(r).data, -1)[1].cpu() for r in predicts_lev1]
                print(list(zip(predicts_list,predic_lev1_index)))
                predicts_lev2 = [r[t + 1].detach().cpu().numpy() for r, t in zip(predicts_list, predic_lev1_index)]
                predic_lev2_index = [torch.max(torch.tensor(r).data, -1)[1].cpu() for r in predicts_lev2]
                predic_lev1 = [config.label_dict[config.tree[1][a][0]] for a in predic_lev1_index]
                predic_lev2 = [config.label_dict[config.tree[1][a][1][b][0]] for a, b in
                               zip(predic_lev1_index, predic_lev2_index)]

                predict_all_lev1 = np.append(predict_all_lev1, predic_lev1)
                predict_all_lev2 = np.append(predict_all_lev2, predic_lev2)
                labels_all_lev1 = np.append(labels_all_lev1, true[:, 0])
                labels_all_lev2 = np.append(labels_all_lev2, true[:, 1])

        acc_lev1 = metrics.accuracy_score(labels_all_lev1, predict_all_lev1)
        acc_lev2 = metrics.accuracy_score(labels_all_lev2, predict_all_lev2)

        if test:
            report = metrics.classification_report(labels_all_lev2, predict_all_lev2)
            confusion = metrics.confusion_matrix(labels_all_lev2, predict_all_lev2)
            return acc_lev1, acc_lev2, loss_total / len(data_iter), report, confusion
        return acc_lev1, acc_lev2, loss_total / len(data_iter)


if __name__ == '__main__':
    dataset = 'D:/sina/Bert-Chinese-Text-Classification-Pytorch-master'  # 数据集
    config = Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = HierarchicalModel(config).to(config.device)
    #model = torch.nn.DataParallel(model, device_ids=[0, 1])
    pro = process(config, model, train_iter, dev_iter, train_iter)
    pro.train()
