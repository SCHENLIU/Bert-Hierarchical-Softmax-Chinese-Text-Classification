# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer
import os
from torch.autograd import Variable
from TreeTools import TreeTools


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train_Merged1125.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev_Merged1125.txt'                                    # 验证集
        self.test_path = dataset + '/data/test_Merged1125.txt'                                  # 测试集
        #self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        #self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 1                                             # epoch数
        self.batch_size = 8                                           # mini-batch大小
        self.pad_size = 400                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = dataset + '/bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.tree =\
            ["root", [
                        [
                            "财经", [
                                        ["期货", None], ["沪深股票", None], ["银行", None], ["保险", None], ["信托", None], ["互联网金融", None], ["基金", None], ["外汇", None], ["贵金属", None], ["港股", None], ["债券", None], ["美股", None], ["行业新闻", None]
                                    ]
                         ],
                        [
                            "汽车", [
                                        ["二手车", None], ["改装/赛事", None], ["试驾评测", None], ["汽车导购", None], ["汽车金融", None], ["新车", None], ["行情报价", None], ["新能源汽车", None], ["汽车文化", None], ["汽车技术", None], ["用车养车", None], ["学车", None], ["汽车服务", None], ["花边", None], ["车展", None], ["人车生活", None]
                                    ]
                         ]
                    ]
            ]


        self.label_dict = {
            "期货": 1,
            "沪深股票": 2,
            "银行": 3,
            "保险": 4,
            "信托": 5,
            "互联网金融": 6,
            "基金": 7,
            "外汇": 8,
            "贵金属": 9,
            "港股": 10,
            "债券": 11,
            "美股": 12,
            "行业新闻": 13,
            "二手车": 14,
            "改装/赛事": 15,
            "试驾评测": 16,
            "汽车导购": 17,
            "汽车金融": 18,
            "新车": 19,
            "行情报价": 20,
            "新能源汽车": 21,
            "汽车文化": 22,
            "汽车技术": 23,
            "用车养车": 24,
            "学车": 25,
            "汽车服务": 26,
            "花边": 27,
            "车展": 28,
            "人车生活": 29,
            "财经":30,
            "汽车":31
        }


class HierarchicalModel(nn.Module):

    def __init__(self, config):
        super(HierarchicalModel, self).__init__()
        self._tree_tools = TreeTools()
        self.tree = config.tree
        self.count_nodes = self._tree_tools.count_nodes(self.tree)
        self.batch_size = config.batch_size
        # create a weight matrix and bias vector for each node in the tree
        self.fc = nn.ModuleList([nn.Linear(config.hidden_size, len(subtree[1])) for subtree in
                                 self._tree_tools.get_subtrees(self.tree)])

        self.value_to_path_and_nodes_dict = {}
        for path, value in self._tree_tools.get_paths(self.tree):
            nodes = self._tree_tools.get_nodes(self.tree, path)
            self.value_to_path_and_nodes_dict[self._tree_tools.label_dict[value]] = path, nodes

        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        #self.features = nn.Sequential(self.bert)

    def forward(self, x, label, mode = None):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        predicts = list(map(self._get_predicts, pooled, label, [mode]*self.batch_size))
        losses = list(map(self._get_loss, predicts, label))
        return losses, predicts

    def _get_loss(self, predicts, label):
        path, _ = self.value_to_path_and_nodes_dict[int(label.data[1])]
        criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available:
            criterion = criterion.cuda()

        def f(predict, p):
            p = torch.LongTensor([p])
            # convert to cuda tensors if cuda flag is true
            if torch.cuda.is_available:
            	p = p.cuda()
            p = Variable(p)
            return criterion(predict.unsqueeze(0), p)

        loss = list(map(f, predicts, path))
        return torch.sum(torch.stack(loss))

    def _get_predicts(self, feature, label, mode):
        #label_value = label.cpu().numpy()
        #_, nodes = self.value_to_path_and_nodes_dict[label_value[1]]
        _, nodes = self.value_to_path_and_nodes_dict[int(label.data[1])]
        if mode == "evl":
            predicts = list(map(lambda n: self.fc[n](feature), range(self.count_nodes)))
        else:
            predicts = list(map(lambda n: self.fc[n](feature), nodes))
        return predicts
