# Bert-Hierarchical-Softmax-Chinese-Text-Classification
bert+hierarchical softmax

## pretrained model
There are three files in 
bert模型放在 bert_pretain目录下，每个目录下都是三个文件：
 - pytorch_model.bin  
 - bert_config.json  
 - vocab.txt  

预训练模型下载地址：  
bert_Chinese: 模型 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz  
              词表 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt  
来自[这里](https://github.com/huggingface/pytorch-transformers)   
备用：模型的网盘地址：https://pan.baidu.com/s/1qSAD5gwClq7xlgzl_4W3Pw

解压后，按照上面说的放在对应目录下，文件名称确认无误即可。  

层级分类的softmax参考：https://talbaumel.github.io/blog/softmax/

## 使用说明
下载好预训练模型就可以跑了。
```
# 训练并测试：
# bert
python train_evl.py








## Reference
[1] 层级分类softmax：https://talbaumel.github.io/blog/softmax/

[2] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

[3] https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch


