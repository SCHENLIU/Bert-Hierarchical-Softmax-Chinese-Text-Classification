# Bert-Hierarchical-Softmax-Chinese-Text-Classification
bert+hierarchical softmax

## pretrained model
Three files are needed in /bert_pretrain:
 - pytorch_model.bin  
 - bert_config.json  
 - vocab.txt  

All these files you can download here:
bert_Chinese: model: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz  
              vocabulary: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt  
reference from(https://github.com/huggingface/pytorch-transformers)   


A backup link for pretrained model：https://pan.baidu.com/s/1qSAD5gwClq7xlgzl_4W3Pw

From the site you can get a zip file, unzip it and put it in /bert_pretrain.

Hierarchical softmax introduction：https://talbaumel.github.io/blog/softmax/

## Data
Some samples are loaded:

```
./data/train.txt
```

where `train.txt` is a text file containing: title, content, label_level_1, label_level_2.

By default, the seperatotr is set to `\t`.


## How to use
Just need to download pretrained files above.
```
# train：
# bert
python train_evl.py
```







## Reference
[1] Hierarchical softmax：https://talbaumel.github.io/blog/softmax/

[2] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

[3] https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch


