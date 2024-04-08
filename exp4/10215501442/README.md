# README

----------------------------

## 环境搭建

-----------------------------

使用命令 `conda create --name test --file requirements.txt`

----------------------

## argparse

---------------------------

使用命令如下

```bash
 python .\main.py --model rnn_rnn --lr 0.001 --epochs 1 
```

各参数可选项

model：



rnn_rnn

LSTM_LSTM

GRU_GRU

transf_transf

T5_T5

-------------------------------------------------------

lr:

建议0.001

---------------------

epochs：

建议1或3，T5的训练很花时间。
