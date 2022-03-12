
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from DotProductAttention import DotProductAttention


class AttentionGRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, sequence_length,classes, **kwargs ):
        super(AttentionGRU,self).__init__(**kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.dropout = dropout
        self.classes = classes
        # -> x needs to be: (batch_size, seq, input_size) #rnn = nn.GRU(10, 20, 2)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.attention = DotProductAttention(dropout)
        self.fc = nn.Linear(hidden_size * sequence_length, classes

    #
    def forward(self,X):

        h0 = torch.randn(2, 5, 20) #h0 = torch.randn(2, 3, 20)
        output, hn = self.gru(X, h0)
        output = self.attention(output, output, output)
        output = output.view(5, -1)
        output = self.fc(output)

        return output



    #parse
    # '''
    # python argparse 子命令
    # argparse模块可以让人轻松编写用户友好的命令行接口
    # tutorials:
    # 1.https://www.jianshu.com/p/27ce67dab97e
    # 2.https://docs.python.org/zh-cn/3/library/argparse.html
    # '''
if __name__ == '__main__':
    
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-input_size', type=int, help='input_size value')
    # parser.add_argument('-hidden_size', type=int, help='hidden_size value')
    # parser.add_argument('-num_layers', type=int, help='num_layers value')
    # parser.add_argument('-dropout', type=float, help='num_layers value')
    # args = parser.parse_args()
    # print(args)
    # %run argparseAttentionGRU.py -input_size 10 -hidden_size 20 -num_layers 2 -dropout 0.5
    # attention = AttentionGRU(input_size = args.input_size, hidden_size = args.hidden_size, num_layers = args.num_layers, dropout = args.dropout)
    # input_size = 10
    # hidden_size = 20
    # num_layers =2
    # dropout = 0.5

    model = AttentionGRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, sequence_length = sequence_length, classes = classes)
    input = torch.randn(5, 3, 10)
    output = attention(input)
    print(output.shape)
    print('/n', output)


#writer
'''
SummaryWriter 
1. https://pytorch.org/docs/stable/tensorboard.html
TensorBoard
https://blog.csdn.net/Jinlong_Xu/article/details/71124589
https://ymiir.top/index.php/2022/02/02/tensorboardx001/

'''
# writer = SummaryWriter('C:/Users/faustineljc')
# 
# x = range(100)
# for i in x:
#     writer.add_scalar('y=2x', i * 2, i)
# writer.close()