import sys
import os

from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model import *
import time
import random
import math
import numpy as np
import torch



if __name__ == '__main__':
    model=LFAConvLSTM().cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)

    batch_size=24
    input_length=6
    output_length=6
    height=32
    width=32
    channle=2#inflow and outflow

    input=torch.randn((batch_size,input_length,height,width,channle)).cuda()
    target = torch.randn((batch_size, output_length, height, width, channle)).cuda()
    optimizer.zero_grad()
    output = model(input.float())
    criterion = torch.nn.MSELoss()
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(loss)


