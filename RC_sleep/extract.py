import torch
import torch.nn as nn
import torch.nn.functional as F

class Extractor(nn.Module):

    def __init__(self, ch=1):
        
        super(Extractor, self).__init__()
        
        # Do I need the bias?
        # How does kernel size and stride affect results?
        # How does activation affect results?
        # Should we try multiple conv layers for different timescales?
        
        # linear kernels, nonlinear activation, linear readout
        # self.conv1 = nn.Conv1d(in_channels=7,out_channels=64,kernel_size=100,stride=20,bias=False)
        # self.act = nn.Tanh()
        # ?? self.conv2 = nn.Conv1d(in_channels=64,out_channels=32,kernel_size=400,stride=100,bias=False)
        # linear kernels, nonlinear recurrent vector, linear readout
        self.conv1 = nn.Conv1d(in_channels=7,out_channels=64,kernel_size=100,stride=20,bias=False)
        self.act = nn.Tanh()

        # linear kernels, nonlinear reservoir, linear readout

    def forward(self, x):
        
        res = torch.zeros(64,146).cuda() # (3000-100)/20+1 
        out = torch.zeros(25,64,146).cuda()
        for idx, win in enumerate(x):
            lin = self.conv1(win)
            res = self.act(lin + res)
            out[idx] = res
        return out

