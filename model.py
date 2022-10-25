import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class LFA(nn.Module):
    def __init__(self, hidden_channel):
        super(LFA, self).__init__()
        self.proj_hq = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=1, stride=1,bias=False)
        self.proj_mk = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=1, stride=1,bias=False)
        self.proj_mv = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=1, stride=1,bias=False)

        self.kernel_size=7
        self.pad=3

        self.dis=self.init_distance()

    def init_distance(self):
        dis=torch.zeros(self.kernel_size,self.kernel_size).cuda()
        certer_x=int((self.kernel_size-1)/2)
        certer_y = int((self.kernel_size - 1) / 2)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                ii=i-certer_x
                jj=j-certer_y
                tmp=(self.kernel_size-1)*(self.kernel_size-1)
                tmp=(ii*ii+jj*jj)/tmp+dis[i,j]
                dis[i,j]=torch.exp(-tmp)
        dis[certer_x,certer_y]=0
        return dis


    def forward(self, H,M):
        b,c, h, w = H.shape
        pad_M=F.pad(M,[self.pad,self.pad,self.pad,self.pad])

        Q_h = self.proj_hq(H)#b,c,h,w
        K_m = self.proj_mk(pad_M)#b,c,h+2,w+2
        V_m = self.proj_mv(pad_M)#b,c,h+2,w+2

        K_m=K_m.unfold(2,self.kernel_size,1).unfold(3,self.kernel_size,1)#b,c,h,w,k,k
        V_m=V_m.unfold(2,self.kernel_size,1).unfold(3,self.kernel_size,1)#b,c,h,w,k,k

        Q_h=Q_h.permute(0,2,3,1)#b,h,w,c
        K_m=K_m.permute(0,2,3,4,5,1)#b,h,w,k,k,c
        K_m=K_m.contiguous().view(b,h,w,-1,c)#b,h,w,(k*k),c
        alpha=torch.einsum('bhwik,bhwkj->bhwij',K_m,Q_h.unsqueeze(-1))#b,h,w,(k*k),1
        dis_alpha=self.dis.view(-1,1)#(k*k),1
        alpha=alpha*dis_alpha
        alpha = F.softmax(alpha.squeeze(dim=-1), dim=-1)  #b,h,w,(k*k)
        V_m=V_m.permute(0,2,3,4,5,1).contiguous().view(b,h,w,-1,c)#b,h,w,(k*k),c
        res=torch.einsum('bhwik,bhwkj->bhwij',alpha.unsqueeze(dim=-2),V_m)#b,h,w,1,c
        res=res.permute(0,4,1,2,3).squeeze(-1)#b,c,h,w
        return res

class LFA2(nn.Module):
    def __init__(self, hidden_channel):
        super(LFA2, self).__init__()
        self.proj_hq = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=1, stride=1)
        self.proj_mk = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=1, stride=1)
        self.proj_mv = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=1, stride=1)

    def forward(self, H,M):
        b, c, h, w = H.shape
        Q_h = self.proj_hq(H)#b, c, h, w
        K_m = self.proj_mk(M)
        V_m = self.proj_mv(M)

        Q_h=Q_h.permute(0,2,3,1)#b, h, w, c
        K_m = K_m.permute(0, 2, 3, 1)  # b, h, w, c
        V_m = V_m.permute(0, 2, 3, 1)  # b, h, w, c

        alpha=torch.einsum('bhwik,bhwkj->bhwij',Q_h.unsqueeze(-2),K_m.unsqueeze(-1))#b,h,w,1,1
        alpha=torch.sigmoid(alpha)

        res=V_m*alpha.squeeze(-1)# b, h, w, c
        res=res.permute(0,3,1,2)

        return res

class LFAConvLSTM_Cell_In(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(LFAConvLSTM_Cell_In, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding)

        self.attention=LFA(self.hidden_dim)

        self.convhf = nn.Conv2d(in_channels=2*self.hidden_dim,
                              out_channels=3* self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding)


    def forward(self, x, h_cur,c_cur,f_cur,xx):

        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h = o * torch.tanh(c_next)

        zf=self.attention(h,xx)
        zf=torch.tanh(zf)
        combinedhf = torch.cat([zf, h], dim=1)
        combined_hf_conv = self.convhf(combinedhf)
        cc_ii, cc_gg, cc_oo = torch.split(combined_hf_conv, self.hidden_dim, dim=1)
        ii = torch.sigmoid(cc_ii)
        oo = torch.sigmoid(cc_oo)
        gg = torch.tanh(cc_gg)

        f_next = ii * gg + (1 - ii) * f_cur
        h_next = oo * f_next

        return h_next, c_next, f_next

class LFAConvLSTM_Cell_Out(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(LFAConvLSTM_Cell_Out, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding)

        self.attention = LFA2(self.hidden_dim)

        self.convhf = nn.Conv2d(in_channels=2 * self.hidden_dim,
                              out_channels=3 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding)

    def forward(self, x, h_cur,c_cur,f_cur,xx):
        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h= o * torch.tanh(c_next)

        zf = self.attention(h, xx)
        zf = torch.tanh(zf)
        combined_hf = torch.cat([zf, h], dim=1)  # concatenate along channel axis
        combined_hf_conv = self.convhf(combined_hf)
        cc_ii,cc_gg,cc_oo=torch.split(combined_hf_conv, self.hidden_dim, dim=1)
        ii=torch.sigmoid(cc_ii)
        oo = torch.sigmoid(cc_oo)
        gg = torch.tanh(cc_gg)

        f_next=ii*gg+(1-ii)*f_cur
        h_next=oo*f_next

        return h_next, c_next,f_next

class LFAConvLSTM(nn.Module):
    def __init__(self):
        super(LFAConvLSTM, self).__init__()
        ##layer1
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 1), stride=1, padding=0,bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.convlstm1_1 = LFAConvLSTM_Cell_In(64, 64, (7, 7))
        self.convlstm2_1 = LFAConvLSTM_Cell_Out(64, 64, (7, 7))

        ##decode
        self.deconv=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3, 3), stride=1, padding=1),
        )

    def forward(self,input,target=None):
        #input:batch*6*32*32*2
        batch=input.shape[0]

        h1_1, c1_1,f1_1, = [Variable(torch.zeros((batch,64, 32, 32))).cuda()]*3
        h2_1, c2_1,f2_1 = [Variable(torch.zeros((batch, 64, 32, 32))).cuda()] * 3

        f1=input[:,:,:,:,0]#batch*6*32*32,inflow
        f2=input[:,:,:,:,1]#batch*6*32*32,outflow

        out1=[]
        out2=[]

        for i in range(6+6-1):
            if(i<6):
                x1=f1[:,i]#batch*32*32
                x1=x1.unsqueeze(1)#batch*1*32*32
                x2 = f2[:, i]  # batch*32*32
                x2 = x2.unsqueeze(1)  # batch*1*32*32
            else:
                x1=nxt1
                x2=nxt2

            x1=self.conv(x1)
            x2 = self.conv(x2)
            h1_1,c1_1,f1_1=self.convlstm1_1(x1,h1_1,c1_1,f1_1,x2)
            h2_1, c2_1,f2_1 = self.convlstm2_1(x2, h2_1, c2_1,f2_1, x1)

            nxt1 = self.deconv(h1_1)
            nxt2 = self.deconv(h2_1)

            out1.append(nxt1)
            out2.append(nxt2)

        out1=out1[-6:]
        out1 = torch.stack(out1, dim=1)
        out2 = out2[-6:]
        out2 = torch.stack(out2, dim=1)
        output=torch.cat((out1,out2),dim=2)

        output = output.permute(0, 1, 3, 4, 2)

        return output
