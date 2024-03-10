import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init
from utils.aug import sim_global, aug_topology, aug_traffic, aug_topology_


class STEncoder(nn.Module):
    def __init__(self, Kt, Ks, blocks, input_length, num_nodes, droprate=0.1):
        super(STEncoder, self).__init__()
        self.Ks=Ks
        c = blocks[0]
        self.tconv11 = TemporalConvLayer(Kt, c[0], c[1], "GLU")
        self.pooler = Pooler(input_length - (Kt - 1), c[1])
        
        self.sconv12 = SpatioConvLayer(Ks, c[1], c[1])
        self.tconv13 = TemporalConvLayer(Kt, c[1], c[2])
        self.ln1 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout1 = nn.Dropout(droprate)

        c = blocks[1]
        self.tconv21 = TemporalConvLayer(Kt, c[0], c[1], "GLU")
        
        self.sconv22 = SpatioConvLayer(Ks, c[1], c[1])
        self.tconv23 = TemporalConvLayer(Kt, c[1], c[2])
        self.ln2 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout2 = nn.Dropout(droprate)
        
        self.s_sim_mx = None
        self.t_sim_mx = None
        
        out_len = input_length - 2 * (Kt - 1) * len(blocks)
        self.out_conv = TemporalConvLayer(out_len, c[2], c[2], "GLU") # out_len 大小的核，输出时间为 1
        self.ln3 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout3 = nn.Dropout(droprate)
        self.receptive_field = input_length + Kt -1

    def forward(self, x0, graph):
        lap_mx = self._cal_laplacian(graph)
        Lk = self._cheb_polynomial(lap_mx, self.Ks)
        
        in_len = x0.size(1) # x0, nlvc
        if in_len < self.receptive_field:
            x = F.pad(x0, (0,0,0,0,self.receptive_field-in_len,0))
        else:
            x = x0
        x = x.permute(0, 3, 1, 2)  # (batch_size, feature_dim, input_length, num_nodes), nclv 
        
        ## ST block 1
        x = self.tconv11(x)    # nclv
        x, x_agg, self.t_sim_mx = self.pooler(x)
        self.s_sim_mx = sim_global(x_agg, sim_type='cos')

        x = self.sconv12(x, Lk)   # nclv
        x = self.tconv13(x)  
        x = self.dropout1(self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)) # dropout 导致的结果不稳定
        
        ## ST block 2
        x = self.tconv21(x)
        x = self.sconv22(x, Lk)
        x = self.tconv23(x)
        x = self.dropout2(self.ln2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

        ## out block
        x = self.out_conv(x) # ncl(=1)v
        x = self.dropout3(self.ln3(x.permute(0, 2, 3, 1))) # nlvc
        return x # nl(=1)vc

    def _cheb_polynomial(self, laplacian, K):
        N = laplacian.size(0)  
        multi_order_laplacian = torch.zeros([K, N, N], device=laplacian.device, dtype=torch.float) 
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k-2]

        return multi_order_laplacian

    def _cal_laplacian(self, graph):
        I = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
        # graph = graph + I # add self-loop to prevent zero in D
        D = torch.diag(torch.sum(graph, dim=-1) ** (-0.5))  # 检查点：
        L = I - torch.mm(torch.mm(D, graph), D)
        return L

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)  # filter=(1,1), similar to fc

    def forward(self, x):  # x: (n,c,l,v)
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x  

class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        x_in = self.align(x)[:, :, self.kt - 1:, :]  # 因为 kernel 的大小是 self.kt，导致减小了时间的维度
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)  
        return torch.relu(self.conv(x) + x_in)  

class SpatioConvLayer(nn.Module):
    def __init__(self, ks, c_in, c_out):
        super(SpatioConvLayer, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks)) # kernel: C_in*C_out*ks
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1))
        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta) # 用来得到线性层的输入和输出
        bound = 1 / math.sqrt(fan_in) # 根据输入的维度进行 uniform 初始化
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)  # Lk 不是对称的
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b 
        x_in = self.align(x) 
        return torch.relu(x_gc + x_in)

class Pooler(nn.Module):
    def __init__(self, n_query, d_model, agg='avg'):
        super(Pooler, self).__init__()

        ## attention matirx
        self.att = FCLayer(d_model, n_query) 
        self.align = Align(d_model, d_model)
        self.softmax = nn.Softmax(dim=2) # softmax on the seq_length dim, nclv

        self.d_model = d_model
        self.n_query = n_query 
        if agg == 'avg':
            self.agg = nn.AvgPool2d(kernel_size=(n_query, 1), stride=1)
        elif agg == 'max':
            self.agg = nn.MaxPool2d(kernel_size=(n_query, 1), stride=1)
        else:
            raise ValueError('Pooler supports [avg, max]')
        
    def forward(self, x):
        x_in = self.align(x)[:, :, -self.n_query:, :] # ncqv
        # calculate the attention matrix A using key x   
        A = self.att(x) # x: nclv, A: nqlv  相当于是直接把特征 c 变成时间 q
        A = F.softmax(A, dim=2) # nqlv

        # calculate region embeding using attention matrix A
        x = torch.einsum('nclv,nqlv->ncqv', x, A)
        x_agg = self.agg(x).squeeze(2) # ncqv->ncv
        x_agg = torch.einsum('ncv->nvc', x_agg) # ncv->nvc

        # calculate the temporal simlarity (prob)
        A = torch.einsum('nqlv->lnqv', A)
        A = self.softmax(self.agg(A).squeeze(2)) # A: lnqv->lnv
        return torch.relu(x + x_in), x_agg.detach(), A.detach()


class FCLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(FCLayer, self).__init__()
        self.linear = nn.Conv2d(c_in, c_out, 1)  

    def forward(self, x):
        return self.linear(x)

class SpatialHeteroModel(nn.Module):
    def __init__(self, c_in, nmb_prototype, batch_size, tau=0.5):
        super(SpatialHeteroModel, self).__init__()
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.prototypes = nn.Linear(c_in, nmb_prototype, bias=False)
        
        self.tau = tau
        self.d_model = c_in
        self.batch_size = batch_size

        for m in self.modules():
            self.weights_init(m)
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, z1, z2):
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = self.l2norm(w)  # 线性层进行 l2 归一化
            self.prototypes.weight.copy_(w)
        
        # l2norm avoids nan of Q in sinkhorn
        zc1 = self.prototypes(self.l2norm(z1.reshape(-1, self.d_model))) # nd -> nk, assignment q, embedding z
        zc2 = self.prototypes(self.l2norm(z2.reshape(-1, self.d_model))) # nd -> nk
        with torch.no_grad():
            q1 = sinkhorn(zc1.detach())
            q2 = sinkhorn(zc2.detach())  # 计算 sinkhorn 熵
        l1 = - torch.mean(torch.sum(q1 * F.log_softmax(zc2 / self.tau, dim=1), dim=1))
        l2 = - torch.mean(torch.sum(q2 * F.log_softmax(zc1 / self.tau, dim=1), dim=1))
        return l1 + l2
    
@torch.no_grad()
def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    sum_Q = torch.sum(Q)
    Q /= sum_Q
    
    for it in range(sinkhorn_iterations):
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B
    return Q.t()


class TemporalHeteroModel(nn.Module):
    def __init__(self, c_in, batch_size, num_nodes, device):
        super(TemporalHeteroModel, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_nodes, c_in)) # representation weights
        self.W2 = nn.Parameter(torch.FloatTensor(num_nodes, c_in)) 
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        
        self.read = AvgReadout()
        self.disc = Discriminator(c_in)
        self.b_xent = nn.BCEWithLogitsLoss()

        lbl_rl = torch.ones(batch_size, num_nodes)
        lbl_fk = torch.zeros(batch_size, num_nodes)
        self.lbl = torch.cat((lbl_rl, lbl_fk), dim=1)
        if device == 'cuda':
            self.lbl = self.lbl.cuda()
        
        self.n = batch_size

    def forward(self, z1, z2):
        h = (z1 * self.W1 + z2 * self.W2).squeeze(1) # nlvc->nvc
        s = self.read(h) # average representation of Graph. s: summary of h, nc

        # select another region in batch
        idx = torch.randperm(self.n) # 重新排序
        shuf_h = h[idx]

        logits = self.disc(s, h, shuf_h)
        loss = self.b_xent(logits, self.lbl)
        return loss

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
        self.sigm = nn.Sigmoid()

    def forward(self, h):
        s = torch.mean(h, dim=1)
        s = self.sigm(s) 
        return s

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.net = nn.Bilinear(n_h, n_h, 1) # similar to score of CPC

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, summary, h_rl, h_fk):
        s = torch.unsqueeze(summary, dim=1)
        s = s.expand_as(h_rl).contiguous()

        # score of real and fake, (batch_size, num_nodes)
        sc_rl = torch.squeeze(self.net(h_rl, s), dim=2)   # 通道变化 64 --64x64--> 64 --64x1--> 1
        sc_fk = torch.squeeze(self.net(h_fk, s), dim=2)

        logits = torch.cat((sc_rl, sc_fk), dim=1)

        return logits


class MLP(nn.Module):
    def __init__(self, c_in, c_out): 
        super(MLP, self).__init__()
        self.fc1 = FCLayer(c_in, int(c_in // 2))
        self.fc2 = FCLayer(int(c_in // 2), c_out)

    def forward(self, x):
        x = torch.tanh(self.fc1(x.permute(0, 3, 1, 2))) # nlvc->nclv
        x = self.fc2(x).permute(0, 2, 3, 1) # nclv->nlvc
        return x


class STSSL(nn.Module):
    def __init__(self, args):
        super(STSSL, self).__init__()
        # spatial temporal encoder
        self.encoder = STEncoder(Kt=3, Ks=3, blocks=[[2, int(args.d_model//2), args.d_model], [args.d_model, int(args.d_model//2), args.d_model]], 
                        input_length=args.hist_num, num_nodes=args.num_nodes, droprate=args.dropout)

        # temporal heterogenrity modeling branch
        self.mlp = MLP(args.d_model, args.d_output)
        self.thm = TemporalHeteroModel(args.d_model, args.batch_size, args.num_nodes, args.device)
        # spatial heterogenrity modeling branch
        self.shm = SpatialHeteroModel(args.d_model, args.nmb_prototype, args.batch_size, args.shm_temp)
        self.mae = masked_mae_loss(mask_value=5.0)
        self.args = args
    
    def forward(self, view1, graph):
        repr1 = self.encoder(view1, graph) # view1: n,l,v,c; graph: v,v 

        s_sim_mx = self.fetch_spatial_sim()  # 结点来做拓扑的相似性
        graph2 = aug_topology_(s_sim_mx, graph, percent=self.args.aug_drop_percent*2)
        
        t_sim_mx = self.fetch_temporal_sim()  # 时间用来做交通的相似性
        view2 = aug_traffic(t_sim_mx, view1, percent=self.args.aug_drop_percent)
        
        repr2 = self.encoder(view2, graph2.to(view1.device))
        return repr1, repr2

    def fetch_spatial_sim(self):
        return self.encoder.s_sim_mx
    
    def fetch_temporal_sim(self):
        return self.encoder.t_sim_mx

    def predict(self, z1, z2):
        return self.mlp(z1)

    def loss(self, z1, z2, y_true, scaler):
        l1 = self.pred_loss(z1, z2, y_true, scaler)
        sep_loss = [l1.item()]
        
        l2 = self.temporal_loss(z1, z2)
        sep_loss.append(l2.item())
        
        l3 = self.spatial_loss(z1, z2)
        sep_loss.append(l3.item())

        loss = l1 + l2 + l3
        return loss, sep_loss

    def pred_loss(self, z1, z2, y_true, scaler):
        y_pred = scaler.inverse_transform(self.predict(z1, z2))  # 只用在 z1 进行预测
        y_true = scaler.inverse_transform(y_true)
 
        y_pred = y_pred.squeeze(1)
        loss = self.args.yita * self.mae(y_pred[..., 0], y_true[..., 0]) + \
                (1 - self.args.yita) * self.mae(y_pred[..., 1], y_true[..., 1])
        return loss
    
    def temporal_loss(self, z1, z2):
        return self.thm(z1, z2)

    def spatial_loss(self, z1, z2):
        return self.shm(z1, z2)
    



