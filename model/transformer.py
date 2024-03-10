
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride, padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims)
        torch.nn.init.xavier_uniform_(self.conv.weight)
        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)


    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class STEmbedding(nn.Module):
    def __init__(self, total_dim):
        super(STEmbedding, self).__init__()
        self.FC_se = FC(
            input_dims=[total_dim, total_dim], units=[total_dim, total_dim], activations=[F.relu, None])
        self.FC_te = FC(
            input_dims=[1, total_dim], units=[total_dim, total_dim], activations=[F.relu, None])
        
        self.total_dim = total_dim
        self.scaler = nn.parameter.Parameter(torch.randn(1,), requires_grad=True)
        self.SE_norm = nn.LayerNorm(total_dim)
        self.TE_norm = nn.LayerNorm(total_dim)

    def forward(self, SE, TE, T=18*12):
        # spatial embedding
        N, F = SE.shape
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.FC_se(SE)
        SE = self.SE_norm(SE)
        # temporal embedding
        TE = TE.float()
        for i in range(TE.shape[0]):
            TE[i] = TE[i]%T

        div = 1/torch.arange(1, self.total_dim+1, 2).to(self.scaler.device) 
        div_term = torch.exp(div * self.scaler)
        v1 = torch.sin(torch.einsum("btn, f->btnf", TE,div_term))
        v2 = torch.cos(torch.einsum("btn, f->btnf", TE,div_term))
        TE = torch.cat([v1, v2], -1)
        TE = self.TE_norm(TE)

        return SE + TE
    

class spatialAttention(nn.Module):
    def __init__(self, K, d):
        super(spatialAttention, self).__init__()
        total_dim = K * d
        self.d = d
        self.K = K
        self.FC_q = FC(input_dims=2 * total_dim, units=total_dim, activations=F.relu)
        self.FC_k = FC(input_dims=2 * total_dim, units=total_dim, activations=F.relu)
        self.FC_v = FC(input_dims=2 * total_dim, units=total_dim, activations=F.relu)
        self.FC = FC(input_dims=total_dim, units=total_dim, activations=F.relu)

    def forward(self, X, STE):
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # [K * batch_size, num_step, num_vertex, num_vertex]
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, total_dim]
        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)  # orginal K, change to batch_size
        X = self.FC(X)
        del query, key, value, attention
        return X


class temporalAttention(nn.Module):
    def __init__(self, K, d, mask=True):
        super(temporalAttention, self).__init__()
        total_dim = K * d
        self.d = d
        self.K = K
        self.mask = mask
        self.FC_q = FC(input_dims=2 * total_dim, units=total_dim, activations=F.relu)
        self.FC_k = FC(input_dims=2 * total_dim, units=total_dim, activations=F.relu)
        self.FC_v = FC(input_dims=2 * total_dim, units=total_dim, activations=F.relu)
        self.FC = FC(input_dims=total_dim, units=total_dim, activations=F.relu)

    def forward(self, X, STE):
        batch_size_ = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        # mask attention score
        if self.mask:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_vertex = X.shape[2]
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, -2 ** 15 + 1)
        # softmax
        attention = F.softmax(attention, dim=-1)

        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size_, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


class gatedFusion(nn.Module):
    def __init__(self, total_dim):
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=total_dim, units=total_dim, activations=None,  use_bias=False)
        self.FC_xt = FC(input_dims=total_dim, units=total_dim, activations=None,  use_bias=True)
        self.FC_h = FC(input_dims=[total_dim, total_dim], units=[total_dim, total_dim], activations=[F.relu, None])

    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.FC_h(H)
        del XS, XT, z
        return H


class STAttBlock(nn.Module):
    def __init__(self, K, d, mask=False):
        super(STAttBlock, self).__init__()
        self.spatialAttention = spatialAttention(K, d)
        self.temporalAttention = temporalAttention(K, d, mask=mask)
        self.gatedFusion = gatedFusion(K * d)

    def forward(self, X, STE):
        HS = self.spatialAttention(X, STE)
        HT = self.temporalAttention(X, STE)
        H = self.gatedFusion(HS, HT)
        del HS, HT
        return torch.add(X, H)


class transformAttention(nn.Module):
    def __init__(self, K, d):
        super(transformAttention, self).__init__()
        total_dim = K * d
        self.K = K
        self.d = d
        self.FC_q = FC(input_dims=total_dim, units=total_dim, activations=F.relu)
        self.FC_k = FC(input_dims=total_dim, units=total_dim, activations=F.relu)
        self.FC_v = FC(input_dims=total_dim, units=total_dim, activations=F.relu)
        self.FC = FC(input_dims=total_dim, units=total_dim, activations=F.relu)

    def forward(self, X, STE_his, STE_pred):
        batch_size = X.shape[0]

        query = self.FC_q(STE_pred)
        key = self.FC_k(STE_his)
        value = self.FC_v(X)

        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)

        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


class transformer(nn.Module):
    def __init__(self, args):
        super(transformer, self).__init__()
        layers = args.block_layers
        num_heads = args.num_heads
        hidden_dim = args.hidden_dim
        total_dim = num_heads * hidden_dim
        self.num_hist = args.num_hist

        self.emb1 = nn.Linear(args.num_nodes, total_dim)
        self.layernorm = nn.LayerNorm(total_dim)
        self.emb2 = nn.Linear(args.num_nodes, total_dim)

        self.STEmbedding = STEmbedding(total_dim)
        self.STAttBlock_1 = nn.ModuleList([STAttBlock(num_heads, hidden_dim) for _ in range(layers)])
        self.STAttBlock_2 = nn.ModuleList([STAttBlock(num_heads, hidden_dim) for _ in range(layers)])
        self.transformAttention = transformAttention(num_heads, hidden_dim)
        self.FC_1 = FC(input_dims=[args.input_dim, total_dim], units=[total_dim, total_dim], activations=[F.relu, None])
        self.FC_2 = FC(input_dims=[total_dim, total_dim], units=[total_dim, args.output_dim], activations=[F.relu, None])

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight.data)
            # elif isinstance(m, nn.LayerNorm):
            #     nn.init.xavier_normal_(m.weight.data)
            # elif isinstance(m, nn.Conv2d):
            #     nn.init.kaiming_normal_(m.weight.data)
            # elif isinstance(m, nn.BatchNorm2d)

    def forward(self, X, SE, TE):
        X = self.FC_1(X)
        SE = self.emb1(SE[0]) + self.layernorm(self.emb2(SE[1]))

        STE = self.STEmbedding(SE, TE)
        STE_his = STE[:, :self.num_hist]
        STE_pred = STE[:, self.num_hist:]

        for net in self.STAttBlock_1:
            X = net(X, STE_his)

        X = self.transformAttention(X, STE_his, STE_pred)

        for net in self.STAttBlock_2:
            X = net(X, STE_pred)

        X = self.FC_2(X)

        del STE, STE_his, STE_pred
        return torch.squeeze(X, 3)

