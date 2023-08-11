from copy import deepcopy
import torch
from torch import nn
from tqdm import tqdm
import gc

import cQIGen as qinfer
import math
import numpy as np
from gekko import GEKKO
from logging import getLogger

logger = getLogger(__name__)


def mem_model(N, M, T, mu, tu, bits, l1, p, gs):
    m = GEKKO() # create GEKKO model
    #cinfergen if bits==3:
        # tu = tu*3
    B = m.Const(value=bits)
    TP = m.Const(value=T//p)
    k = m.Var(1,integer=True,lb=1)
    z = m.Var(1,integer=True,lb=1)
    w = m.Var(1,integer=True,lb=1)
    y = m.Var(1,integer=True,lb=1)
    v = m.Var(1,integer=True,lb=1)
    mb = m.Var(mu,integer=True,lb=1)
    if gs != -1:
        gg = m.Var(1,integer=True,lb=1)
    tb = m.Var(tu,integer=True,lb=1,ub=int(T/p))
    L = m.Var(integer=True,lb=0,ub=l1)
    m.Equation(L == 32 * mb * N + B * mb * tb + 32 * tb * N)
    m.Equation(mb * k == M)
    if gs != -1:
        m.Equation(gs * gg == mb)
    # m.Equation(tb * z == T)
    m.Equation(tb * z == TP)
    m.Equation(mu * w == mb)
    m.Equation(tu * y == tb)
    # m.Equation(tb * v == tt)
    m.Maximize(L)
    m.options.SOLVER = 1
    m.solver_options = ['minlp_maximum_iterations 1000', \
                # minlp iterations with integer solution
                'minlp_max_iter_with_int_sol 10', \
                # treat minlp as nlp
                'minlp_as_nlp 0', \
                # nlp sub-problem max iterations
                'nlp_maximum_iterations 100', \
                # 1 = depth first, 2 = breadth first
                'minlp_branch_method 2', \
                # maximum deviation from whole number
                'minlp_integer_tol 0.00', \
                # covergence tolerance
                'minlp_gap_tol 0.01']
    try:
        m.solve(disp=False)
    except:
        try:
            m.solver_options = ['minlp_maximum_iterations 1000', \
                            # minlp iterations with integer solution
                            'minlp_max_iter_with_int_sol 10', \
                            # treat minlp as nlp
                            'minlp_as_nlp 0', \
                            # nlp sub-problem max iterations
                            'nlp_maximum_iterations 100', \
                            # 1 = depth first, 2 = breadth first
                            'minlp_branch_method 1', \
                            # maximum deviation from whole number
                            'minlp_integer_tol 0.00', \
                            # covergence tolerance
                            'minlp_gap_tol 0.01']
            m.solve(disp=False)
        except:
            # mytb = T//p
            mytb = tu
            if gs != -1:
                mymb = gs
                while 32 * (mymb + gs) * N + bits * (mymb + gs) * mytb + 32 * mytb * N < l1:
                    mymb += gs
                while M % mymb != 0:
                    mymb -= gs
                return (int(mymb), int(mytb))
            else:
                mymb = mu
                while 32 * (mymb + mu) * N + bits * (mymb + mu) * mytb + 32 * mytb * N < l1:
                    mymb += mu
                while M % mymb != 0:
                    mymb -= mu
                return (int(mymb), int(mytb))

    return (int(mb.value[0]), int(tb.value[0]))

params = {}

def compute_reductions(x, gs=-1, cpp=True):
    if cpp:
        if len(x.shape) != 1:
            rows, cols = x.shape
        else:
            rows = 1
            cols = x.shape[0]
        if gs == -1:
            out = torch.zeros(rows).float().contiguous()
            mygs = cols
        else:
            out = torch.zeros(rows, cols // gs).float().contiguous()
            mygs = gs
        
        qinfer.compute_reduction_cpp(x, out, rows, cols, mygs)
        return out
    if gs == -1: 
        if len(x.shape) != 1:
            return torch.sum(x,1)
        else:
            return torch.sum(x)
    else:
        if len(x.shape) != 1:
            rows, cols = x.shape
            out = torch.zeros(rows, cols // gs).float().contiguous()
            for i in range(cols // gs):
                out[:,i] = torch.sum(x[:,i*gs:(i+1)*gs],1)
            return out
        else:
            cols = x.shape[0]
            out = torch.zeros(cols // gs).float().contiguous()
            for i in range(cols // gs):
                out[i] = torch.sum(x[i*gs:(i+1)*gs])
            return out

def process_zeros_scales(zeros, scales, bits, M):
    if zeros.dtype != torch.float32:
        new_zeros = torch.zeros_like(scales).float().contiguous()
        if bits == 4:
            qinfer.unpack_zeros4(zeros, new_zeros, new_zeros.shape[0], new_zeros.shape[1])
        elif bits == 2:
            qinfer.unpack_zeros2(zeros, new_zeros, new_zeros.shape[0], new_zeros.shape[1])
        elif bits == 3:
            logger.info("Unpacking zeros for 3 bits")
        new_scales = scales.contiguous()
    else:
        if scales.shape[1] != M:
            new_scales = scales.transpose(0,1).contiguous()
        else:
            new_scales = scales.contiguous()
        if zeros.shape[1] != M:
            new_zeros = zeros.transpose(0,1).contiguous()
        else:
            new_zeros = zeros.contiguous()

    return new_zeros, new_scales
    
class QuantLinear(nn.Module):
    QUANT_TYPE = "qigen"

    def __init__(self, bits, group_size, infeatures, outfeatures, bias=None, qweights=None, zeros=None, scales=None, hint=1, p=8, l1=2**18):
        super().__init__()
        if bits not in [2, 4]:
            raise NotImplementedError("Only 2,4 bits are supported.")
        
        self.bits = bits
        pack = 32 // bits

        self.infeatures = infeatures
        self.outfeatures = outfeatures

        n = hint
        m = self.infeatures
        t = self.outfeatures

        #registers for now are fixed
        if bits == 3:
            packed = 32
            unroll = 3
            nu = 1 #args.n
            mu = 32
            tu = 32
        else:
            packed = 32 // bits
            unroll = 2
            nu = 1 #args.n
            mu = 16
            tu = 32
        
        nb = n # it's always small for transformers

        global params
        if (m,t) in params:
            mb = params[(m,t)][0]
            tb = params[(m,t)][1]
        else:
            mb, tb = mem_model(n, m, t, mu, tu, bits, l1, p, group_size)
            params[(m,t)] = (mb,tb)

        split = np.ones(p)
        split = split * tb
        while np.sum(split) < t:
            split = split + tb

        idx = p - 1
        while np.sum(split) > t:
            split[idx] = split[idx] - tb
            idx = idx - 1

        assert(np.sum(split) == t)

        split = split.astype(int)
        self.tt = int(split[0])

        if split[0] == split[-1]:
            self.cutoff = int(p+1)
        else:
            self.cutoff = int(idx + 1)

        self.mb = mb #// packed
        self.tb = tb

        self.group_size = group_size

        if bias is None:
            self.bias = torch.zeros(self.outfeatures)
        else:
            self.bias = bias.contiguous()

        self.zeros, self.scales = process_zeros_scales(zeros, scales, bits, self.outfeatures)

        if bits == 4:
            self.weight = torch.zeros(int(self.infeatures // packed * self.outfeatures)).int().contiguous()
            qinfer.pack4(qweights.int().contiguous(),self.weight, self.infeatures // packed, self.outfeatures, self.mb, self.tb, self.cutoff)# * (self.tt//tb))
        elif bits == 3:
            self.weight = torch.zeros(int(self.infeatures // packed * 3 * self.outfeatures)).int().contiguous()
            qinfer.pack3(qweights.int().contiguous(),self.weight, self.infeatures // packed * 3, self.outfeatures, self.mb // packed * 3, self.tb, self.cutoff)
        elif bits == 2:
            self.weight = torch.zeros(int(self.infeatures // packed * self.outfeatures)).int().contiguous()
            qinfer.pack2(qweights.int().contiguous(),self.weight, self.infeatures // packed, self.outfeatures, self.mb, self.tb, self.cutoff)# * (self.tt//tb))
                
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape((-1, x.shape[-1])).to(torch.float32)
        B = x.shape[0]
        new_x = x.T.contiguous()
        out = torch.zeros((B, self.outfeatures), dtype=torch.float32)
        sums = compute_reductions(x,gs=self.group_size,cpp=True).contiguous()
        if self.group_size == -1:
            if self.bits == 4:
                qinfer.forward4(new_x, self.weight, out, self.bias, self.scales, self.zeros, sums, 
                                B, self.infeatures, self.outfeatures, B, self.mb, self.tb, self.tt, self.cutoff)
            elif self.bits == 2:
                qinfer.forward2(new_x, self.weight, out, self.bias, self.scales, self.zeros, sums, 
                                B, self.infeatures, self.outfeatures, B, self.mb, self.tb, self.tt, self.cutoff)
            elif self.bits == 3:
                qinfer.forward3(new_x, self.weight, out, self.bias, self.scales, self.zeros, sums, 
                                B, self.infeatures, self.outfeatures, B, self.mb, self.tb, self.tt, self.cutoff)
        else:
            if self.bits == 4:
                qinfer.forward_gs4(new_x, self.weight, out, self.bias, self.scales, self.zeros, sums, 
                                   B, self.infeatures, self.outfeatures, B, self.mb, self.tb, self.tt, self.group_size, self.cutoff)
            elif self.bits == 2:
                qinfer.forward_gs2(new_x, self.weight, out, self.bias, self.scales, self.zeros, sums, 
                                   B, self.infeatures, self.outfeatures, B, self.mb, self.tb, self.tt, self.group_size, self.cutoff)
            elif self.bits == 3:
                qinfer.forward_gs3(new_x, self.weight, out, self.bias, self.scales, self.zeros, sums,
                                   B, self.infeatures, self.outfeatures, B, self.mb, self.tb, self.tt, self.group_size, self.cutoff)
        return out.reshape(out_shape)