import torch

from ..qlinear import GeneralQuantLinear
from ..qlinear.qlinear_cuda import QuantLinear as CudaQuantLinear
from ..qlinear.qlinear_cuda_old import QuantLinear as OldCudaQuantLinear
try:
    from ..qlinear.qlinear_triton import QuantLinear as TritonQuantLinear
except:
    TritonQuantLinear = None
try:
    from ..qlinear.qlinear_exllama import QuantLinear as ExllamaQuantLinear
except:
    ExllamaQuantLinear = None


class FusedGeneralQuantLinear(GeneralQuantLinear):
    def __init__(self, quant_linear_module):
        super(FusedGeneralQuantLinear, self).__init__(quant_linear_module)

    @classmethod
    def fuse(
        cls,
        q_proj,
        k_proj=None,
        v_proj=None,
    ):
        qweights, qzeros, scales, g_idx, bias = [], [], [], [], []
        outfeatures = 0
        for module in [q_proj, k_proj, v_proj]:
            if module is not None:
                qweights.append(module.qweight)
                qzeros.append(module.qzeros)
                scales.append(module.scales)
                g_idx.append(module.g_idx)
                bias.append(module.bias)
                outfeatures += module.outfeatures

        if bias[0] is None:
            bias = None

        if len(qweights) > 1:
            qweights = torch.cat(qweights, dim=1)
            qzeros = torch.cat(qzeros, dim=1)
            scales = torch.cat(scales, dim=1)
            g_idx = torch.cat(g_idx, dim=0)
            if bias is not None:
                bias = torch.cat(bias, dim=0)

        qlinear_args = (
            q_proj.bits,
            q_proj.group_size,
            q_proj.infeatures,
            outfeatures,
            bias is not None
        )
        qlinear_kwargs = {"trainable": q_proj.trainable}
        if isinstance(q_proj, (OldCudaQuantLinear, CudaQuantLinear)):
            qlinear_kwargs["kernel_switch_threshold"] = q_proj.kernel_switch_threshold
            if isinstance(q_proj, OldCudaQuantLinear):
                qlinear_kwargs["use_cuda_fp16"] = q_proj.use_cuda_fp16
                QuantLinear = OldCudaQuantLinear
            else:
                QuantLinear = CudaQuantLinear
        elif isinstance(q_proj, TritonQuantLinear):
            QuantLinear = TritonQuantLinear
        else:
            QuantLinear = ExllamaQuantLinear
        fused_proj = QuantLinear(*qlinear_args, **qlinear_kwargs)

        fused_proj.qweight = qweights
        fused_proj.qzeros = qzeros
        fused_proj.scales = scales
        fused_proj.g_idx = g_idx
        fused_proj.bias = bias

        del q_proj, k_proj, v_proj

        return cls(fused_proj)
