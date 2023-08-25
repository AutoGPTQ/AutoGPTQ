import torch.nn as nn


class GeneralQuantLinear(nn.Linear):
    def __init__(self, quant_linear_module):
        super().__init__(
            in_features=quant_linear_module.infeatures,
            out_features=quant_linear_module.outfeatures,
            bias=quant_linear_module.bias is not None
        )
        self.infeatures = quant_linear_module.infeatures
        self.outfeatures = quant_linear_module.outfeatures
        self.bits = quant_linear_module.bits
        self.group_size = quant_linear_module.group_size
        self.maxq = quant_linear_module.maxq

        self.weight.requires_grad = False

        self.weight.data = quant_linear_module.qweight
        self.register_buffer('qweight', quant_linear_module.qweight)
        if quant_linear_module.bias is not None:
            self.bias.data = quant_linear_module.bias

        self.register_buffer('qzeros', quant_linear_module.qzeros)
        self.register_buffer('scales', quant_linear_module.scales)
        self.register_buffer('g_idx', quant_linear_module.g_idx)

        # arg of qlinear_cuda and qlinear_cuda_old
        if hasattr(quant_linear_module, "wf"):
            self.wf = quant_linear_module.wf
        # arg of qlinaer_cuda and qlinear_cuda_old
        if hasattr(quant_linear_module, "kernel_switch_threshold"):
            self.kernel_switch_threshold = quant_linear_module.kernel_switch_threshold
        # arg of qlinaer_cuda and qlinear_cuda_old
        if hasattr(quant_linear_module, "autogptq_cuda_available"):
            self.autogptq_cuda_available = quant_linear_module.autogptq_cuda_available
        # arg of qlinaer_cuda and qlinear_cuda_old
        if hasattr(quant_linear_module, "autogptq_cuda"):
            self.autogptq_cuda = quant_linear_module.autogptq_cuda
        # arg of qlinear_cuda_old
        if hasattr(quant_linear_module, "half_indim"):
            self.half_indim = quant_linear_module.half_indim
        # arg of qlinear_cuda_old
        if hasattr(quant_linear_module, "use_cuda_fp16"):
            self.use_cuda_fp16 = quant_linear_module.use_cuda_fp16
        # args of qlinear_exllama
        if hasattr(quant_linear_module, "_use_act_order"):
            self._use_act_order = quant_linear_module._use_act_order
        # arg of qlinaer_exllama
        if hasattr(quant_linear_module, "width"):
            self.width = quant_linear_module.width
        # arg of qlinear_exllama
        if hasattr(quant_linear_module, "q4"):
            self.q4 = quant_linear_module.q4

        self.trainable = quant_linear_module.trainable

        self.forward = quant_linear_module.forward

    @classmethod
    def convert_to_torch_linear(cls, model: nn.Module, target_module_type: "QuantLinear"):
        for name, m in model.named_modules():
            if not isinstance(m, target_module_type):
                continue
            new_m = cls(m)
            if '.' in name:
                parent_name = name.rsplit('.', 1)[0]
                child_name = name[len(parent_name) + 1:]
                parent = model.get_submodule(parent_name)
            else:
                parent_name = ''
                parent = model
                child_name = name

            setattr(parent, child_name, new_m)
