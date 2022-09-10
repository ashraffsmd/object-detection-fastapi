import torch

class MishCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return torch.nn.functional.mish(inp)
    
    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        if not ctx.needs_input_grad[0]: return (None,)
        return torch.nn.functional.mish(inp, grad_out)
        

class MishCuda(torch.nn.Module):
    def forward(self, inp): return MishCudaFunction.apply(inp)
