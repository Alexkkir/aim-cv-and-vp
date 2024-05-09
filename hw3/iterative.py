import torch
from torch.autograd import Variable
import numpy as np
from diffjpeg import DiffJPEG
import random


# iterative attack baseline (IFGSM attack)
def attack(
    image,
    model=None,
    metric_range=100,
    device="cpu",
    eps=10 / 255,
    iters=10,
    alpha=1 / 255,
    gamma = 1e4
):
    """
    Attack function.
    Args:
    image: (torch.Tensor of shape [1,3,H,W]) clear image to be attacked.
    model: (PyTorch model): Metric model to be attacked. Should be an object of a class that inherits torch.nn.module and has a forward method that supports backpropagation.
    iters: (int) number of iterations. Can be ignored, during testing always set to default value.
    alpha: (float) step size for signed gradient methods. Can be ignored, during testing always set to default value.
    device (str or torch.device()): Device to use in computaions.
    eps: (float) maximum allowed pixel-wise difference between clear and attacked images (in 0-1 scale).
    Returns:
        torch.Tensor of shape [1,3,H,W]: adversarial image with same shape as image argument.
    """
    image = image.clone().to(device)
    additive = torch.zeros_like(image).to(device)
    additive = Variable(additive, requires_grad=True)

    qfs = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    # qfs = np.array([30])
    # qfs = np.array([5, 10, 20, 30, 40, 50])

    optimizer = torch.optim.Adam([additive], lr=1e0)

    for _ in range(iters):
        # <YOUR CODE HERE>
        x_adv = image + additive
        x_adv = x_adv.clamp(min=0, max=1)

        qf = random.choice(qfs)
        jpeg = DiffJPEG(
            height=x_adv.size(-2), width=x_adv.size(-1), differentiable=True, quality=qf
        ).to(device)
        x_comp = jpeg(x_adv)

        pred = model(x_comp)
        # print(pred, eps, additive, gamma, torch.log(eps - additive.abs()), torch.log(eps - additive.abs()) * gamma)
        loss = 1 - pred / metric_range  - ((eps - additive.abs()) ** 5).mean().cpu() * gamma
        #torch.log(eps - additive.abs() + 1e-8).mean().cpu() * gamma

        # loss += (x_comp
        loss.backward()
        # signs = additive.grad.data.sign()
        # additive.data = additive.data - signs * alpha
        # # print(additive)
        # # print(additive, grad)
        # additive.data.clamp_(-eps, eps)
        # additive.grad.zero_()

        optimizer.step()
        optimizer.zero_grad()
        additive.data.clamp_(-eps, eps)

    res_image = image + additive

    res_image = (res_image).data.clamp_(min=0, max=1)
    return res_image
