import torch
from typing import Optional
from typing import Callable, Iterable
import math

def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """ 4.1 Cross-entropy loss """
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
    log_sum_exp = torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))
    log_target = logits[torch.arange(logits.shape[0]), labels]
    cross_entropy = log_sum_exp - log_target
    return cross_entropy.mean()


class AdamW(torch.optim.Optimizer):
    """ 4.3 AdamW """
    def __init__(self,               
        params: Iterable[torch.nn.Parameter], 
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-8):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "m": None,
            "v": None,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                t = state.get("t", 0)
                if t == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                m = state["m"]
                v = state["v"]
                grad = p.grad.data

                t += 1
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= lr_t * m / (v**0.5 + eps)

                if weight_decay != 0:
                    p.data -= lr * weight_decay * p.data

                state["t"] = t
                state["m"] = m
                state["v"] = v

        return loss


def cosine_learing_rate_scheduler(
    time_step: int,
    max_lr: float,
    min_lr: float,
    t_warmup: int,
    t_iter: int,
) -> float:
    """ 4.4 Learning rate scheduling
    Cosine learning rate scheduler with warmup.

    Args:
        time_step: Current iteration.
        max_lr: Maximum learning rate.
        min_lr: Minimum learning rate.
        t_warmup: Number of warmup iterations.
        t_iter: Number of iterations in one cosine cycle.

    Returns:
        Learning rate at the given iteration.
    """
    if time_step < t_warmup:
        return max_lr * time_step / t_warmup
    if time_step <= t_iter:
        return min_lr + 0.5 * (
            1 + math.cos(math.pi * (time_step - t_warmup) / (t_iter - t_warmup))
        ) * (max_lr - min_lr)
    return min_lr


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """ 4.5 Gradient clipping
    Clips gradients of the given parameters to have a maximum L2 norm.

    Args:
        parameters: Iterable of model parameters.
        max_l2_norm: Maximum allowed L2 norm for the gradients.

    Returns:
        None
    """
    pd = [p.grad.data for p in parameters if p.grad is not None]
    if not pd:
        return
    norm = torch.norm(torch.stack(pd), p=2)
    if norm <= max_l2_norm:
        return
    for pt in pd:
        pt *= max_l2_norm / (norm + 1e-6)
