from typing import Tuple

from .tensor import Tensor
from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor_functions import Function, rand

# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw

    # Reshape and permute to prepare for pooling
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    tiled = reshaped.permute(0, 1, 2, 4, 3, 5)
    tiled = tiled.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply 2D average pooling on the input tensor.

    Args:
    ----
        input: input tensor with shape (batch, channel, height, width).
        kernel: Tuple (kernel_height, kernel_width) specifying the pooling dimensions.

    Returns:
    -------
        A tensor of shape (batch, channel, new_height, new_width) after pooling.

    """
    tiled, new_height, new_width = tile(input, kernel)

    return tiled.mean(dim=4).view(input.shape[0], input.shape[1], new_height, new_width)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor

    Args:
    ----
        input: the input tensor.
        dim: dimension to reduce.

    Returns:
    -------
        A tensor of shape (batch, channel, height, width)

    """
    max_val = max_reduce(input, dim)
    return input == max_val


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Performs the forward pass of the max function.

        Args:
        ----
            ctx: context object to store any values for the backward pass.
            input: the input tensor to the max function.
            dim: the input dimension.

        Returns:
        -------
            The result of the max function.

        """
        ctx.save_for_backward(input, dim)
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Performs the backward pass of the max function.

        Args:
        ----
            ctx: context object containing saved values from the forward pass.
            grad_output: the gradient of the output with respect to the final objective.

        Returns:
        -------
            The gradient with respect to both inputs.

        """
        input, dim = ctx.saved_values
        return grad_output * argmax(input, int(dim.item())), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the max of a tensor.

    Args:
    ----
        input: the input tensor.
        dim: dimension to perform max.

    Returns:
    -------
        A tensor with max function performed on it.

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax of a tensor.

    Args:
    ----
        input: the input tensor.
        dim: the input dimension.

    Returns:
    -------
        A tensor with the softmax function performed on it.

    """
    num = input.exp()
    denom = num.sum(dim)
    return num / denom


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax of a tensor.

    Args:
    ----
        input: the input tensor.
        dim: the dimension to perform the log of the softmax.

    Returns:
    -------
        A tensor with the log of the softmax function performed on it.

    """
    z = input - max(input, dim)
    logsumexp = (z).exp().sum(dim).log()
    return z - logsumexp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply 2D max pooling on input tensor.

    Args:
    ----
        input: the input tensor.
        kernel: the input kernel.

    Returns:
    -------
        A tensor of sixe (batch, channel, new_height, new_width)

    """
    tiled, new_height, new_width = tile(input, kernel)
    return max(tiled, dim=4).view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor.

    Args:
    ----
        input: input tensor.
        p: drouput rate.
        ignore: if True, dropout is ignored and input returmed unchanged.

    Returns:
    -------
        A tensor with dropout applied.

    """
    if ignore:
        return input
    return input * (rand(input.shape) > p)
