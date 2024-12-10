"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Sets whether the tensor should track gradients for automatic differentiation.

        Args:
        ----
            x: If True, the tensor will track history for gradient computation.
                    If False, the tensor will not track gradients.

        """
        self.history = History()

    def requires_grad(self) -> bool:
        """Checks whether the tensor is tracking gradients for automatic differentiation.

        Returns
        -------
            True if the tensor is tracking gradients, otherwise False.

        """
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Creates a tensor filled with zeros.

        If no shape is provided, the tensor will have the same shape as the calling tensor.
        The resulting tensor uses the same backend as the current tensor.

        Args:
        ----
            shape: Shape of the tensor to create. If None,
                the shape of the current tensor is used.

        Returns:
        -------
            A new tensor filled with zeros, having the specified shape or
            the shape of the current tensor.

        """

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Checks if the tensor is constant (i.e., it does not require gradient tracking).

        A tensor is considered constant if it has no associated computation history,
        meaning that it does not require gradient computation.

        Returns
        -------
            True if the tensor is constant (no gradient required), False otherwise.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Retrieves the parent variables that contributed to this tensor's creation.

        The parents are the input variables used in the operation that created this tensor.
        This property assumes the tensor has a computation history (i.e., gradient tracking is enabled).

        Returns
        -------
            An iterable containing the parent variables (inputs).

        Raises
        ------
            AssertionError: If the tensor does not have a computation history (history is None).

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Implements the chain rule for backpropagation, calculating the gradients
        of the input variables with respect to the output of a function.

        This method assumes that the tensor has a computation history, and it calls
        the `_backward` method of the function used to create the tensor. It applies
        the chain rule by propagating the gradient `d_output` backward through the
        function.

        Args:
        ----
            d_output: The gradient of the output with respect to some loss function.

        Returns:
        -------
            An iterable of tuples, where each tuple contains:
                - A parent `Variable` that contributed to this tensor.
                - The corresponding gradient with respect to that variable, expanded
                  to match the variable's shape if necessary.

        Raises:
        ------
            AssertionError: If the tensor does not have a valid computation history,
                            if the function that created the tensor is not available,
                            or if the gradients returned do not match the number of inputs.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Computes the backward pass for this tensor, calculating the gradients of
        all inputs that contributed to its value through the computation graph.

        If no `grad_output` is provided, it is assumed that this tensor is a scalar
        (with shape `(1,)`), and the gradient is initialized to 1.0. Otherwise, the
        provided `grad_output` is used to backpropagate through the computation graph.

        Args:
        ----
            grad_output: The gradient of the output with respect
                to some loss. If `None`, the gradient is set to a scalar tensor of 1.0
                (default is `None`).

        Raises:
        ------
            AssertionError: If the tensor is not a scalar and no `grad_output` is provided.

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    @property
    def dims(self) -> int:
        """Returns
        int : dimensionality of the tensor

        """
        return self._tensor.dims

    @property
    def size(self) -> int:
        """Returns
        int : size of the tensor

        """
        return self._tensor.size

    # Functions
    def __add__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, -self._ensure_tensor(b))

    def __mul__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, self._ensure_tensor(b))

    def __lt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b: TensorLike) -> Tensor:
        return EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self._ensure_tensor(b), self)

    def __neg__(self) -> Tensor:
        return Neg.apply(self)

    def __radd__(self, b: TensorLike) -> Tensor:
        return self + b

    def __rmul__(self, b: TensorLike) -> Tensor:
        return self * b

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Computes the logical 'all' operation along a specified dimension of the tensor.
        If `dim` is provided, the reduction is performed along that dimension. If no
        dimension is provided, the reduction is applied across the entire tensor.

        Args:
        ----
            dim: The dimension along which to perform the 'all' operation.
                If `None`, the operation is applied to all elements in the tensor (default is `None`).

        Returns:
        -------
            A tensor containing the result of the 'all' operation, either reduced
            along the specified dimension or over the entire tensor.

        """
        if dim is None:
            return All.apply(self.view(int(self.size)), self._ensure_tensor(0))
        else:
            return All.apply(self, self._ensure_tensor(dim))

    def is_close(self, y: Tensor) -> Tensor:
        """Compares each element of this tensor to another tensor-like object `b`,
        returning a tensor that indicates if the values are element-wise close
        within a tolerance.

        Args:
        ----
            y : A tensor-like object to compare against this tensor. It
                will be converted to a tensor if it is not already one.

        Returns:
        -------
            A tensor of boolean values where each element is `True` if the
            corresponding elements in the two tensors are close to each other, and `False` otherwise.

        """
        return IsClose.apply(self, y)

    def sigmoid(self) -> Tensor:
        """Computes the sigmoid of the scalar.

        Returns
        -------
        The result of applying the sigmoid function to the scalar.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Computes the relu of the scalar.

        Returns
        -------
        The result of applying the relu function to the scalar.

        """
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Computes the logarithm of the scalar.

        Returns
        -------
        The result of applying the logarithmic function to the scalar.

        """
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Computes the exponential of the scalar.

        Returns
        -------
        The result of applying the exponent function to the scalar.

        """
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Computes the sum of elements in the tensor along the specified dimension.
        If no dimension is provided, it computes the sum over all dimensions.

        Args:
        ----
            dim: The dimension along which to compute the sum. If
                `None`, the sum is computed over all dimensions.

        Returns:
        -------
            A tensor representing the sum of elements along the specified
            dimension or the entire tensor if `dim` is not provided.

        """
        if dim is None:
            return Sum.apply(
                self.contiguous().view(int(self.size)), self._ensure_tensor(0)
            )
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Computes the mean of elements in the tensor along the specified dimension.
        If no dimension is provided, it computes the mean over all elements.

        Args:
        ----
            dim: The dimension along which to compute the mean.
                If `None`, the mean is computed over all elements in the tensor.

        Returns:
        -------
            A tensor representing the mean of elements along the specified
            dimension or the entire tensor if `dim` is not provided.

        """
        if dim is not None:
            return self.sum(dim) / int(self.shape[dim])
        else:
            return self.sum() / int(self.size)

    def permute(self, *order: int) -> Tensor:
        """Permutes the dimensions of the tensor according to the specified order.

        Args:
        ----
            *order: The dimensions to permute. The order of dimensions specified
                determines the new arrangement of the tensor's dimensions.

        Returns:
        -------
            A new tensor with dimensions permuted according to the specified
            order.

        """
        return Permute.apply(self, tensor(list(order)))

    def view(self, *shape: int) -> Tensor:
        """Reshapes the tensor to the specified dimensions.

        Args:
        ----
            *shape: The desired dimensions for the reshaped tensor. The product
                of these dimensions must equal the number of elements in the original
                tensor.

        Returns:
        -------
            A new tensor with the specified shape.

        """
        return View.apply(self, tensor(list(shape)))

    def zero_grad_(self) -> None:
        """Clears the gradients of the tensor, setting them to None.

        This function is typically used before performing backpropagation to
        ensure that gradients from the previous step do not accumulate.
        """
        self.grad = None
