import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Create a list of N random 2D points.

    Args:
    ----
        N: number of points.

    Returns:
    -------
        List of N Points.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generates a Graph with N points, labeled with y={0,1}. Points are split in
    half at x=0.5.

    Args:
    ----
        N: number of points.

    Returns:
    -------
        Graph with N points split about x=0.5.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generate a Graph with N points, labeled with y={0,1}. Points are split along a diagonal line.

    Args:
    ----
        N: number of points.

    Returns:
    -------
        Graph with N points split diagonally.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generates a Graph with N points, labeled with y={0,1}. Class of points split down the middle by other class of points.

    Args:
    ----
        N: number of points.

    Returns:
    -------
        Graph with N points split in the middle.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generates a Graph with N points, labeled with y={0,1}. Points are split into 4 quadrants.

    Args:
    ----
        N: number of points.

    Returns:
    -------
        Graph with N points split into 4 quadrants.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Geneartes a Graph with N points, labeled with y={0,1}. Class of points in a circle in the center, other class surrounds the circle.

    Args:
    ----
        N: number of points.

    Returns:
    -------
        Graph with N points split into a circle.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generates a Graph with N points, labeled with y={0,1}. Points form a spiral.

    Args:
    ----
        N: number of points.

    Returns:
    -------
        Graph with N points aligned in a spiral.

    """

    def x(t: float) -> float:
        """Calculates the value of t * math.cos(t) / 20.0.

        Args:
        ----
            t: input value.

        Returns:
        -------
            result of the expression t * math.cos(t) / 20.0.

        """
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        """Calculates the value of t * math.sin(t) / 20.0.

        Args:
        ----
            t: input value.

        Returns:
        -------
            Result of the expression t * math.sin(t) / 20.0.

        """
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
