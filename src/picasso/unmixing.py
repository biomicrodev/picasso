from __future__ import annotations

import math
import sys

import numpy as np
import numpy.typing as npt
from fast_histogram import histogram2d, histogram1d
from scipy.optimize import OptimizeResult, minimize
from skimage import img_as_float
from tqdm import tqdm


def shannon_entropy(a: npt.NDArray) -> float:
    """
    This runs very often, so we should do our best to make this fast.
    See 'profiling/shannon-entropy.ipynb' for why this was chosen
    """
    # seems like flattened arrays are faster, but .flatten() produces a copy so we
    # stick with .ravel()
    a = a.ravel()
    a /= a.sum()
    a = a[np.nonzero(a != 0)]
    a *= np.log2(a)
    return -a.sum().item()


def mutual_information(x: npt.NDArray, y: npt.NDArray, *, bins=100) -> float:
    x = x.ravel()
    y = y.ravel()

    x_range = (x.min(), x.max())
    y_range = (y.min(), y.max())

    # TODO: even though fast-histogram is pretty dang fast, consider boost-histogram?
    #  measure perf diff
    c_xy = histogram2d(x, y, bins, (x_range, y_range))
    c_x = histogram1d(x, bins, x_range)
    c_y = histogram1d(y, bins, y_range)

    h_xy = shannon_entropy(c_xy)
    h_x = shannon_entropy(c_x)
    h_y = shannon_entropy(c_y)

    return h_x + h_y - h_xy


def regional_mi(x: npt.NDArray, y: npt.NDArray) -> float:
    """
    FIXME not optimized, but doesn't give us better results than regular mi, so not
     worth it at the moment
    """
    x = np.copy(x)
    y = np.copy(y)

    x /= x.max()
    y /= y.max()

    r = 1

    stack = []
    for ri in range(2 * r):
        for rj in range(2 * r):
            stack.append(x[ri : -2 * r + ri, rj : -2 * r + rj])
    for ri in range(2 * r):
        for rj in range(2 * r):
            stack.append(y[ri : -2 * r + ri, rj : -2 * r + rj])
    stack = np.stack(stack)
    stack = np.reshape(stack, (stack.shape[0], -1))

    dim, n_points = stack.shape[:2]
    hdim = dim // 2

    mean = np.mean(stack, axis=1, keepdims=True)
    stack -= mean
    cov = stack @ stack.T / n_points
    h_xy = math.log(np.linalg.det(cov))
    h_x = math.log(np.linalg.det(cov[:hdim, :hdim]))
    h_y = math.log(np.linalg.det(cov[hdim:, hdim:]))
    return h_x + h_y - h_xy


def minimize_mi(x: npt.NDArray, y: npt.NDArray, *, init_alpha=0.0) -> float:
    def func(alpha: float):
        # return regional_mutual_information(x, y - alpha * x)
        return mutual_information(x, y - alpha * x)

    non_neg_alpha_constraint = dict(
        type="ineq",
        fun=lambda a: a,
        catol=1e-7,
    )

    result: OptimizeResult = minimize(
        func,
        x0=np.array([init_alpha]),
        method="COBYLA",
        constraints=non_neg_alpha_constraint,
    )
    return result.x.item()


def compute_unmixing_matrix(
    image: npt.NDArray,
    *,
    max_iters=1_000,
    step_mult=0.1,
    constrain_diag=True,
    verbose=False,
    return_iters=False,
) -> npt.NDArray:
    n_channels = image.shape[0]
    mats = []

    image = img_as_float(image)
    image_orig = image.copy()

    mat_cumul = np.eye(n_channels)
    mat_last = np.eye(n_channels)

    for _ in tqdm(
        range(max_iters),
        disable=not verbose,
        desc="Unmixing iterations",
        total=0,
        file=sys.stdout,
    ):
        mat = np.eye(n_channels, dtype=float)

        for row in range(n_channels):
            for col in range(n_channels):
                if row == col:
                    continue

                mat[row, col] = -step_mult * minimize_mi(
                    image[col, ...], image[row, ...]
                )

        # check this early on
        if np.allclose(mat, mat_last):
            break

        # update matrix
        mat_last = mat
        assert mat_cumul.shape == (n_channels, n_channels)
        mat_cumul = mat @ mat_cumul
        if constrain_diag:
            for i in range(n_channels):
                mat_cumul[i, i] = 1
        mats.append(mat_cumul)

        # update the next iteration of image
        # several times faster than np.einsum
        assert mat.shape == (n_channels, n_channels)
        assert image.ndim == 3
        image = np.tensordot(mat_cumul, image_orig, axes=1)

    if return_iters:
        return np.stack(mats)
    else:
        return mats[-1]
