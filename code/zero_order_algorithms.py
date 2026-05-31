import numpy as np
from abc import ABC, abstractmethod
from base_optimizer import BaseOptimizer
import math
import warnings
from typing import Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import norm


class NelderMead(BaseOptimizer):
    """Nelder-Mead simplex algorithm for black-box optimization. Integrates directly with the WIND benchmark."""

    def __init__(
        self, alpha=1.0, beta=0.5, gamma=2.0, init_step=0.1, name="NelderMead"
    ):
        super().__init__(name)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.init_step = init_step

        self.reset()

    def reset(self) -> None:
        self.state = "init"
        self.n = None
        self.simplex = []
        self.f_simplex = []
        self.x_centroid = None

        self.x_r = None
        self.f_r = None

        self.shrink_idx = 1

    def _start_iteration(self) -> np.ndarray:
        order = np.argsort(self.f_simplex)
        self.simplex = self.simplex[order]
        self.f_simplex = self.f_simplex[order]

        self.x_centroid = np.mean(self.simplex[:-1], axis=0)

        worst_point = self.simplex[-1]
        self.x_r = (1 + self.alpha) * self.x_centroid - self.alpha * worst_point

        self.state = "wait_reflect"
        return self.x_r

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))

        if self.state == "init":
            if self.n is None:
                self.n = obs.x.shape[0]

            self.simplex.append(obs.x.copy())
            self.f_simplex.append(val)

            if len(self.simplex) < self.n + 1:
                idx = len(self.simplex) - 1
                x_next = self.simplex[0].copy()
                x_next[idx] += self.init_step
                return x_next
            else:
                self.simplex = np.array(self.simplex)
                self.f_simplex = np.array(self.f_simplex)
                return self._start_iteration()

        elif self.state == "wait_reflect":
            self.f_r = val
            f_best = self.f_simplex[0]
            f_second_worst = self.f_simplex[-2]
            f_worst = self.f_simplex[-1]

            if self.f_r < f_best:
                self.state = "wait_expand"
                x_e = self.gamma * self.x_r + (1 - self.gamma) * self.x_centroid
                return x_e

            elif f_best <= self.f_r < f_second_worst:
                self.simplex[-1] = self.x_r
                self.f_simplex[-1] = self.f_r
                return self._start_iteration()

            else:
                if self.f_r < f_worst:
                    self.simplex[-1] = self.x_r
                    self.f_simplex[-1] = self.f_r

                self.state = "wait_contract"
                x_c = self.beta * self.simplex[-1] + (1 - self.beta) * self.x_centroid
                return x_c

        elif self.state == "wait_expand":
            f_e = val
            f_best = self.f_simplex[0]

            if f_e < f_best:
                self.simplex[-1] = obs.x
                self.f_simplex[-1] = f_e
            else:
                self.simplex[-1] = self.x_r
                self.f_simplex[-1] = self.f_r

            return self._start_iteration()

        elif self.state == "wait_contract":
            f_c = val
            f_worst = self.f_simplex[-1]

            if f_c < f_worst:
                self.simplex[-1] = obs.x
                self.f_simplex[-1] = f_c
                return self._start_iteration()
            else:
                self.shrink_idx = 1
                self.state = "wait_shrink"
                x_shrink = (self.simplex[self.shrink_idx] + self.simplex[0]) / 2.0
                return x_shrink

        elif self.state == "wait_shrink":
            self.simplex[self.shrink_idx] = obs.x
            self.f_simplex[self.shrink_idx] = val
            self.shrink_idx += 1

            if self.shrink_idx <= self.n:
                x_shrink = (self.simplex[self.shrink_idx] + self.simplex[0]) / 2.0
                return x_shrink
            else:
                return self._start_iteration()


class FiniteDiff2dPlus1(BaseOptimizer):
    """Gradient approximation via finite differences utilizing 2d+1 zero-order oracle queries."""

    def __init__(self, lr=0.01, h=1e-4, name="FiniteDiff(2d+1)"):
        super().__init__(name)
        self.lr = lr
        self.h = h

        self.reset()

    def reset(self) -> None:
        self.state = "base"
        self.dim = None
        self.base_x = None
        self.f_base = None

        self.current_dim = 0
        self.f_plus = None
        self.f_minus = None

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))

        if self.state == "base":
            self.base_x = obs.x.copy()
            self.f_base = val

            if self.dim is None:
                self.dim = self.base_x.shape[0]

            self.current_dim = 0
            self.f_plus = np.zeros(self.dim)
            self.f_minus = np.zeros(self.dim)

            self.state = "wait_plus"
            e = np.zeros(self.dim)
            e[self.current_dim] = self.h
            return self.base_x + e

        elif self.state == "wait_plus":
            self.f_plus[self.current_dim] = val

            self.state = "wait_minus"
            e = np.zeros(self.dim)
            e[self.current_dim] = self.h
            return self.base_x - e

        elif self.state == "wait_minus":
            self.f_minus[self.current_dim] = val

            self.current_dim += 1

            if self.current_dim < self.dim:
                self.state = "wait_plus"
                e = np.zeros(self.dim)
                e[self.current_dim] = self.h
                return self.base_x + e

            else:
                grad_approx = (self.f_plus - self.f_minus) / (2 * self.h)
                x_new = self.base_x - self.lr * grad_approx
                self.state = "base"
                return x_new


class FiniteDiffCentral(BaseOptimizer):
    """Central finite differences optimizer. Strict NFE control requiring 2d queries per step."""

    def __init__(
        self, lr: float = 0.02, h: float = 1e-4, name: str = "FiniteDiffCentral"
    ):
        super().__init__(name)
        self.lr = lr
        self.h = h

        self.reset()

    def reset(self) -> None:
        self.x_base = None
        self.dim = None
        self.query_buffer = []

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))

        if self.x_base is None:
            self.x_base = obs.x.copy()
            self.dim = self.x_base.shape[0]
            self.query_buffer = []

            e1 = np.zeros(self.dim)
            e1[0] = self.h
            return self.x_base + e1

        self.query_buffer.append(val)
        q_len = len(self.query_buffer)

        if q_len == 2 * self.dim:
            grad = np.zeros(self.dim)
            for i in range(self.dim):
                f_plus = self.query_buffer[2 * i]
                f_minus = self.query_buffer[2 * i + 1]
                grad[i] = (f_plus - f_minus) / (2 * self.h)

            x_new = self.x_base - self.lr * grad
            self.x_base = x_new.copy()
            self.query_buffer = []

            e1 = np.zeros(self.dim)
            e1[0] = self.h
            return x_new + e1

        e = np.zeros(self.dim)
        dim_idx = q_len // 2
        sign = 1 if q_len % 2 == 0 else -1
        e[dim_idx] = sign * self.h

        return self.x_base + e


class FiniteDiffForward(BaseOptimizer):
    """Forward finite differences gradient approximation. Minimizes NFE down to d+1 queries per step."""

    def __init__(
        self, lr: float = 0.02, h: float = 1e-4, name: str = "FiniteDiffForward"
    ):
        super().__init__(name)
        self.lr = lr
        self.h = h

        self.reset()

    def reset(self) -> None:
        self.state = "base"
        self.dim = None
        self.x_base = None
        self.f_base = None
        self.f_forward = None
        self.current_dim = 0

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))

        if self.state == "base":
            self.x_base = obs.x.copy()
            self.f_base = val

            if self.dim is None:
                self.dim = self.x_base.shape[0]

            self.f_forward = np.zeros(self.dim)
            self.current_dim = 0
            self.state = "wait_forward"

            e = np.zeros(self.dim)
            e[0] = self.h
            return self.x_base + e

        elif self.state == "wait_forward":
            self.f_forward[self.current_dim] = val
            self.current_dim += 1

            if self.current_dim < self.dim:
                e = np.zeros(self.dim)
                e[self.current_dim] = self.h
                return self.x_base + e

            else:
                grad_approx = (self.f_forward - self.f_base) / self.h
                x_new = self.x_base - self.lr * grad_approx
                self.state = "base"
                return x_new


class ClassicalFDSA(BaseOptimizer):
    """Classical FDSA algorithm (Spall, 1992). Optimized for noisy oracle responses using decaying sequences."""

    def __init__(
        self,
        a: float = 0.1,
        c: float = 0.1,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float = 100.0,
        name: str = "Classical_FDSA",
    ):
        super().__init__(name)
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.A = A

        self.reset()

    def reset(self) -> None:
        self.x_base = None
        self.dim = None
        self.query_buffer = []
        self.k = 1

    def _get_current_params(self):
        a_k = self.a / ((self.k + self.A) ** self.alpha)
        c_k = self.c / (self.k**self.gamma)
        return a_k, c_k

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))
        a_k, c_k = self._get_current_params()

        if self.x_base is None:
            self.x_base = obs.x.copy()
            self.dim = self.x_base.shape[0]
            self.query_buffer = []

            e1 = np.zeros(self.dim)
            e1[0] = c_k
            return self.x_base + e1

        self.query_buffer.append(val)
        q_len = len(self.query_buffer)

        if q_len == 2 * self.dim:
            grad = np.zeros(self.dim)
            for i in range(self.dim):
                f_plus = self.query_buffer[2 * i]
                f_minus = self.query_buffer[2 * i + 1]
                grad[i] = (f_plus - f_minus) / (2 * c_k)

            x_new = self.x_base - a_k * grad
            self.x_base = x_new.copy()
            self.query_buffer = []
            self.k += 1

            _, c_k_new = self._get_current_params()
            e1 = np.zeros(self.dim)
            e1[0] = c_k_new
            return x_new + e1

        e = np.zeros(self.dim)
        dim_idx = q_len // 2
        sign = 1 if q_len % 2 == 0 else -1
        e[dim_idx] = sign * c_k

        return self.x_base + e


class SPSA(BaseOptimizer):
    """Simultaneous Perturbation Stochastic Approximation (SPSA). Scales to high dimensions with exactly 2 NFE per step."""

    def __init__(
        self,
        a: float = 0.1,
        c: float = 0.1,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float = 100.0,
        name: str = "SPSA",
    ):
        super().__init__(name)
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.A = A

        self.reset()

    def reset(self) -> None:
        self.x_base = None
        self.dim = None
        self.k = 1

        self.state = "init"
        self.delta = None
        self.f_plus = None

    def _get_current_params(self):
        a_k = self.a / ((self.k + self.A) ** self.alpha)
        c_k = self.c / (self.k**self.gamma)
        return a_k, c_k

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))
        a_k, c_k = self._get_current_params()

        if self.state == "init":
            self.x_base = obs.x.copy()
            self.dim = self.x_base.shape[0]

            self.delta = np.random.choice([-1.0, 1.0], size=self.dim)
            self.state = "wait_minus"
            return self.x_base + c_k * self.delta

        elif self.state == "wait_minus":
            self.f_plus = val
            self.state = "update"
            return self.x_base - c_k * self.delta

        elif self.state == "update":
            f_minus = val
            grad_approx = (self.f_plus - f_minus) / (2 * c_k) * self.delta
            grad_approx = np.clip(grad_approx, -10.0, 10.0)

            x_new = self.x_base - a_k * grad_approx
            self.x_base = x_new.copy()
            self.k += 1

            _, c_k_new = self._get_current_params()
            self.delta = np.random.choice([-1.0, 1.0], size=self.dim)
            self.state = "wait_minus"
            return self.x_base + c_k_new * self.delta


class OneMeasurementSPSA(BaseOptimizer):
    """One-measurement SPSA with strict budget control. Evaluates the gradient utilizing exactly 1 NFE per iteration."""

    def __init__(
        self,
        a: float = 0.1,
        c: float = 0.1,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float = 100.0,
        name: str = "1-SPSA",
    ):
        super().__init__(name)
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.A = A

        self.reset()

    def reset(self) -> None:
        self.x_base = None
        self.dim = None
        self.k = 1

        self.state = "init"
        self.delta = None
        self.y_prev = None

    def _get_current_params(self):
        a_k = self.a / ((self.k + self.A) ** self.alpha)
        c_k = self.c / (self.k**self.gamma)
        return a_k, c_k

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))
        a_k, c_k = self._get_current_params()

        if self.state == "init":
            self.x_base = obs.x.copy()
            self.dim = self.x_base.shape[0]

            self.y_prev = val
            self.delta = np.random.choice([-1.0, 1.0], size=self.dim)
            self.state = "eval_and_update"
            return self.x_base + c_k * self.delta

        elif self.state == "eval_and_update":
            y_curr = val
            grad_approx = (y_curr - self.y_prev) / c_k * self.delta
            grad_approx = np.clip(grad_approx, -10.0, 10.0)

            x_new = self.x_base - a_k * grad_approx
            self.y_prev = y_curr
            self.x_base = x_new.copy()
            self.k += 1

            _, c_k_new = self._get_current_params()
            self.delta = np.random.choice([-1.0, 1.0], size=self.dim)
            return self.x_base + c_k_new * self.delta


class ZOSignSGD(BaseOptimizer):
    """Zeroth-Order Sign SGD. Provides robustness against extreme outliers in WIND oracle evaluations."""

    def __init__(self, lr: float = 0.01, h: float = 1e-4, name: str = "ZO-SignSGD"):
        super().__init__(name)
        self.lr = lr
        self.h = h

        self.reset()

    def reset(self) -> None:
        self.x_base = None
        self.dim = None
        self.query_buffer = []

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))

        if self.x_base is None:
            self.x_base = obs.x.copy()
            self.dim = self.x_base.shape[0]
            self.query_buffer = []

            e1 = np.zeros(self.dim)
            e1[0] = self.h
            return self.x_base + e1

        self.query_buffer.append(val)
        q_len = len(self.query_buffer)

        if q_len == 2 * self.dim:
            grad = np.zeros(self.dim)
            for i in range(self.dim):
                f_plus = self.query_buffer[2 * i]
                f_minus = self.query_buffer[2 * i + 1]
                grad[i] = (f_plus - f_minus) / (2 * self.h)

            grad_sign = np.sign(grad)
            x_new = self.x_base - self.lr * grad_sign

            self.x_base = x_new.copy()
            self.query_buffer = []

            e1 = np.zeros(self.dim)
            e1[0] = self.h
            return x_new + e1

        e = np.zeros(self.dim)
        dim_idx = q_len // 2
        sign_dir = 1 if q_len % 2 == 0 else -1
        e[dim_idx] = sign_dir * self.h

        return self.x_base + e


class QuadraticInterpolationDPlus1(BaseOptimizer):
    """Zeroth-order optimization via quadratic interpolation. Controls NFE requiring d+1 queries to build parabolas."""

    def __init__(self, h: float = 0.01, name: str = "QuadraticInterpolation(d+1)"):
        super().__init__(name)
        self.h = h
        self.reset()

    def reset(self) -> None:
        self.state = "init"
        self.dim = None

        self.x_prev = None
        self.f_prev = None

        self.x_base = None
        self.f_base = None

        self.f_plus = None
        self.current_dim = 0

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))

        if self.state == "init":
            self.x_prev = obs.x.copy()
            self.f_prev = val
            self.dim = self.x_prev.shape[0]

            self.state = "wait_base"
            return self.x_prev + self.h * np.ones(self.dim)

        elif self.state == "wait_base":
            self.x_base = obs.x.copy()
            self.f_base = val

            self.f_plus = np.zeros(self.dim)
            self.current_dim = 0

            self.state = "wait_plus"
            e = np.zeros(self.dim)
            e[0] = self.h
            return self.x_base + e

        elif self.state == "wait_plus":
            self.f_plus[self.current_dim] = val
            self.current_dim += 1

            if self.current_dim < self.dim:
                e = np.zeros(self.dim)
                e[self.current_dim] = self.h
                return self.x_base + e
            else:
                x_new = np.zeros(self.dim)

                for i in range(self.dim):
                    x1 = self.x_prev[i]
                    f1 = self.f_prev

                    x2 = self.x_base[i]
                    f2 = self.f_base

                    x3 = self.x_base[i] + self.h
                    f3 = self.f_plus[i]

                    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)

                    if abs(denom) < 1e-10:
                        x_new[i] = x2
                        continue

                    a = (x3 * (f2 - f1) + x2 * (f1 - f3) + x1 * (f3 - f2)) / denom
                    b = (
                        x3**2 * (f1 - f2) + x2**2 * (f3 - f1) + x1**2 * (f2 - f3)
                    ) / denom

                    if a > 1e-8:
                        t_opt = -b / (2 * a)
                        t_opt = np.clip(t_opt, x2 - 5 * self.h, x2 + 5 * self.h)
                        x_new[i] = t_opt
                    else:
                        grad = (f3 - f2) / self.h
                        x_new[i] = x2 - self.h * np.sign(grad)

                self.x_prev = self.x_base.copy()
                self.f_prev = self.f_base

                self.state = "wait_base"
                return x_new


class CMA_ES(BaseOptimizer):
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES). Ideal for non-convex functions in black-box optimization."""

    def __init__(
        self, sigma: float = 0.5, popsize: Optional[int] = None, name: str = "CMA-ES"
    ):
        super().__init__(name)
        self.sigma = sigma
        self.popsize_init = popsize
        self.reset()

    def reset(self) -> None:
        self.dim = None
        self.lambda_ = self.popsize_init
        self.m = None
        self.C = None

        self.population = []
        self.fitness = []
        self.current_idx = 0
        self.state = "init"

    def _generate_population(self):
        D, B = np.linalg.eigh(self.C)
        D = np.sqrt(np.maximum(D, 1e-9))

        self.population = []
        for _ in range(self.lambda_):
            z = np.random.standard_normal(self.dim)
            x = self.m + self.sigma * (B @ (D * z))
            self.population.append(x)
        self.fitness = np.zeros(self.lambda_)

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))

        if self.state == "init":
            self.dim = obs.x.shape[0]
            self.m = obs.x.copy()
            if self.lambda_ is None:
                self.lambda_ = 4 + int(3 * np.log(self.dim))

            self.C = np.eye(self.dim)
            self._generate_population()

            self.state = "eval"
            self.current_idx = 0
            return self.population[self.current_idx]

        elif self.state == "eval":
            self.fitness[self.current_idx] = val
            self.current_idx += 1

            if self.current_idx < self.lambda_:
                return self.population[self.current_idx]
            else:
                self._update_distribution()
                self._generate_population()
                self.current_idx = 0
                return self.population[self.current_idx]

    def _update_distribution(self):
        indices = np.argsort(self.fitness)
        mu = self.lambda_ // 2
        best_indices = indices[:mu]

        old_m = self.m.copy()
        parents = np.array([self.population[i] for i in best_indices])
        self.m = np.mean(parents, axis=0)

        diff = parents - old_m
        new_C = (diff.T @ diff) / (mu * (self.sigma**2))

        c1 = 0.1
        self.C = (1 - c1) * self.C + c1 * new_C


class RandomSearch(BaseOptimizer):
    """Randomized Search method. Direct stochastic optimization bypassing gradient approximation with 1 NFE per step."""

    def __init__(self, scale: float = 0.5, name: str = "RandomSearch"):
        super().__init__(name)
        self.scale = scale
        self.reset()

    def reset(self) -> None:
        self.best_x = None
        self.best_val = float("inf")
        self.state = "init"

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))

        if self.state == "init":
            self.best_x = obs.x.copy()
            self.best_val = val
            self.state = "search"
            return self.best_x + np.random.randn(*self.best_x.shape) * self.scale

        elif self.state == "search":
            if val < self.best_val:
                self.best_x = obs.x.copy()
                self.best_val = val
            return self.best_x + np.random.randn(*self.best_x.shape) * self.scale


class GP_UCB(BaseOptimizer):
    """Gaussian Process Upper Confidence Bound (GP-UCB). Surrogate modeling for sample-efficient optimization on WIND."""

    def __init__(self, kappa: float = 2.5, n_initial: int = 5, name: str = "GP-UCB"):
        super().__init__(name)
        self.kappa = kappa
        self.n_initial = n_initial
        self.reset()

    def reset(self) -> None:
        self.X_history = []
        self.y_history = []
        self.dim = None
        self.bounds = None

        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6)
        self.state = "init"

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))

        if self.dim is None:
            self.dim = obs.x.shape[0]
            self.bounds = [(-5.0, 5.0)] * self.dim

        self.X_history.append(obs.x.copy())
        self.y_history.append(val)

        if len(self.X_history) < self.n_initial:
            return np.random.uniform(-5.0, 5.0, size=self.dim)

        X = np.array(self.X_history)
        y = np.array(self.y_history)
        self.gp.fit(X, y)

        def acquisition(x_query):
            x_query = x_query.reshape(1, -1)
            mu, sigma = self.gp.predict(x_query, return_std=True)
            return mu[0] - self.kappa * sigma[0]

        best_val = float("inf")
        next_x = None

        for _ in range(3):
            x_start = np.random.uniform(-5.0, 5.0, size=self.dim)
            res = minimize(acquisition, x_start, bounds=self.bounds, method="L-BFGS-B")
            if res.fun < best_val:
                best_val = res.fun
                next_x = res.x

        return next_x


class UHCMAES(BaseOptimizer):
    """Uncertainty Handling CMA-ES. Employs sampling techniques to compensate for high noise in oracle responses."""

    def __init__(self, dim, popsize=None, initial_step=0.1, name="UH-CMA-ES"):
        super().__init__(name)
        self.dim = dim

        self.lambda_ = popsize if popsize else max(4 + int(3 * np.log(dim)), 5)
        self.mu = self.lambda_ // 2

        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)
        self.mu_eff = 1.0 / np.sum(self.weights**2)

        self.c_c = 4.0 / (dim + 4.0)
        self.c_sigma = (self.mu_eff + 2) / (dim + self.mu_eff + 3)
        self.c_1 = 2.0 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c_1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((dim + 2) ** 2 + self.mu_eff),
        )
        self.d_sigma = (
            1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (dim + 1)) - 1) + self.c_sigma
        )

        self.initial_step = initial_step
        self.reset()

    def reset(self) -> None:
        self.m = None
        self.sigma = self.initial_step
        self.C = np.eye(self.dim)
        self.p_c = np.zeros(self.dim)
        self.p_sigma = np.zeros(self.dim)

        self.n_eval = 1.0
        self.alpha = 1.5
        self.lambda_reev = max(self.lambda_ // 10, 2)

        self.offspring = []
        self.f_values = np.zeros(self.lambda_)
        self.f_reev = np.zeros(self.lambda_)

        self.state = "init"
        self.current_idx = 0
        self.evals_done = 0
        self.current_f_sum = 0.0

    def _generate_population(self):
        self.offspring = []
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        D, B = np.linalg.eigh(self.C)
        D = np.maximum(D, 1e-8)
        std_matrix = B @ np.diag(np.sqrt(D))

        for _ in range(self.lambda_):
            z = np.random.randn(self.dim)
            x = self.m + self.sigma * (std_matrix @ z)
            self.offspring.append(x)

    def _update_cma_es(self):
        order = np.argsort(self.f_values)
        x_sorted = np.array(self.offspring)[order]

        x_old = self.m.copy()
        self.m = np.sum(x_sorted[: self.mu] * self.weights[:, np.newaxis], axis=0)

        D, B = np.linalg.eigh(self.C)
        D = np.maximum(D, 1e-8)
        invsqrtC = B @ np.diag(1.0 / np.sqrt(D)) @ B.T

        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(
            self.c_sigma * (2 - self.c_sigma) * self.mu_eff
        ) * (invsqrtC @ (self.m - x_old)) / self.sigma

        self.p_c = (1 - self.c_c) * self.p_c + np.sqrt(
            self.c_c * (2 - self.c_c) * self.mu_eff
        ) * (self.m - x_old) / self.sigma

        rank1_update = np.outer(self.p_c, self.p_c)

        y = (x_sorted[: self.mu] - x_old) / self.sigma
        rankmu_update = np.zeros_like(self.C)
        for i in range(self.mu):
            rankmu_update += self.weights[i] * np.outer(y[i], y[i])

        self.C = (
            (1 - self.c_c - self.c_mu) * self.C
            + self.c_c * rank1_update
            + self.c_mu * rankmu_update
        )

        expected_norm = np.sqrt(self.dim) * (
            1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2)
        )
        self.sigma = self.sigma * np.exp(
            (self.c_sigma / self.d_sigma)
            * (np.linalg.norm(self.p_sigma) / expected_norm - 1)
        )

    def _uncertainty_handling(self):
        rank_changes = 0
        for i in range(self.lambda_reev):
            diff = abs(self.f_values[i] - self.f_reev[i])
            if diff > self.sigma * 0.1:
                rank_changes += 1

        if rank_changes > self.lambda_reev / 2.0:
            self.n_eval = min(self.n_eval * self.alpha, 100.0)
        else:
            self.n_eval = max(self.n_eval / self.alpha, 1.0)

        for i in range(self.lambda_reev):
            self.f_values[i] = (self.f_values[i] + self.f_reev[i]) / 2.0

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))

        if self.state == "init":
            self.m = obs.x.copy()
            self._generate_population()

            self.state = "eval_main"
            self.current_idx = 0
            self.evals_done = 0
            self.current_f_sum = 0.0
            return self.offspring[self.current_idx]

        elif self.state == "eval_main":
            self.current_f_sum += val
            self.evals_done += 1

            n_eval_target = max(1, int(np.round(self.n_eval)))

            if self.evals_done < n_eval_target:
                return self.offspring[self.current_idx]

            self.f_values[self.current_idx] = self.current_f_sum / n_eval_target

            self.current_idx += 1
            self.evals_done = 0
            self.current_f_sum = 0.0

            if self.current_idx < self.lambda_:
                return self.offspring[self.current_idx]
            else:
                self.state = "eval_reev"
                self.current_idx = 0
                return self.offspring[self.current_idx]

        elif self.state == "eval_reev":
            self.current_f_sum += val
            self.evals_done += 1

            n_eval_target = max(1, int(np.round(self.n_eval)))

            if self.evals_done < n_eval_target:
                return self.offspring[self.current_idx]

            self.f_reev[self.current_idx] = self.current_f_sum / n_eval_target

            self.current_idx += 1
            self.evals_done = 0
            self.current_f_sum = 0.0

            if self.current_idx < self.lambda_reev:
                return self.offspring[self.current_idx]
            else:
                self._uncertainty_handling()
                self._update_cma_es()

                self._generate_population()
                self.state = "eval_main"
                self.current_idx = 0

                return self.offspring[self.current_idx]


class SANE(BaseOptimizer):
    """Simulated Annealing in Noisy Environments. Features strict NFE bounds and temperature thresholds."""

    def __init__(
        self,
        dim,
        sigma_E,
        T0=50.0,
        alpha=0.999,
        step_size=0.1,
        max_samples=100,
        name="SANE",
    ):
        super().__init__(name)
        self.dim = dim
        self.sigma_E = sigma_E
        self.sigma_delta_E = np.sqrt(2) * sigma_E

        self.T0 = T0
        self.alpha = alpha
        self.step_size = step_size
        self.max_samples = max_samples
        self.reset()

    def reset(self):
        self.T = self.T0
        self.x_c = None
        self.f_c = None
        self.x_n = None

        self.state = "init"
        self.n = 0
        self.current_f_sum = 0.0

    def _generate_neighbor(self):
        cooling_factor = max(self.T / self.T0, 1e-8)
        current_step = self.step_size * cooling_factor
        return self.x_c + np.random.randn(self.dim) * current_step

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))

        if self.state == "init":
            self.x_c = obs.x.copy()
            self.state = "wait_init"
            return self.x_c

        elif self.state == "wait_init":
            self.f_c = val
            self.x_n = self._generate_neighbor()
            self.n = 0
            self.current_f_sum = 0.0
            self.state = "eval_candidate"
            return self.x_n

        elif self.state == "eval_candidate":
            self.n += 1
            self.current_f_sum += val

            temp_threshold = self.sigma_delta_E * np.sqrt(np.pi / 8.0)

            if self.T >= temp_threshold:
                delta_hat = self.current_f_sum - self.f_c

                threshold = -0.5 * (self.sigma_delta_E**2) / self.T
                if delta_hat <= threshold:
                    p_accept = 1.0
                else:
                    exponent = -(
                        delta_hat / self.T + 0.5 * (self.sigma_delta_E**2) / (self.T**2)
                    )
                    p_accept = np.exp(exponent)

                if np.random.rand() < p_accept:
                    self.x_c = self.x_n.copy()
                    self.f_c = self.current_f_sum

                self.T *= self.alpha
                self.x_n = self._generate_neighbor()
                self.n = 0
                self.current_f_sum = 0.0
                return self.x_n

            else:
                f_n_mean = self.current_f_sum / self.n
                delta_hat = f_n_mean - self.f_c

                p_err = norm.cdf(
                    -np.abs(delta_hat) * np.sqrt(self.n) / self.sigma_delta_E
                )
                p_glauber = 1.0 / (1.0 + np.exp(np.abs(delta_hat) / self.T))

                if p_err > p_glauber and self.n < self.max_samples:
                    return self.x_n

                if delta_hat < 0:
                    self.x_c = self.x_n.copy()
                    self.f_c = f_n_mean

                self.T *= self.alpha
                self.x_n = self._generate_neighbor()
                self.n = 0
                self.current_f_sum = 0.0
                return self.x_n


class REMBO(BaseOptimizer):
    """Random Embedding Bayesian Optimization. Projects high dimensional inputs to low dimensions for sample-efficient tuning."""

    def __init__(self, D, d, bounds_D=(-1.0, 1.0), bounds_d=(-2.0, 2.0), name="REMBO"):
        super().__init__(name)
        self.D = D
        self.d = d
        self.bounds_D = bounds_D
        self.bounds_d = bounds_d

        self.A = np.random.randn(D, d)
        self.A_pinv = np.linalg.pinv(self.A)
        self.reset()

    def reset(self):
        self.state = "init"
        self.Y = []
        self.F = []
        self.last_y = None

        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(self.d), nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            alpha=1e-4,
        )

    def _project(self, y):
        x = self.A @ y
        return np.clip(x, self.bounds_D[0], self.bounds_D[1])

    def _expected_improvement(self, y):
        y = y.reshape(1, -1)
        mu, sigma = self.gp.predict(y, return_std=True)

        if sigma == 0.0:
            return 0.0

        f_best = np.min(self.F)

        Z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)

        return -ei[0]

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))

        if self.state == "init":
            y_0 = self.A_pinv @ obs.x
            self.Y.append(y_0)
            self.F.append(val)
            self.state = "bo_loop"
        else:
            self.Y.append(self.last_y)
            self.F.append(val)

        if len(self.Y) < max(5, self.d + 1):
            y_next = np.random.uniform(self.bounds_d[0], self.bounds_d[1], self.d)
            self.last_y = y_next
            return self._project(y_next)

        y_train = np.array(self.Y)
        f_train = np.array(self.F)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(y_train, f_train)

        best_y = None
        best_ei = float("inf")

        for _ in range(5):
            y_start = np.random.uniform(self.bounds_d[0], self.bounds_d[1], self.d)
            res = minimize(
                self._expected_improvement,
                y_start,
                bounds=[self.bounds_d] * self.d,
                method="L-BFGS-B",
            )
            if res.fun < best_ei:
                best_ei = res.fun
                best_y = res.x

        self.last_y = best_y
        return self._project(best_y)


class ZO_AdaMM(BaseOptimizer):
    """Zeroth-Order AdaMM. Adapts Adam-like momentum and scaling for stable black-box convergence."""

    def __init__(
        self, dim, alpha=0.1, beta1=0.9, beta2=0.99, mu=0.001, name="ZO-AdaMM"
    ):
        super().__init__(name)
        self.dim = dim

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.mu = mu

        self.reset()

    def reset(self):
        self.state = "init"

        self.m = np.zeros(self.dim)
        self.v = np.zeros(self.dim)
        self.v_hat = np.zeros(self.dim)

        self.x_c = None
        self.f_c = None
        self.u = None

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))

        if self.state == "init":
            self.x_c = obs.x.copy()
            self.f_c = val

            u = np.random.randn(self.dim)
            self.u = u / np.linalg.norm(u)

            self.state = "eval_shift"
            x_shift = self.x_c + self.mu * self.u
            return x_shift

        elif self.state == "eval_shift":
            f_shift = val

            g_hat = (self.dim / self.mu) * (f_shift - self.f_c) * self.u

            self.m = self.beta1 * self.m + (1 - self.beta1) * g_hat
            self.v = self.beta2 * self.v + (1 - self.beta2) * (g_hat**2)
            self.v_hat = np.maximum(self.v_hat, self.v)

            self.x_c = self.x_c - self.alpha * self.m / (np.sqrt(self.v_hat) + 1e-8)

            self.state = "eval_base"
            return self.x_c

        elif self.state == "eval_base":
            self.f_c = val

            u = np.random.randn(self.dim)
            self.u = u / np.linalg.norm(u)

            self.state = "eval_shift"
            x_shift = self.x_c + self.mu * self.u
            return x_shift


class ZO_SGD(BaseOptimizer):
    """Zeroth-Order SGD. Implements straightforward gradient approximation moving strictly against it without momentum."""

    def __init__(self, dim, alpha=0.1, mu=0.001, name="ZO-SGD"):
        super().__init__(name)
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.reset()

    def reset(self):
        self.state = "init"
        self.x_c = None
        self.f_c = None
        self.u = None

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))

        if self.state == "init":
            self.x_c = obs.x.copy()
            self.f_c = val

            u = np.random.randn(self.dim)
            self.u = u / np.linalg.norm(u)

            self.state = "eval_shift"
            return self.x_c + self.mu * self.u

        elif self.state == "eval_shift":
            f_shift = val

            g_hat = (self.dim / self.mu) * (f_shift - self.f_c) * self.u

            self.x_c = self.x_c - self.alpha * g_hat

            self.state = "eval_base"
            return self.x_c

        elif self.state == "eval_base":
            self.f_c = val

            u = np.random.randn(self.dim)
            self.u = u / np.linalg.norm(u)

            self.state = "eval_shift"
            return self.x_c + self.mu * self.u


class MeZO(BaseOptimizer):
    """Memory-Efficient Zeroth-Order Optimizer. Reconstructs sampling vectors dynamically via seeded random generation."""

    def __init__(self, dim, alpha=1e-3, mu=1e-3, name="MeZO"):
        super().__init__(name)
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.reset()

    def reset(self):
        self.state = "start"
        self.x_c = None
        self.seed = None
        self.f_plus = None

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))

        if self.state == "start":
            self.x_c = obs.x.copy()
            self.seed = np.random.randint(0, 2**32 - 1)

            rng = np.random.default_rng(self.seed)
            z = rng.standard_normal(self.dim)

            self.state = "eval_plus"
            return self.x_c + self.mu * z

        elif self.state == "eval_plus":
            self.f_plus = val

            rng = np.random.default_rng(self.seed)
            z = rng.standard_normal(self.dim)

            self.state = "eval_minus"
            return self.x_c - self.mu * z

        elif self.state == "eval_minus":
            f_minus = val

            rng = np.random.default_rng(self.seed)
            z = rng.standard_normal(self.dim)

            projected_grad = ((self.f_plus - f_minus) / (2 * self.mu)) * z
            self.x_c = self.x_c - self.alpha * projected_grad

            self.seed = np.random.randint(0, 2**32 - 1)
            rng = np.random.default_rng(self.seed)
            z_new = rng.standard_normal(self.dim)

            self.state = "eval_plus"
            return self.x_c + self.mu * z_new


class AdaptiveFD_BFGS(BaseOptimizer):
    """Adaptive Finite-Difference L-BFGS. Estimates noise via Hamming tables and automatically adjusts finite difference steps."""

    def __init__(
        self,
        lr: float = 0.1,
        q: int = 4,
        delta: float = 1e-3,
        nu2: float = 1.0,
        name: str = "AdaptiveFD-BFGS",
    ):
        super().__init__(name)
        self.lr_init = lr
        self.lr = lr
        self.q = q
        self.delta = delta
        self.nu2 = nu2
        self.reset()

    def reset(self) -> None:
        self.state = "init"
        self.dim = None
        self.x_base = None
        self.f_base = None

        self.H = None
        self.x_prev = None
        self.grad_prev = None
        self.grad = None
        self.x_next = None

        self.noise_idx = 0
        self.noise_vals = []
        self.v = None
        self.h = 1e-4

        self.grad_idx = 0
        self.f_plus = []

    def _compute_noise_level(self) -> float:
        T = np.zeros((self.q + 1, self.q + 1))
        T[:, 0] = self.noise_vals

        for j in range(1, self.q + 1):
            for i in range(self.q + 1 - j):
                T[i, j] = T[i + 1, j - 1] - T[i, j - 1]

        gamma_q = (math.factorial(self.q) ** 2) / math.factorial(2 * self.q)

        s_q = np.sqrt(gamma_q * (T[0, self.q] ** 2))
        return s_q

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))

        if self.state == "init":
            self.x_base = obs.x.copy()
            self.f_base = val
            self.dim = self.x_base.shape[0]
            self.H = np.eye(self.dim)

            self.v = np.random.randn(self.dim)
            self.v /= np.linalg.norm(self.v)
            self.noise_idx = 0
            self.noise_vals = []

            self.state = "noise_probe"
            return self.x_base + (-self.q / 2 + self.noise_idx) * self.delta * self.v

        elif self.state == "noise_probe":
            self.noise_vals.append(val)
            self.noise_idx += 1

            if self.noise_idx <= self.q:
                return (
                    self.x_base + (-self.q / 2 + self.noise_idx) * self.delta * self.v
                )
            else:
                noise_level = self._compute_noise_level()

                self.h = (8**0.25) * np.sqrt(noise_level / self.nu2)
                self.h = np.clip(self.h, 1e-8, 1.0)

                self.grad_idx = 0
                self.f_plus = np.zeros(self.dim)
                self.state = "grad_eval"

                e = np.zeros(self.dim)
                e[0] = self.h
                return self.x_base + e

        elif self.state == "grad_eval":
            self.f_plus[self.grad_idx] = val
            self.grad_idx += 1

            if self.grad_idx < self.dim:
                e = np.zeros(self.dim)
                e[self.grad_idx] = self.h
                return self.x_base + e
            else:
                self.grad = (self.f_plus - self.f_base) / self.h

                if self.x_prev is not None and self.grad_prev is not None:
                    s = self.x_base - self.x_prev
                    y = self.grad - self.grad_prev
                    rho_inv = np.dot(y, s)

                    if rho_inv > 1e-8 * np.linalg.norm(s) * np.linalg.norm(y):
                        rho = 1.0 / rho_inv
                        I = np.eye(self.dim)
                        A = I - rho * np.outer(s, y)
                        B = I - rho * np.outer(y, s)
                        self.H = A @ self.H @ B + rho * np.outer(s, s)

                p = -self.H @ self.grad
                self.x_next = self.x_base + self.lr * p

                self.state = "check_step"
                return self.x_next

        elif self.state == "check_step":
            expected_drop = 1e-4 * self.lr * np.dot(self.grad, -self.H @ self.grad)

            if val <= self.f_base + expected_drop + 1e-6:
                self.x_prev = self.x_base.copy()
                self.grad_prev = self.grad.copy()

                self.x_base = self.x_next.copy()
                self.f_base = val

                self.lr = min(self.lr_init * 2.0, self.lr * 1.2)

                self.grad_idx = 0
                self.f_plus = np.zeros(self.dim)
                self.state = "grad_eval"

                e = np.zeros(self.dim)
                e[0] = self.h
                return self.x_base + e
            else:
                self.lr *= 0.5

                self.v = np.random.randn(self.dim)
                self.v /= np.linalg.norm(self.v)
                self.noise_idx = 0
                self.noise_vals = []

                self.state = "noise_probe"
                return (
                    self.x_base + (-self.q / 2 + self.noise_idx) * self.delta * self.v
                )


class AdaZORO(BaseOptimizer):
    """Adaptive Zeroth-Order Regularized Optimization. Recovers sparse gradients leveraging CoSaMP from fewer oracle queries."""

    def __init__(
        self,
        lr: float = 0.1,
        delta: float = 1e-3,
        s_init: int = 2,
        phi: float = 0.1,
        b1: float = 2.0,
        name: str = "AdaZORO",
    ):
        super().__init__(name)
        self.lr = lr
        self.delta = delta
        self.s_init = s_init
        self.phi = phi
        self.b1 = b1
        self.reset()

    def reset(self) -> None:
        self.state = "init"
        self.x_c = None
        self.f_base = None
        self.dim = None

        self.s = self.s_init
        self.m_target = None
        self.query_idx = 0
        self.Z_history = []
        self.y_history = []
        self.current_z = None

    def _calc_m(self, s: int) -> int:
        log_term = np.log(max(self.dim / s, 1.1))
        m = int(self.b1 * s * log_term)
        return min(self.dim, max(s + 2, m))

    def _cosamp(
        self, Z: np.ndarray, y: np.ndarray, s: int, max_iter: int = 10
    ) -> np.ndarray:
        g_hat = np.zeros(self.dim)
        v = y.copy()

        for _ in range(max_iter):
            u = Z.T @ v
            omega_add = np.argsort(np.abs(u))[-2 * s :]
            omega = np.union1d(np.nonzero(g_hat)[0], omega_add)

            Z_omega = Z[:, omega]
            try:
                b_omega, _, _, _ = np.linalg.lstsq(Z_omega, y, rcond=None)
            except np.linalg.LinAlgError:
                b_omega = np.zeros(len(omega))

            b = np.zeros(self.dim)
            b[omega] = b_omega

            g_hat = np.zeros(self.dim)
            top_s = np.argsort(np.abs(b))[-s:]
            g_hat[top_s] = b[top_s]

            v = y - Z @ g_hat
            if np.linalg.norm(v) < 1e-8:
                break

        return g_hat

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))

        if self.state == "init":
            self.x_c = obs.x.copy()
            self.dim = self.x_c.shape[0]
            self.state = "wait_base"
            return self.x_c

        elif self.state == "wait_base":
            self.f_base = val
            self.s = self.s_init
            self.m_target = self._calc_m(self.s)
            self.Z_history = []
            self.y_history = []
            self.query_idx = 0

            self.state = "wait_probe"
            self.current_z = np.random.choice([-1.0, 1.0], size=self.dim)
            return self.x_c + self.delta * self.current_z

        elif self.state == "wait_probe":
            y_i = (val - self.f_base) / self.delta

            self.Z_history.append(self.current_z.copy())
            self.y_history.append(y_i)
            self.query_idx += 1

            if self.query_idx < self.m_target:
                self.current_z = np.random.choice([-1.0, 1.0], size=self.dim)
                return self.x_c + self.delta * self.current_z
            else:
                Z_mat = np.array(self.Z_history)
                y_vec = np.array(self.y_history)

                scale = 1.0 / np.sqrt(self.query_idx)
                Z_scaled = Z_mat * scale
                y_scaled = y_vec * scale

                g_hat = self._cosamp(Z_scaled, y_scaled, self.s)

                residual = np.linalg.norm(Z_scaled @ g_hat - y_scaled)
                y_norm = np.linalg.norm(y_scaled) + 1e-8
                rel_residual = residual / y_norm

                if rel_residual > self.phi and self.m_target < self.dim:
                    self.s += 1
                    next_m = self._calc_m(self.s)

                    if next_m > self.m_target:
                        self.m_target = next_m
                        self.current_z = np.random.choice([-1.0, 1.0], size=self.dim)
                        return self.x_c + self.delta * self.current_z

                self.x_c = self.x_c - self.lr * g_hat
                self.state = "wait_base"
                return self.x_c


class ZOSPIDER_ADMM(BaseOptimizer):
    """Zeroth-Order SPIDER ADMM. Hybrid sampling strategy combining variance reduction (SPIDER) and L1-regularization."""

    def __init__(
        self,
        lr: float = 0.01,
        rho: float = 1.0,
        l1_ratio: float = 0.1,
        q: int = 5,
        mu: float = 1e-3,
        nu: float = 1e-3,
        name: str = "ZO-SPIDER-ADMM",
    ):
        super().__init__(name)
        self.lr = lr
        self.rho = rho
        self.l1_ratio = l1_ratio
        self.q = q
        self.mu = mu
        self.nu = nu
        self.reset()

    def reset(self) -> None:
        self.state = "init"
        self.dim = None

        self.x_k = None
        self.y_k = None
        self.lambda_k = None

        self.v_k = None
        self.x_prev = None
        self.k = 0

        self.cooge_i = 0
        self.cooge_grad = None
        self.f_plus = None

        self.u = None
        self.f_xk_plus = None
        self.f_xk = None
        self.f_xprev_plus = None

    def _do_admm_update(self):
        self.x_prev = self.x_k.copy()

        z = self.x_k - self.lambda_k / self.rho
        threshold = self.l1_ratio / self.rho
        self.y_k = np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)

        grad_aug = self.v_k - self.lambda_k + self.rho * (self.x_k - self.y_k)
        self.x_k = self.x_k - self.lr * grad_aug

        self.lambda_k = self.lambda_k - self.rho * (self.x_k - self.y_k)

    def _start_next_iter(self) -> np.ndarray:
        if self.k % self.q == 0:
            self.state = "cooge_plus"
            self.cooge_i = 0
            self.cooge_grad = np.zeros(self.dim)

            e = np.zeros(self.dim)
            e[0] = self.mu
            return self.x_k + e
        else:
            self.state = "unige_0"
            self.u = np.random.randn(self.dim)
            self.u /= np.linalg.norm(self.u)
            return self.x_k + self.nu * self.u

    def step(self, obs) -> np.ndarray:
        val = float(np.squeeze(obs.value))

        if self.state == "init":
            self.x_k = obs.x.copy()
            self.dim = len(self.x_k)
            self.y_k = self.x_k.copy()
            self.lambda_k = np.zeros(self.dim)
            self.v_k = np.zeros(self.dim)
            self.x_prev = self.x_k.copy()
            self.k = 0
            return self._start_next_iter()

        elif self.state == "cooge_plus":
            self.f_plus = val
            self.state = "cooge_minus"
            e = np.zeros(self.dim)
            e[self.cooge_i] = self.mu
            return self.x_k - e

        elif self.state == "cooge_minus":
            f_minus = val
            self.cooge_grad[self.cooge_i] = (self.f_plus - f_minus) / (2 * self.mu)

            self.cooge_i += 1
            if self.cooge_i < self.dim:
                self.state = "cooge_plus"
                e = np.zeros(self.dim)
                e[self.cooge_i] = self.mu
                return self.x_k + e
            else:
                self.v_k = self.cooge_grad.copy()
                self._do_admm_update()
                self.k += 1
                return self._start_next_iter()

        elif self.state == "unige_0":
            self.f_xk_plus = val
            self.state = "unige_1"
            return self.x_k

        elif self.state == "unige_1":
            self.f_xk = val
            self.state = "unige_2"
            return self.x_prev + self.nu * self.u

        elif self.state == "unige_2":
            self.f_xprev_plus = val
            self.state = "unige_3"
            return self.x_prev

        elif self.state == "unige_3":
            f_xprev = val
            grad_xk = (self.dim / self.nu) * (self.f_xk_plus - self.f_xk) * self.u
            grad_xprev = (self.dim / self.nu) * (self.f_xprev_plus - f_xprev) * self.u

            self.v_k = grad_xk - grad_xprev + self.v_k
            self._do_admm_update()
            self.k += 1
            return self._start_next_iter()
