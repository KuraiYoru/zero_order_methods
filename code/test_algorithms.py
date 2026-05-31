import pytest
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

from zero_order_algorithms import (
    NelderMead,
    FiniteDiff2dPlus1,
    FiniteDiffCentral,
    FiniteDiffForward,
    ClassicalFDSA,
    SPSA,
    OneMeasurementSPSA,
    ZOSignSGD,
    QuadraticInterpolationDPlus1,
    CMA_ES,
    RandomSearch,
    GP_UCB,
    UHCMAES,
    SANE,
    REMBO,
    ZO_AdaMM,
    ZO_SGD,
    MeZO,
    AdaptiveFD_BFGS,
    AdaZORO,
    ZOSPIDER_ADMM,
)


class MockObservation:
    """Mock observation object representing the response from the WIND benchmark oracle."""

    def __init__(self, x: np.ndarray, value: float):
        self.x = x
        self.value = value


class MockWindOracle:
    """Mock oracle simulating the WIND benchmark with a noisy sphere function and strict NFE tracking."""

    def __init__(self):
        self.nfe = 0

    def __call__(self, x: np.ndarray) -> MockObservation:
        self.nfe += 1
        noise = np.random.normal(0, 1e-3)
        value = float(np.sum(x**2) + noise)
        return MockObservation(x.copy(), value)


@pytest.fixture
def wind_oracle():
    """Provides a fresh instance of the simulated WIND oracle for each test execution."""
    return MockWindOracle()


AUTO_DIM_ALGOS = [
    NelderMead,
    FiniteDiff2dPlus1,
    FiniteDiffCentral,
    FiniteDiffForward,
    ClassicalFDSA,
    SPSA,
    OneMeasurementSPSA,
    ZOSignSGD,
    QuadraticInterpolationDPlus1,
    CMA_ES,
    RandomSearch,
    GP_UCB,
    AdaptiveFD_BFGS,
    AdaZORO,
    ZOSPIDER_ADMM,
]


@pytest.mark.parametrize("AlgoClass", AUTO_DIM_ALGOS)
def test_auto_dim_algorithms(AlgoClass, wind_oracle):
    """Verifies initialization, execution loop stability, array dimensions, and state reset."""
    opt = AlgoClass()
    start_x = np.array([1.5, -2.0], dtype=float)
    obs = wind_oracle(start_x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for _ in range(25):
            next_x = opt.step(obs)
            assert isinstance(
                next_x, np.ndarray
            ), f"{AlgoClass.__name__} returned non-array"
            assert (
                next_x.shape == start_x.shape
            ), f"{AlgoClass.__name__} altered array shape"
            obs = wind_oracle(next_x)

    assert wind_oracle.nfe > 0

    opt.reset()


EXPLICIT_DIM_ALGOS = [ZO_AdaMM, ZO_SGD, MeZO]


@pytest.mark.parametrize("AlgoClass", EXPLICIT_DIM_ALGOS)
def test_explicit_dim_algorithms(AlgoClass, wind_oracle):
    """Verifies execution for zeroth-order algorithms requiring explicit dimensions."""
    dim = 3
    opt = AlgoClass(dim=dim)
    obs = wind_oracle(np.array([1.0, 2.0, 3.0], dtype=float))

    for _ in range(25):
        next_x = opt.step(obs)
        assert isinstance(next_x, np.ndarray)
        assert next_x.shape == (dim,)
        obs = wind_oracle(next_x)

    assert wind_oracle.nfe > 0


def test_uhcmaes_algorithm(wind_oracle):
    """Tests the UH-CMA-ES algorithm evaluating its internal population mechanics."""
    dim = 2
    opt = UHCMAES(dim=dim, popsize=6)
    obs = wind_oracle(np.array([1.0, -1.0], dtype=float))

    for _ in range(40):
        next_x = opt.step(obs)
        assert isinstance(next_x, np.ndarray)
        obs = wind_oracle(next_x)

    assert wind_oracle.nfe > 20
    opt.reset()
    assert len(opt.offspring) == 0


def test_sane_algorithm(wind_oracle):
    """Tests Simulated Annealing in Noisy Environments verifying functional oracle interactions."""
    dim = 2
    opt = SANE(dim=dim, sigma_E=0.001)
    obs = wind_oracle(np.array([2.0, 2.0], dtype=float))

    initial_T = opt.T
    for _ in range(30):
        next_x = opt.step(obs)
        assert isinstance(next_x, np.ndarray)
        obs = wind_oracle(next_x)

    assert opt.T < initial_T


def test_rembo_algorithm(wind_oracle):
    """Tests the REMBO algorithm space projections securely bypassing internal surrogate modeling crashes."""
    D = 10
    d = 2
    opt = REMBO(D=D, d=d)
    obs = wind_oracle(np.ones(D, dtype=float))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for _ in range(20):
            next_x = opt.step(obs)
            assert isinstance(next_x, np.ndarray)
            assert next_x.shape == (D,)
            obs = wind_oracle(next_x)

    assert wind_oracle.nfe > 0
