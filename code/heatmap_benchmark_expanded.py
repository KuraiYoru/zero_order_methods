import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
from pathlib import Path
import inspect

current_dir = Path(__file__).parent.absolute()
wind_core_path = current_dir / "WIND" / "KDD_paper_first_submission"
sys.path.insert(0, str(wind_core_path))

from core import (
    DynamicEnvironment,
    StationaryDrift,
    LinearDrift,
    JumpDrift,
    QuadraticLandscape,
    PNormLandscape,
    RosenbrockLandscape,
    MultiExtremalLandscape,
)
from oracle import ZeroOrderOracle

from zero_order_algorithms import (
    NelderMead,
    SPSA,
    OneMeasurementSPSA,
    CMA_ES,
    UHCMAES,
    RandomSearch,
    SANE,
    ZOSignSGD,
    FiniteDiffCentral,
    ClassicalFDSA,
    QuadraticInterpolationDPlus1,
    ZO_AdaMM,
    ZO_SGD,
    MeZO,
    AdaptiveFD_BFGS,
)


def run_algo_for_heatmap(algo_class, env, max_nfe, dim, seed=42):
    """
    Run an optimization algorithm through the environment and return its convergence history.
    Uses smart initialization to handle algorithm-specific hyperparameter requirements.
    Enforces the WIND protocol: start_step -> query -> end_step -> env.step.
    """
    np.random.seed(seed)
    env.reset()
    oracle = ZeroOrderOracle(environment=env, seed=seed)

    try:
        sig = inspect.signature(algo_class.__init__)
        kwargs = {}
        for name, param in sig.parameters.items():
            if name in ("self", "args", "kwargs"):
                continue
            if param.default == inspect.Parameter.empty:
                if name == "dim":
                    kwargs["dim"] = dim
                elif name == "sigma_E":
                    kwargs["sigma_E"] = 0.1
                elif name == "sigma":
                    kwargs["sigma"] = 0.1
                elif name == "lr":
                    kwargs["lr"] = 0.01
                elif name == "mu":
                    kwargs["mu"] = 0.01

        opt = algo_class(**kwargs)
    except Exception:
        try:
            opt = algo_class(dim=dim)
        except TypeError:
            opt = algo_class()

    x_curr = np.random.uniform(-2, 2, size=dim)

    best_val = float("inf")
    history_nfe = []
    history_val = []

    t = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while oracle.n_queries < max_nfe:
            try:
                oracle.start_step(t)
                obs = oracle.query(x_curr)
                oracle.end_step()

                val = float(np.squeeze(obs.value))
                if best_val == float("inf"):
                    best_val = val
                else:
                    best_val = min(best_val, val)

                history_nfe.append(oracle.n_queries)
                history_val.append(best_val)

                env.step()

                x_curr = opt.step(obs)
                if x_curr is None:
                    break

                t += 1
            except Exception:
                break

    if not history_nfe or history_nfe[-1] < max_nfe:
        history_nfe.append(max_nfe)
        history_val.append(best_val if best_val != float("inf") else 1.0)

    history_nfe, unique_idx = np.unique(history_nfe, return_index=True)
    history_val = np.array(history_val)[unique_idx]

    return history_nfe, history_val


def generate_expanded_heatmap():
    """
    Generate an expanded heatmap comparing various algorithms across stress-test environments.
    """
    DIM = 5
    MAX_NFE = 500
    NFE_MILESTONES = [50, 100, 250, 500]
    SEED = 42

    environments = {
        "1. Baseline (Convex)": DynamicEnvironment(
            dim=DIM,
            drift=StationaryDrift(),
            landscape=QuadraticLandscape(dim=DIM, condition_number=1.0),
        ),
        "2. Rosenbrock (Narrow Valley)": DynamicEnvironment(
            dim=DIM, drift=StationaryDrift(), landscape=RosenbrockLandscape()
        ),
        "3. Multi-Extremal (Local Minima)": DynamicEnvironment(
            dim=DIM, drift=StationaryDrift(), landscape=MultiExtremalLandscape()
        ),
        "4. Non-Smooth (L1-Norm)": DynamicEnvironment(
            dim=DIM, drift=StationaryDrift(), landscape=PNormLandscape(p=1.0)
        ),
        "5. Linear Drift (Moving Target)": DynamicEnvironment(
            dim=DIM,
            drift=LinearDrift(velocity=np.ones(DIM) * 0.05),
            landscape=QuadraticLandscape(dim=DIM),
        ),
    }

    algorithms = [
        NelderMead,
        CMA_ES,
        UHCMAES,
        SPSA,
        OneMeasurementSPSA,
        FiniteDiffCentral,
        ClassicalFDSA,
        AdaptiveFD_BFGS,
        ZOSignSGD,
        ZO_AdaMM,
        ZO_SGD,
        MeZO,
        QuadraticInterpolationDPlus1,
        SANE,
        RandomSearch,
    ]

    fig, axes = plt.subplots(3, 2, figsize=(18, 20))
    axes = axes.flatten()

    print("🚀 Running large-scale stress testing...")

    for idx, (env_name, env) in enumerate(environments.items()):
        print(f"[{idx+1}/6] Analyzing environment: {env_name}")

        results_matrix = np.zeros((len(algorithms), len(NFE_MILESTONES)))
        algo_names = []

        for i, algo in enumerate(algorithms):
            algo_names.append(algo.__name__)
            nfe_hist, val_hist = run_algo_for_heatmap(algo, env, MAX_NFE, DIM, SEED)

            interp_vals = np.interp(NFE_MILESTONES, nfe_hist, val_hist)
            log_vals = np.log10(np.clip(interp_vals, 1e-8, None))
            results_matrix[i, :] = log_vals

        ax = axes[idx]
        sns.heatmap(
            results_matrix,
            annot=True,
            fmt=".1f",
            cmap="viridis_r",
            xticklabels=NFE_MILESTONES,
            yticklabels=algo_names,
            ax=ax,
            cbar_kws={"label": "Log10(Loss)"},
        )
        ax.set_title(env_name, fontsize=14, fontweight="bold")
        ax.set_xlabel("Spent NFE (Budget)", fontsize=11)
        if idx % 2 == 0:
            ax.set_ylabel("Algorithms", fontsize=11)

    plt.suptitle(
        "Global Algorithm Comparison: Robustness to Geometry, Noise, and Drift",
        fontsize=20,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    plt.savefig("expanded_algorithms_heatmap.png", dpi=300, bbox_inches="tight")
    print("✅ Super-test complete! Results saved to 'expanded_algorithms_heatmap.png'")
    plt.show()


if __name__ == "__main__":
    generate_expanded_heatmap()
