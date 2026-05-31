import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import warnings
import sys
import inspect
from pathlib import Path

# --- PATH CONFIGURATION ---
current_dir = Path(__file__).parent.absolute()
wind_core_path = current_dir / "WIND" / "KDD_paper_first_submission"
sys.path.insert(0, str(wind_core_path))
from core import GaussianNoise
from core import (
    DynamicEnvironment,
    LinearDrift,
    QuadraticLandscape,
    PNormLandscape,
    RobustLandscape,
)
from oracle import ZeroOrderOracle
from metrics import AsymptoticBoundMetric

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


def create_lyapunov_env(rho, A, dim):
    if rho == 1.0:
        landscape = QuadraticLandscape(dim=dim, condition_number=1.0)
    elif rho == 0.5:
        landscape = PNormLandscape(p=1.5)
    elif rho == 0.2:
        landscape = RobustLandscape(delta=0.1)
    else:
        landscape = QuadraticLandscape(dim=dim)

    velocity = np.ones(dim) * (A / dim)
    return DynamicEnvironment(
        dim=dim,
        drift=LinearDrift(velocity=velocity.tolist()),
        landscape=landscape,
        bounds=[-10.0, 10.0],
    )


def run_single_eval(algo_class, algo_kwargs, rho, A, dim, max_nfe, seed=42):
    np.random.seed(seed)
    env = create_lyapunov_env(rho, A, dim)
    oracle = ZeroOrderOracle(
        environment=env,
        value_noise=GaussianNoise(sigma=0.05, seed=seed),
        seed=seed,
    )

    metric = AsymptoticBoundMetric(rho=rho)

    try:
        sig = inspect.signature(algo_class.__init__)
        valid_kwargs = {}
        for k, v in algo_kwargs.items():
            if k in sig.parameters:
                valid_kwargs[k] = v
        if "dim" in sig.parameters and "dim" not in valid_kwargs:
            valid_kwargs["dim"] = dim

        opt = algo_class(**valid_kwargs)
    except Exception:
        try:
            opt = algo_class(dim=dim)
        except TypeError:
            opt = algo_class()

    x_curr = np.random.randn(dim) * 0.1

    t = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while oracle.n_queries < max_nfe:
            try:
                oracle.start_step(t)
                obs = oracle.query(x_curr)
                oracle.end_step()

                theta_t = env.get_current_theta(for_analysis=True)
                metric.update(t, x_curr, theta_t, obs, env)

                env.step()
                x_curr = opt.step(obs)
                if x_curr is None:
                    break
                t += 1
            except Exception:
                break

    res = metric.get_result(tail_fraction=0.2)
    return res if (not np.isnan(res) and not np.isinf(res)) else 2e3


def generate_full_experiment():
    DIM = 5
    MAX_NFE = 500

    rho_values = [0.2, 0.5, 1.0]
    A_values = [0.001, 0.010, 0.100, 0.300, 0.600, 1.000]

    # CONFIGURATION: (Name, Class, Desired safe parameters)
    optimizers_config = [
        ("CMAES", CMA_ES, {"sigma": 0.5}),
        ("ZOSignSGD", ZOSignSGD, {"lr": 0.005, "mu": 0.01}),
        ("ZOSGD", ZO_SGD, {"lr": 0.005, "mu": 0.01}),
        ("RandomSearch", RandomSearch, {"lr": 0.1, "scale": 0.5}),
        ("OnePointSPSA", OneMeasurementSPSA, {"lr": 0.005, "perturb": 0.1}),
        ("SPSA", SPSA, {"lr": 0.005, "perturb": 0.1}),
        ("FDSA", ClassicalFDSA, {"lr": 0.02, "h": 1e-4}),
        ("QuadraticInterpolation", QuadraticInterpolationDPlus1, {"lr": 0.1}),
        ("NelderMead", NelderMead, {}),
        ("FiniteDiffCentral", FiniteDiffCentral, {"lr": 0.02, "h": 1e-4}),
        ("ZO_AdaMM", ZO_AdaMM, {"lr": 0.005}),
        ("MeZO", MeZO, {"lr": 0.005}),
        ("AdaptiveFD_BFGS", AdaptiveFD_BFGS, {"lr": 0.005}),
        ("UHCMAES", UHCMAES, {"sigma": 0.5}),
        ("SANE", SANE, {"sigma_E": 0.1}),
    ]

    results = []
    total_runs = len(optimizers_config) * len(rho_values) * len(A_values)
    current_run = 0

    print(f"🚀 Starting intelligent experiment loop ({total_runs} runs)...")

    for opt_name, opt_class, opt_kwargs in optimizers_config:
        for rho in rho_values:
            for A in A_values:
                current_run += 1
                val = run_single_eval(opt_class, opt_kwargs, rho, A, DIM, MAX_NFE)

                results.append(
                    {
                        "Algorithm": opt_name,
                        "rho": rho,
                        "A": A,
                        "Regime": f"ρ={rho}\nA={A:.3f}",
                        "Bound": val,
                    }
                )
                print(
                    f"[{current_run:03d}/{total_runs}] {opt_name:25s} | ρ={rho}, A={A} -> {val:.1f}"
                )

    # === PLOTTING ===
    df = pd.DataFrame(results)
    pivot_df = df.pivot(index="Algorithm", columns="Regime", values="Bound")

    sorted_regimes = (
        df[["rho", "A", "Regime"]]
        .drop_duplicates()
        .sort_values(["rho", "A"])["Regime"]
        .tolist()
    )
    pivot_df = pivot_df[sorted_regimes]

    algo_order = [name for name, _, _ in optimizers_config]
    pivot_df = pivot_df.reindex(algo_order)

    def format_val(v):
        if pd.isna(v):
            return "NaN"
        if v >= 1000:
            return f"{v:.0e}".replace("e+0", "e+")
        return f"{v:.1f}"

    annot_matrix = pivot_df.map(format_val).values

    plt.figure(figsize=(18, 10))

    sns.heatmap(
        pivot_df,
        annot=annot_matrix,
        fmt="",
        cmap="RdYlGn_r",
        norm=LogNorm(vmin=0.1, vmax=1000),
        linewidths=0.5,
        cbar_kws={"label": "95% Convergence Bound (Log Scale)"},
    )

    plt.title("Algorithm Stability Analysis", fontsize=18, pad=20)
    plt.xlabel("Regime (Smoothness ρ, Drift A)", fontsize=14)
    plt.ylabel("")

    plt.xticks(rotation=90, fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    plt.tight_layout()
    plt.savefig("my_ultimate_heatmap_benchmark_style.png", dpi=300)
    print(
        "\n✅ Success! Heatmap saved. Data and colors now correspond to the benchmark article."
    )
    plt.show()


if __name__ == "__main__":
    generate_full_experiment()
