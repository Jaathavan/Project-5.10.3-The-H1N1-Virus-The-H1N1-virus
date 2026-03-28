import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# ============================================================
# H1N1 outbreak model with 9-day infectious period
# State vector:
# y = [S, I1, I2, I3, I4, I5, I6, I7, I8, I9, R]
# ============================================================

# ----------------------------
# Model parameters
# ----------------------------
N = 1500                  # total population
N_STAGES = 9              # infected stages I1,...,I9
GAMMA = 1.0               # progression rate: 1 stage per day
TARGET_I7 = 10.0          # condition: total infected at day 7
T_END = 200               # simulation length (days)
BETA_BRACKET = (0.3, 0.4) # search interval for beta

# ----------------------------
# Initial conditions
# ----------------------------
# S(0) = 1499, I1(0) = 1, I2...I9(0)=0, R(0)=0
y0 = np.zeros(N_STAGES + 2)
y0[0] = N - 1   # susceptible
y0[1] = 1       # first infected compartment


def h1n1_system(t, y, beta):
    """
    11-compartment H1N1 ODE system.

    Parameters
    ----------
    t : float
        Time variable.
    y : ndarray
        State vector [S, I1, ..., I9, R].
    beta : float
        Transmission parameter.

    Returns
    -------
    dydt : ndarray
        Time derivatives of the state variables.
    """
    S = y[0]
    I = y[1:1 + N_STAGES]
    Itot = np.sum(I)

    dydt = np.zeros_like(y)

    # Susceptible equation
    dydt[0] = -(beta / N) * S * Itot

    # First infected stage
    dydt[1] = (beta / N) * S * Itot - GAMMA * I[0]

    # Intermediate infected stages I2,...,I9
    for j in range(1, N_STAGES):
        dydt[j + 1] = GAMMA * I[j - 1] - GAMMA * I[j]

    # Recovered equation
    dydt[-1] = GAMMA * I[-1]

    return dydt


def solve_model(beta, t_end=T_END, num_points=2001):
    """
    Solve the H1N1 ODE system for a given beta using RK45.
    """
    t_eval = np.linspace(0, t_end, num_points)

    sol = solve_ivp(
        fun=lambda t, y: h1n1_system(t, y, beta),
        t_span=(0, t_end),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-9,
        atol=1e-12
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    return sol


def total_infected_at_day_7(beta):
    """
    Compute I(7) = I1(7) + ... + I9(7) for a given beta.
    """
    sol = solve_ivp(
        fun=lambda t, y: h1n1_system(t, y, beta),
        t_span=(0, 7),
        y0=y0,
        t_eval=[7],
        method="RK45",
        rtol=1e-9,
        atol=1e-12
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed during beta fit: {sol.message}")

    infected_stages = sol.y[1:1 + N_STAGES, 0]
    return np.sum(infected_stages)


def fit_beta(target_i7=TARGET_I7, bracket=BETA_BRACKET):
    """
    Find beta so that total infected at day 7 satisfies I(7) = target_i7.
    """
    def objective(beta):
        return total_infected_at_day_7(beta) - target_i7

    return brentq(objective, bracket[0], bracket[1])


def compute_summary(sol):
    """
    Compute key summary statistics from the numerical solution.
    """
    t = sol.t
    S = sol.y[0]
    I_stages = sol.y[1:1 + N_STAGES]
    I_total = np.sum(I_stages, axis=0)
    R = sol.y[-1]

    peak_index = np.argmax(I_total)
    peak_I = I_total[peak_index]
    t_peak = t[peak_index]
    final_S = S[-1]
    final_R = R[-1]
    conservation_error = np.max(np.abs(S + I_total + R - N))

    return {
        "t": t,
        "S": S,
        "I_stages": I_stages,
        "I_total": I_total,
        "R": R,
        "peak_I": peak_I,
        "t_peak": t_peak,
        "final_S": final_S,
        "final_R": final_R,
        "conservation_error": conservation_error,
    }


def make_plots(results):
    """
    Save the two project plots:
    1. S(t), I(t), R(t)
    2. I1(t), ..., I9(t)
    """
    t = results["t"]
    S = results["S"]
    I_total = results["I_total"]
    R = results["R"]
    I_stages = results["I_stages"]

    # Plot 1: S, I, R
    plt.figure(figsize=(8, 5))
    plt.plot(t, S, label="S(t)", linewidth=2)
    plt.plot(t, I_total, label="I(t) = Î£ I_j(t)", linewidth=2)
    plt.plot(t, R, label="R(t)", linewidth=2)
    plt.xlabel("Time (days)")
    plt.ylabel("People")
    plt.title("H1N1 outbreak model (N=1500, 9-day infectious period)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sir_total.png", dpi=300)
    plt.close()

    # Plot 2: infected stages
    plt.figure(figsize=(8, 5))
    for j in range(N_STAGES):
        plt.plot(t, I_stages[j], label=f"I_{j+1}(t)", linewidth=2)
    plt.xlabel("Time (days)")
    plt.ylabel("People")
    plt.title("Infected compartments by day of infection")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig("infected_stages.png", dpi=300)
    plt.close()


def main():
    # Step 1: Fit beta using I(7) = 10
    beta_star = fit_beta()

    # Step 2: Check calibration
    i7_value = total_infected_at_day_7(beta_star)

    # Step 3: Solve full outbreak
    sol = solve_model(beta_star)
    results = compute_summary(sol)

    # Step 4: Print results
    print("=== H1N1 Outbreak Model Results ===")
    print(f"Fitted transmission parameter beta = {beta_star:.6f} day^-1")
    print(f"Check condition: I(7) = {i7_value:.6f} people")
    print(f"Peak infected = {results['peak_I']:.2f} people")
    print(f"Time of peak = {results['t_peak']:.1f} days")
    print(f"Final susceptible = {results['final_S']:.2f} people")
    print(f"Final recovered = {results['final_R']:.2f} people")
    print(f"Max conservation error = {results['conservation_error']:.2e}")

    # Step 5: Save plots
    make_plots(results)
    print("Plots saved as 'sir_total.png' and 'infected_stages.png'.")


if __name__ == "__main__":
    main()