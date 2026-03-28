import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# --- Constants ---
N = 1500
gamma = 1.0  # day^-1 (1 day per infection stage; 9 stages => ~9 days infectious)

# --- ODE system: y = [S, I1, I2, ..., I9, R] ---
def rhs(t, y, beta):
    S = y[0]
    I = y[1:10]               # I1..I9
    Itot = I.sum()

    dS = -(beta / N) * S * Itot

    dI = np.zeros(9)
    dI[0] = (beta / N) * S * Itot - gamma * I[0]  # I1
    for j in range(1, 9):                          # I2..I9
        dI[j] = gamma * I[j-1] - gamma * I[j]

    dR = gamma * I[8]  # leaves I9 into R

    return np.concatenate([[dS], dI, [dR]])

# --- Initial conditions ---
y0 = np.zeros(11)
y0[0] = N - 1   # S(0)
y0[1] = 1       # I1(0)
# I2..I9(0)=0, R(0)=0 already

# --- Helper: total infected at t=7 for a given beta ---
def total_infected_at_7(beta):
    sol = solve_ivp(lambda t, y: rhs(t, y, beta),
                    t_span=(0, 7),
                    y0=y0,
                    t_eval=[7],
                    rtol=1e-9,
                    atol=1e-12)
    return sol.y[1:10, 0].sum()

# --- Fit beta so that I(7)=10 ---
target = 10.0
f = lambda beta: total_infected_at_7(beta) - target

beta_star = brentq(f, 0.3, 0.4)  # bracket found empirically
print(f"Fitted beta = {beta_star:.6f} per day")

# --- Solve outbreak over time ---
t_end = 200
t_eval = np.linspace(0, t_end, 2001)

sol = solve_ivp(lambda t, y: rhs(t, y, beta_star),
                t_span=(0, t_end),
                y0=y0,
                t_eval=t_eval,
                rtol=1e-8,
                atol=1e-10)

t = sol.t
S = sol.y[0]
I = sol.y[1:10].sum(axis=0)  # total infected
R = sol.y[10]

# --- Key outputs ---
peak_I = I.max()
t_peak = t[I.argmax()]
final_R = R[-1]
final_S = S[-1]

print(f"Peak infected ~ {peak_I:.2f} people at t ~ {t_peak:.1f} days")
print(f"Final recovered ~ {final_R:.2f} people")
print(f"Final susceptible ~ {final_S:.2f} people")

# --- Plot ---
plt.figure()
plt.plot(t, S, label="S(t)")
plt.plot(t, I, label="Total infected I(t)")
plt.plot(t, R, label="R(t)")
plt.xlabel("Time (days)")
plt.ylabel("People")
plt.title("11-compartment H1N1 model (9 infectious stages)")
plt.legend()
plt.show()