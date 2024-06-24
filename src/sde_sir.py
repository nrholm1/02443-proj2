#%%
import torch
import torchsde

import matplotlib.pyplot as plt
from seaborn import set_style
set_style('darkgrid')

from functools import lru_cache

#%%
eps = 1e-5


# Define the SDE
class SDESIRModel(torchsde.SDEIto):
    def __init__(self, beta, gamma, N):
        super().__init__(noise_type="general")
        self.beta = beta
        self.gamma = gamma
        self.N = N

    def f(self, t, y):
        """Drift term"""
        S, I = y[:, 0], y[:, 1]
        SI_N = self.beta * S * I / self.N
        dSdt = -SI_N
        dIdt = -dSdt - self.gamma * I
        return torch.stack([dSdt, dIdt], dim=1)

    def g(self, t, y):
        """Diffusion term"""
        S, I = y[:, 0], y[:, 1]
        SI_N = torch.clamp(self.beta * S * I / self.N, min=0)#, max=self.N)
        gamma_I = torch.clamp(self.gamma * I, min=0)#, max=self.N)
        # SI_N = self.beta * S * I / self.N
        # gamma_I = self.gamma * I
        
        dS_noise = -torch.sqrt(SI_N + eps)
        dI_noise_1 = torch.sqrt(SI_N + eps)
        dI_noise_2 = -torch.sqrt(gamma_I + eps)
        
        g_matrix = torch.zeros(y.size(0), 2, 2, device=y.device)
        g_matrix[:, 0, 0] = dS_noise
        g_matrix[:, 1, 0] = dI_noise_1
        g_matrix[:, 1, 1] = dI_noise_2
        
        return g_matrix



# Parameters for the SDE
beta = 0.8   # Infection rate
gamma = 0.04  # Recovery rate
N = 10_000.     # Total population

# Create the SDE object
sde = SDESIRModel(beta=beta, gamma=gamma, N=N)

# Initial condition
I0 = 10
S0 = N - I0     # Initial value
R0 = 0
y0 = torch.tensor([S0,I0]).unsqueeze(0)
num_repeats = 10
y0s = torch.repeat_interleave(y0, num_repeats, dim=0)


# Time points to solve the SDE
t0, t1 = 0.0, 50.0  # start and end times
steps = 1_000
ts = torch.linspace(t0, t1, steps)


#%% 
with torch.no_grad():
    # Solve the SDE
    ys = torchsde.sdeint(sde, y0s, ts)

#%%
ts_truncated = ts#[:200]
ys_truncated = ys#[:200]
ys_clamped = torch.clamp(ys_truncated, min=0, max=N)

ys_mean = ys_clamped.mean(dim=1)
ys_std = ys_clamped.std(dim=1)

Rs = N - ys_truncated.sum(dim=2)
Rs_mean = Rs.mean(dim=1)
Rs_std = Rs.std(dim=1)

# Print the results
# print(ys)
plt.plot(ts_truncated, torch.round(ys_mean[:,0]), label="S mean", color='red', linewidth=2)
plt.plot(ts_truncated, torch.round(ys_mean[:,1]), label="I mean", color='green', linewidth=2)
plt.plot(ts_truncated, torch.round(Rs_mean),      label="R mean", color='blue', linewidth=2)

plt.fill_between(ts_truncated, torch.clamp(ys_mean[:,0] - 2*ys_std[:,0], min=0), torch.clamp(ys_mean[:,0] + 2*ys_std[:,0], max=N), alpha=.2, color='red')
plt.fill_between(ts_truncated, torch.clamp(ys_mean[:,1] - 2*ys_std[:,1], min=0), torch.clamp(ys_mean[:,1] + 2*ys_std[:,1], max=N), alpha=.2, color='green')
plt.fill_between(ts_truncated, torch.clamp(Rs_mean - 2*Rs_std, min=0), torch.clamp(Rs_mean + 2*Rs_std, max=N), alpha=.2, color='blue')

plt.legend()
plt.xlabel("Time $t$")
plt.ylabel("Population")

plt.title(f"SDE-based SIR with ~95% CI, based on {num_repeats} realizations")

plt.show()