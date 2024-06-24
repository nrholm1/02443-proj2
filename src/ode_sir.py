import torch
import torch.distributions as td
from torchdiffeq import odeint

from typing import Callable


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()


class ODESIRModel(BaseModel):
    def __init__(self, beta, gamma):
        super(ODESIRModel, self).__init__()
        """
        Params:
            beta: float - transmission rate,
                likelihood of disease spread per contact between an S and an I individual.
            gamma: float - recovery rate, 
                proportion of infected individuals who recover per time unit.
        """
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, t, y):
        S, I, R = y
        dSdt = -self.beta * S * I
        dRdt = self.gamma * I
        dIdt = -dSdt - dRdt
        return torch.tensor([dSdt, dIdt, dRdt])


class ODESIRDModel(BaseModel):
    def __init__(self, beta, gamma, mu):
        super(ODESIRDModel, self).__init__()
        """
        Params:
            beta: float - transmission rate,
                likelihood of disease spread per contact between an S and an I individual.
            gamma: float - recovery rate, 
                proportion of infected individuals who recover per time unit.
            mu: float - mortality rate,
                proportion of infected individuals who die per time unit.
        """
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
    
    def forward(self, t, y):
        S, I, R, D = y
        N = S+I+R
        dSdt = -self.beta * S * I / N
        dRdt = self.gamma * I
        dDdt = self.mu*I
        dIdt = -dSdt - dRdt - dDdt
        return torch.tensor([dSdt, dIdt, dRdt, dDdt])



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from seaborn import set_style
    set_style('darkgrid')

    # Initial conditions
    S0 = .997
    I0 = .003
    R0 = .0
    D0 = .0
    y0 = torch.tensor([S0, I0, R0], dtype=torch.float32) # ! SIR
    # y0 = torch.tensor([S0, I0, R0, D0], dtype=torch.float32) # ! SIRD

    # Parameters
    beta  = .4
    gamma = .035
    mu = .005

    model = ODESIRModel(beta=beta, gamma=gamma)
    # model = SIRDModel(beta=beta, gamma=gamma, mu=mu)

    # Time points
    T = 100
    t = torch.linspace(0, T, steps=1000)
    res = odeint(model.forward, y0, t)
    print(res)

    population = 1_000

    fig,ax = plt.subplots(1,1,figsize=(10,6))
    ax.plot(t, res*population, label=['[S]usceptible', '[I]nfectious', '[R]ecovered'], linewidth=3)
    # ax.plot(t, res*population, label=['[S]usceptible', '[I]nfectious', '[R]ecovered', '[D]ead'], linewidth=3)

    ax.set_xlabel("$t$")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.set_title("[ODE] Compartment Plot Over Time")
    plt.tight_layout()
    plt.savefig("img/sir_traj.pdf")
    