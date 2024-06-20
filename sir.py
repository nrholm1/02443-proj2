import torch
import torch.distributions as td
from torchdiffeq import odeint


class BaseModel(torch.nn.Module):
    ...


class SIR(BaseModel):
    def __init__(self, beta, gamma):
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
        S,I,R = y
        dSdt = - self.beta*S*I
        dRdt = self.gamma*I
        dIdt = - dSdt - dRdt
        return torch.Tensor([dSdt, dRdt, dIdt])



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from seaborn import set_style
    set_style('darkgrid')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="sir", choices=["sir"], help='which model to use for epidemic simulation (default: %(default)s)')
    parser.add_argument('--S0', type=int, default=1, help='initial condition for [S]usceptible (default: %(default)s)')
    parser.add_argument('--I0', type=int, default=1, help='initial condition for [I]nfected (default: %(default)s)')
    parser.add_argument('--R0', type=int, default=1, help='initial condition for [R]ecovered (default: %(default)s)')
    parser.add_argument('--beta', type=float, default=1e-3, help='infection likelihood parameter (default: %(default)s)')
    parser.add_argument('--gamma', type=float, default=1e-3, help='recovery rate parameter (default: %(default)s)')
    args = parser.parse_args()

    print('\n# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)
    print("")


    # Initial conditions
    S0, I0, R0 = 990, 10, 0
    y0 = torch.tensor([S0, I0, R0], dtype=torch.float32)

    # SIR model
    beta = 0.002
    gamma = 0.01
    sir = SIR(beta=beta, gamma=gamma)

    # Time points
    num_time_units = 50
    T = torch.linspace(0, num_time_units, 4*(num_time_units+1))
    res = odeint(sir.forward, y0, t=T).long()
    print(res)
    plt.plot(T, res, label=['[S]usceptible', '[I]nfected', '[R]ecovered'])
    plt.savefig("img/sir_traj.pdf")
    