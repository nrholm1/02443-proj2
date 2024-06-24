#%%
import torch
import torch.distributions as td
import matplotlib.pyplot as plt
from seaborn import set_style, color_palette

set_style('darkgrid')
palette = color_palette("bright")

# Enumerating the colors in the "Deep Palette"
colors = [color for color in palette]

torch.set_default_dtype(torch.double)

#%%

class SSASIRModel:
    # Define constants for state indices
    S_INDEX = 0
    E_INDEX = 1
    I_INDEX = 2
    R_INDEX = 3
    D_INDEX = 4  # Dead state

    def __init__(self, beta, a, gamma, death_erlang_rate, k, S0, E0, I0, R0, D0):
        self.beta = beta
        self.mu = 0 # todo something with birth/death rate
        self.a = a
        self.gamma = gamma
        
        self.death_erlang_rate = death_erlang_rate  # Rate parameter for the Erlang distribution
        self.k = k    # Shape parameter for the Erlang distribution
        self.deathtime_dist = td.Gamma(self.k, self.death_erlang_rate) # Erlang distribution
        # self.deathtime_dist = td.Exponential(self.mu)

        self.state = torch.tensor([S0, E0, I0, R0, D0], dtype=torch.long)
        self.times = [0]
        self.states = [self.state.clone()]
        self.event_actions = {
            0: self.exposure_event,
            1: self.infection_event,
            2: self.recovery_event,
            3: self.lose_immunity_event
        }
        self.death_times = []

    @property
    def N(self):
        return self.state[:self.D_INDEX].sum() + self.state[self.D_INDEX+1:].sum()

    def exposure_rate(self):
        return self.beta*self.state[self.S_INDEX] * self.state[self.I_INDEX] / self.N
    
    def infection_rate(self):
        return self.a * self.state[self.E_INDEX]

    def recovery_rate(self):
        return self.gamma * self.state[self.I_INDEX] - self.mu * self.state[self.R_INDEX]
    
    def lose_immunity_rate(self):
        return self.mu * self.state[self.R_INDEX]

    def select_event_and_compute_dt(self):
        rate_exposure = self.exposure_rate()
        rate_infection = self.infection_rate()
        rate_recovery = self.recovery_rate()
        total_rate = rate_exposure + rate_infection + rate_recovery

        if total_rate == 0:
            return None, None
        
        dt = td.Exponential(total_rate).sample().item()
        event_probs = torch.tensor([rate_exposure, rate_infection, rate_recovery]) / total_rate
        event = td.Categorical(event_probs).sample().item()
        return event, dt

    def exposure_event(self):
        self.state[self.S_INDEX] -= 1  # S -= 1
        self.state[self.E_INDEX] += 1  # E += 1

    def infection_event(self):
        self.state[self.E_INDEX] -= 1  # E -= 1
        self.state[self.I_INDEX] += 1  # I += 1
        # Schedule a death event for the new infected individual
        death_time = self.deathtime_dist.sample().item()
        self.death_times.append(death_time)

    def recovery_event(self):
        self.state[self.I_INDEX] -= 1  # I -= 1
        self.state[self.R_INDEX] += 1  # R += 1
        if self.death_times:
            self.death_times.pop(0)  # Remove the scheduled death time for the recovered individual

    def death_event(self):
        self.state[self.I_INDEX] -= 1  # I -= 1
        self.state[self.D_INDEX] += 1  # D += 1
        if self.death_times:
            self.death_times.pop(0)  # Remove the scheduled death time for the deceased individual
    
    def lose_immunity_event(self):
        self.state[self.R_INDEX] -= 1  # R -= 1
        self.state[self.S_INDEX] += 1  # S += 1


    def perform_event(self, event):
        if event in self.event_actions:
            self.event_actions[event]()

    def update_state(self, t):
        self.times.append(t)
        self.states.append(self.state.clone())

    @property
    def trajectory(self):
        return torch.tensor(self.times), torch.stack(self.states)
    
    @property
    def num_carriers(self):
        return self.state[SSASIRModel.I_INDEX] + self.state[SSASIRModel.E_INDEX]


def simulate(model, t_max):
    t = 0

    while (model.num_carriers > 0) and t < t_max:
        if model.death_times:
            # Find the next scheduled death time
            next_death_time = model.death_times[0]
        else:
            next_death_time = float('inf')

        # Select next event and compute time to next event
        event, dt = model.select_event_and_compute_dt()

        if event is None or dt is None:
            break

        t += dt

        # Check if a death event should occur before the next scheduled event
        if t >= next_death_time:
            model.death_event()
            model.update_state(t)
            continue

        # Perform the event
        model.perform_event(event)

        # Update state and time
        model.update_state(t)

    return model.trajectory



# Define parameters
beta = 0.5  # Exposure rate
# mu = 0.1  # something with birth/death rate
a = 0.1  # Latency for exposed -> infected
gamma = 0.1  # Recovery rate
death_erlang_rate = .1  # Rate parameter for the Erlang distribution
k = 10  # Shape parameter for the Erlang distribution
N = 50_000  # Total population
S0 = N - 1  # Initial number of susceptible individuals
E0 = 5  # Initial number of exposed individuals
I0 = 0  # Initial number of infected individuals
R0 = 0  # Initial number of recovered individuals
D0 = 0  # Initial number of dead individuals
t_max = 400  # Maximum simulation time

# Create an instance of the SSA SIR model
self = SSASIRModel(beta, a, gamma,
                        death_erlang_rate, k, 
                        S0, E0, I0, R0, D0)

# Run the simulation
times, states = simulate(self, t_max)
S_values,E_values,I_values,R_values,D_values = torch.chunk(states, 5, dim=1)

# Plot the results
fig,ax = plt.subplots(1,1,figsize=(10,6))

ax.plot(times, S_values, label='[S]usceptible', linewidth=3, color=colors[0])
ax.plot(times, E_values, label='[E]xposed',  linewidth=3, color=colors[7])
ax.plot(times, I_values, label='[I]nfected',  linewidth=3, color=colors[1])
ax.plot(times, R_values, label='[R]ecovered',   linewidth=3, color=colors[2])
ax.plot(times, D_values, label='[D]ead',   linewidth=3, color=colors[3])

ax.set_xlabel("$t$")
ax.set_ylabel("Frequency")
ax.legend()
ax.set_title("[Stochastic] Compartment Plot Over Time")
plt.tight_layout()
plt.savefig("../img/stoch_sir_traj.pdf")
# %%
