import math
import matplotlib.pyplot as plt
import torch
from torch.distributions import Exponential, Poisson, Normal

# Based on this paper http://alun.math.ncsu.edu/static/alun/tpb_01.pdf


class CountryPlagueSimulator:

    def __init__(self, population=1000000, infected_amount=5, rec_rate=0.1, infect_rate=0.1, death_rate=0.00001, immune_rate=0.02, seasonal_multiplier=1.5, time_step=0.5):
        self.S = population - infected_amount
        self.I = infected_amount
        self.Im = 0
        self.D = 0
        self.recovery_rate = rec_rate
        self.disease_contact_rate = infect_rate
        self.death_rate = death_rate
        self.immune_rate = immune_rate
        self.seasonal_multiplier_val = seasonal_multiplier
        self.time_step = time_step
        self.progression_seq = []

    def getN(self):
        return self.S + self.I + self.Im

    def seasonal_multiplier(self, t):
        min_val = 1
        max_val = self.seasonal_multiplier_val
        amplitude = (max_val - min_val) / 2
        offset = (max_val + min_val) / 2
        return amplitude * math.cos(t * 2 * math.pi / 365) + offset

    def get_I_R_rate(self):
        return self.recovery_rate * self.I

    def get_S_I_rate(self, time):
        return self.disease_contact_rate * self.S * self.I / self.getN() * self.seasonal_multiplier(time)

    def get_I_D_rate(self):
        return self.death_rate * self.I

    def get_R_S_rate(self):
        return self.immune_rate * self.Im

    def get_total_rate(self, time):
        return self.get_I_R_rate() + self.get_S_I_rate(time) + self.get_I_D_rate() + self.get_R_S_rate()

    def get_next_event(self, time):
        rate = self.get_total_rate(time) * self.time_step
        dist = Poisson(rate)
        return dist.sample().item()

    def get_event_type(self, time, amount):
        rates = [self.get_I_R_rate(), self.get_S_I_rate(time), self.get_I_D_rate(), self.get_R_S_rate()]
        rates_tensor = torch.tensor(rates, dtype=torch.float32)
        # Normalize the rates to get probabilities
        total_rate = torch.sum(rates_tensor)
        probabilities = rates_tensor / total_rate
        # Sample an event based on the probabilities
        event_index = torch.multinomial(probabilities, amount, replacement=True)
        # Index of event = event type => 0: Someone recovering (getting immune), 1: Someone getting infected, 2: someone dying, 3: Someone getting un-immune. 
        return event_index
    
    # Performs a single simulation step over the specified self.timestep time.
    def simulate(self, time):
        next_events = self.get_next_event(current_time)
        if next_events > 0:
            next_event_type = self.get_event_type(current_time, int(next_events))

            self.Im += next_event_type[next_event_type == 0].shape[0]
            self.I -= next_event_type[next_event_type == 0].shape[0]

            self.S -= next_event_type[next_event_type == 1].shape[0]
            self.I += next_event_type[next_event_type == 1].shape[0]

            self.D += next_event_type[next_event_type == 2].shape[0]
            self.I -= next_event_type[next_event_type == 2].shape[0]

            self.Im -= next_event_type[next_event_type == 3].shape[0]
            self.S += next_event_type[next_event_type == 3].shape[0]

        self.progression_seq.append((time, self.S, self.I, self.Im, self.D))

    def get_progression_sequence(self):
        return self.progression_seq
        



# Simulation Time
stop_at = 365*5

milestone_step = stop_at/20
next_milestone = stop_at/20
current_time = 0

# Initiate Denmark Simulation
denmark_plague = CountryPlagueSimulator(population=6000000, infected_amount=5, rec_rate=1/10, infect_rate=0.2, death_rate=1/1600, immune_rate=1/90, seasonal_multiplier=1.5, time_step=0.5)

while True:
    # Logging
    if current_time >= next_milestone:
        print("Simulation " + str(int(next_milestone/stop_at*100)) + "% done.")
        next_milestone += milestone_step
    if current_time > stop_at:
        break

    # Event Logic
    denmark_plague.simulate(current_time)
    
    current_time += 0.5 
    


progression_seq = denmark_plague.get_progression_sequence()
print("Simulation Ended!")


# Plot the data
times = [t for t, n1, n2, n3, n4 in progression_seq]
amounts1 = [n1 for t, n1, n2, n3, n4 in progression_seq]
amounts2 = [n2 for t, n1, n2, n3, n4 in progression_seq]
amounts3 = [n3 for t, n1, n2, n3, n4 in progression_seq]
amounts4 = [n4 for t, n1, n2, n3, n4 in progression_seq]

plt.figure(figsize=(10, 6))

plt.plot(times, amounts1, label='Susceptible (S)')
plt.plot(times, amounts2, label='Infected (I)')
plt.plot(times, amounts3, label='Immune (Im)')
plt.plot(times, amounts4, label='Deaths (D)')

plt.xlabel('Time (days)')
plt.ylabel('Number of Individuals')
plt.title(f'Disease Progression Over Time\nRecovery Rate: {denmark_plague.recovery_rate}, Contact Rate: {denmark_plague.disease_contact_rate}, Death Rate: {denmark_plague.death_rate}, Immune Rate: {denmark_plague.immune_rate:.4f}')
plt.legend()
plt.grid(True)

plt.show()
