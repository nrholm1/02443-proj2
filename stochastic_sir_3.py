import random, math
import matplotlib.pyplot as plt
import heapq 
import torch
from torch.distributions import Exponential, Poisson, Normal

# Based on this paper http://alun.math.ncsu.edu/static/alun/tpb_01.pdf

# Initial Values
S = 99947   # Susceptible
I = 5      # Infected
R = 0      # Recovered
D = 0      # Amount Died
N = S + I + R

current_time = 0

# Simulation Time
stop_at = 365*5

avg_rec_time = 10  # Average time in days to recover
recovery_rate = 1/avg_rec_time # Theta

disease_contact_rate = 0.1  # Beta: Rate at which S becomes I

avg_time_to_death = 1600
death_rate = 1 / avg_time_to_death

avg_immune_time = 30*3
immune_rate = 1 / avg_immune_time

def seasonal_multiplier(t):
    return (math.cos(t * 2 * math.pi / 365) + 5) / 4

def get_I_R_rate():
    return recovery_rate * I

def get_S_I_rate(time):
    return disease_contact_rate * S * I / N * seasonal_multiplier(time)

def get_I_D_rate():
    return death_rate * I

def get_R_S_rate():
    return immune_rate * R

def get_total_rate(time):
    return get_I_R_rate() + get_S_I_rate(time) + get_I_D_rate() + get_R_S_rate()


def get_next_event(time):
    dist = Poisson(rate=get_total_rate(time)*0.5)
    return dist.sample().item()

def get_event_type(time, amount):
    rates = [get_I_R_rate(), get_S_I_rate(time), get_I_D_rate(), get_R_S_rate()]
    rates_tensor = torch.tensor(rates, dtype=torch.float32)
    # Normalize the rates to get probabilities
    total_rate = torch.sum(rates_tensor)
    probabilities = rates_tensor / total_rate
    # Sample an event based on the probabilities
    event_index = torch.multinomial(probabilities, amount, replacement=True)
    # Index of event = event type => 0: Someone recovering (getting immune), 1: Someone getting infected, 2: someone dying, 3: Someone getting un-immune. 
    return event_index

progression_seq = []
milestone_step = stop_at/20
next_milestone = stop_at/20

while True:
    # Logging
    if current_time >= next_milestone:
        print("Simulation " + str(int(next_milestone/stop_at*100)) + "% done.")
        next_milestone += milestone_step
    if current_time > stop_at:
        break

    # Event Logic
    next_events = get_next_event(current_time)
    if next_events > 0:
        next_event_type = get_event_type(current_time, int(next_events))

        R += next_event_type[next_event_type == 0].shape[0]
        I -= next_event_type[next_event_type == 0].shape[0]

        S -= next_event_type[next_event_type == 1].shape[0]
        I += next_event_type[next_event_type == 1].shape[0]

        D += next_event_type[next_event_type == 2].shape[0]
        I -= next_event_type[next_event_type == 2].shape[0]

        R -= next_event_type[next_event_type == 3].shape[0]
        S += next_event_type[next_event_type == 3].shape[0]
    
    current_time += 0.5 
    progression_seq.append((current_time, S, I, R, D))



print("Simulation Ended! Amount Not Infected: " + str(S))
print("Amount Recovered: " + str(R))
print("Amount Died: " + str(D))

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
plt.title(f'Disease Progression Over Time\nRecovery Rate: {recovery_rate}, Contact Rate: {disease_contact_rate}, Death Rate: {death_rate}, Immune Rate: {immune_rate}')
plt.legend()
plt.grid(True)

plt.show()
