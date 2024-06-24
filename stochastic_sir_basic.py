import random, math
import matplotlib.pyplot as plt
import heapq 
import torch
from torch.distributions import Exponential, Normal

# Custom Heap
class CustomHeap(object):
    def __init__(self, initial=None, key=lambda x:x):
        self.key = key
        self.index = 0
        if initial:
            self._data = [(key(item), i, item) for i, item in enumerate(initial)]
            self.index = len(self._data)
            heapq.heapify(self._data)
        else:
            self._data = []

    def push(self, item):
        heapq.heappush(self._data, (self.key(item), self.index, item))
        self.index += 1

    def pop(self):
        return heapq.heappop(self._data)[2]

    def get_size(self):
        return len(self._data)

    def get_next_time(self):
        if self.get_size() > 0:
            return self._data[0][2].time
        return -1

# Initial Values
S = 9947   # Susceptible
I = 5      # Infected
R = 0      # Recovered
D = 0      # Amount Died
N = S + I + R

# Simulation Time
stop_at = 365*2

avg_rec_time = 10  # Average time in days to recover
recovery_rate = 1/avg_rec_time # Theta

disease_contact_rate = 0.2  # Beta: Rate at which S becomes I


def seasonal_multiplier(t):
    return (math.cos(t*2*math.pi/365)+5)/4

def get_S_I_rate(time):
    return disease_contact_rate * I / ( S + I + R) * seasonal_multiplier(time)

# Distributions
class DistributionSamplers:
    def __init__(self, recovery_rate):
        self.recovery_dist  = Exponential(recovery_rate)
        # self.recovery_dist  = Normal(loc=14, scale=2)
        

    def sample_rnd_recovery_time(self):
        return self.recovery_dist.sample().abs().item()

    def sample_rnd_infection_times(self, current_time):
        infection_dist = Exponential(rate=get_S_I_rate(current_time))
        return infection_dist.sample(sample_shape=torch.tensor([S]))

# Event Class
class Event:
    def __init__(self, event_type, time):
        self.time = time
        self.event_type = event_type # Event type 1 => S-I, 2 => I-D, 3 => R-S

# Event Queue
events = CustomHeap(key=lambda x: x.time)

# Generate first couple of events - Events for Recovery of first I patients
current_time = 0
verbose = False

def infect_someone(sampler):
    global S, I, current_time
    S -= 1
    I += 1

    time_to_recover = current_time + sampler.sample_rnd_recovery_time()
    
    events.push(Event(1, time_to_recover))
    
    if verbose:
        print("Someone got infected at time " + str(current_time) + "!")

def handle_event(sampler):
    global S, I, R, D, current_time
    
    event = events.pop()

    if event.event_type == 1:  # Someone recovers and becomes immune - add stop being immune time
        R += 1
        I -= 1

    
    current_time = event.time
    if verbose:
        print("Event handled at time " + str(current_time) + "!")

progression_seq = []

sampler = DistributionSamplers(recovery_rate)

for i in range(I):
    time_to_recover = current_time + sampler.sample_rnd_recovery_time()
    
    events.push(Event(1, time_to_recover))

    if verbose:
        print("Someone got infected!")


milestone_step = stop_at/20
next_milestone = stop_at/20

while events.get_size() > 0: # While there are still events to occur
    
    if current_time >= next_milestone:
        print("Simulation " + str(int(next_milestone/stop_at*100)) + "% done.")
        next_milestone += milestone_step
        

    if current_time > stop_at:
        break

    if S > 0 and I > 0:  # Someone might get infected!
        next_infected_time = current_time + sampler.sample_rnd_infection_times(current_time)

        # Find values within next event time
        next_event_time = events.get_next_time()
        next_infected_time = next_infected_time[next_infected_time < next_event_time]

        for i in range(next_infected_time.shape[0]):
            infect_someone(sampler)
            current_time = next_infected_time[i].item()

    handle_event(sampler)
    progression_seq.append((current_time, S, I, R, D))

print("Simulation Ended! Amount Not Infected: " + str(S))
print("Amount Recovered: " + str(R))
print("Amount Died: " + str(D))

# Plot the data
times    = [t for t, n1, n2, n3, n4  in progression_seq]
amounts1 = [n1 for t, n1, n2, n3, n4 in progression_seq]
amounts2 = [n2 for t, n1, n2, n3, n4 in progression_seq]
amounts3 = [n3 for t, n1, n2, n3, n4 in progression_seq]
amounts4 = [n4 for t, n1, n2, n3, n4 in progression_seq]

plt.figure(figsize=(10, 6))

plt.plot(times, amounts1, label='Susceptible (S)')
plt.plot(times, amounts2, label='Infected (I)')
plt.plot(times, amounts3, label='Recovered (R)')

plt.xlabel('Time (days)')
plt.ylabel('Number of Individuals')
plt.title(f'Disease Progression Over Time\nRecovery Rate: {recovery_rate}, Contact Rate: {disease_contact_rate}')
plt.legend()
plt.grid(True)

plt.show()
