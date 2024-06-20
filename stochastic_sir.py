
import random
import matplotlib.pyplot as plt
import heapq 
import torch
from torch.distributions import Exponential

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

# Inintial Values Put Here

S = 997 # Susceptible
I = 3    # Infected
R = 0    # Recovered
N = S + I + R

avg_rec_time = 5 # Average time in days to recover
recovery_rate = 0.04 #1 / avg_rec_time # Beta

disease_contact_rate = 0.4  # Beta -> Rate at which S becomes I - Rate pr susceptible-infected contact per unit time

def get_S_I_rate():
    return disease_contact_rate * I / N

# Distributions
def sample_rnd_recovery_time():
    dist = Exponential(recovery_rate)
    return dist.sample().item()

def sample_rnd_infection_times():
    dist = Exponential(rate=get_S_I_rate())
    return dist.sample(sample_shape = torch.tensor([S]))


# Event Class
class Event:
    def __init__(self, event_type, time):
        self.time = time
        self.event_type = event_type # Event type 1 => S-I && 2 => I - R

# Event Queue
events = CustomHeap(key=lambda x: x.time)

# Generate first couple of event - Events for Recovery of first I patients
current_time = 0
verbose = True

for i in range(I):
    rec_time = sample_rnd_recovery_time()
    events.push(Event(2, rec_time))
    if verbose:
        print("Someone got infected!")


def infect_someone():
    global S, I, current_time
    S -= 1
    I += 1
    events.push(Event(2, current_time + sample_rnd_recovery_time()))
    if verbose:
        print("Someone got infected at time " + str(current_time) + "!")

def recover_someone():
    global R, I, current_time
    event = events.pop()
    R += 1
    I -= 1
    current_time = event.time
    if verbose:
        print("Someone Recovered at time " + str(current_time) + "!")

progression_seq = []

while I > 0:
    if current_time > 100:
        break

    if S > 0: # Someone might get infected!
        next_infected_time = current_time + sample_rnd_infection_times()

        # Find values within next 
        next_event_time = events.get_next_time()
        next_infected_time = next_infected_time[next_infected_time < next_event_time]

        
        if next_infected_time.shape[0] == 0:
            recover_someone()
        else:
            for i in range(next_infected_time.shape[0]):
                infect_someone()
                current_time = next_infected_time[i].item()
    else:
        recover_someone()

    progression_seq.append((current_time, S, I, R))

print("Simulation Ended! Amount Not Infected: " + str(S))
print("Amount Recovered: " + str(R))


# Plot Data
# progression_seq = progression_seq[:-20] # Remove outliers
times = [t for t, n1, n2, n3 in progression_seq]
amounts1 = [n1 for t, n1, n2, n3 in progression_seq]
amounts2 = [n2 for t, n1, n2, n3 in progression_seq]
amounts3 = [n3 for t, n1, n2, n3 in progression_seq]

# Plot the data
plt.figure(figsize=(10, 6))

plt.plot(times, amounts1, label='Series 1')
plt.plot(times, amounts2, label='Series 2')
plt.plot(times, amounts3, label='Series 3')

plt.xlabel('Time')
plt.ylabel('Amount')
plt.title('Amount Change Over Time')
plt.legend()
plt.grid(True)
plt.show()