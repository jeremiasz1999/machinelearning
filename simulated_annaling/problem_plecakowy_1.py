import random
import math

#   Dane problemu
items = [
    {"weight": 10, "value": 60},
    {"weight": 20, "value": 100},
    {"weight": 30, "value": 120},
    {"weight": 25, "value": 75},
    {"weight": 15, "value": 45},
]

capacity = 50
num_items = len(items)

#   Funkcje pomocnicze
def total_weight(solution):
    return sum(items[i]['weight'] for i in range(num_items) if solution[i] == 1)

def total_value(solution):
    return sum(items[i]['value'] for i in range(num_items) if solution[i] == 1)

def random_solution():
    while True:
        sol = [random.randint(0, 1) for _ in range(num_items)]
        if total_weight(sol) <= capacity:
            return sol

def neighbor(solution):
    neighbor = solution[:]
    index = random.randint(0, num_items - 1)
    neighbor[index] = 1 - neighbor[index]  # toggle 0<->1
    if total_weight(neighbor) <= capacity:
        return neighbor
    else:
        return solution  # return original if neighbor is invalid

#   Symulowane wyżarzanie
def simulated_annealing(initial_temp=1000, cooling_rate=0.95, iterations=1000):
    current = random_solution()
    best = current[:]
    temperature = initial_temp

    for _ in range(iterations):
        candidate = neighbor(current)
        delta = total_value(candidate) - total_value(current)

        if delta > 0 or random.uniform(0, 1) < math.exp(delta / temperature):
            current = candidate
            if total_value(current) > total_value(best):
                best = current[:]

        temperature *= cooling_rate

    return best

#   Uruchomienie
solution = simulated_annealing()
print("Najlepsze rozwiązanie:", solution)
print("Całkowita wartość:", total_value(solution))
print("Całkowita waga:", total_weight(solution))
