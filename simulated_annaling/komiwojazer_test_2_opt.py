import random
import math
import csv
import time

# --- TSP Helpers --- #
def load_cities_from_csv(filename="generated_cities.csv"):
    cities = []
    try:
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                city_id, x, y = row
                cities.append((float(x), float(y)))
        print(f"Miasta zostały wczytane z pliku: {filename}")
    except FileNotFoundError:
        print(f" Plik {filename} nie został znaleziony.")
        return None
    except Exception as e:
        print(f" Wystąpił błąd podczas wczytywania pliku: {e}")
        return None
    return cities

def total_distance(tour, cities):
    dist = 0
    for i in range(len(tour)):
        x1, y1 = cities[tour[i]]
        x2, y2 = cities[tour[(i + 1) % len(tour)]]
        dist += math.hypot(x2 - x1, y2 - y1)
    return dist

# --- Two versions of 2-opt --- #
def naive_two_opt(tour):
    a, b = sorted(random.sample(range(len(tour)), 2))
    return tour[:a] + tour[a:b+1][::-1] + tour[b+1:]

def classic_two_opt(tour):
    n = len(tour)
    if n < 4:
        return tour[:]
    while True:
        a, b = sorted(random.sample(range(n), 2))
        if b - a > 1:
            break
    return tour[:a] + tour[a:b+1][::-1] + tour[b+1:]

# --- Transition Sampling --- #
def generate_positive_transitions(cities, swap_function, num_transitions=100):
    transitions = []
    for _ in range(num_transitions * 2):
        tour = list(range(len(cities)))
        random.shuffle(tour)
        before = total_distance(tour, cities)
        after_tour = swap_function(tour)
        after = total_distance(after_tour, cities)
        if after > before:
            transitions.append((before, after))
        if len(transitions) >= num_transitions:
            break
    return transitions

def estimate_acceptance(transitions, T):
    numerator = sum(math.exp(-Emax / T) for (_, Emax) in transitions)
    denominator = sum(math.exp(-Emin / T) for (Emin, _) in transitions)
    return numerator / denominator

def compute_initial_temperature(transitions, chi_0=0.8, p=1, epsilon=1e-3, T1=None):
    if T1 is None:
        avg_delta = sum(Emax - Emin for (Emin, Emax) in transitions) / len(transitions)
        T1 = -avg_delta / math.log(chi_0)
    Tn = T1
    iteration = 0
    while True:
        chi_hat = estimate_acceptance(transitions, Tn)
        if abs(chi_hat - chi_0) <= epsilon:
            break
        factor = (math.log(chi_hat) / math.log(chi_0)) ** (1 / p)
        Tn = Tn * factor
        iteration += 1
        if iteration > 1000:
            break
    return Tn

def adaptive_temperature_estimation(cities, swap_function, chi_0=0.8, epsilon_T=1e-2, max_steps=5):
    previous_T = None
    transitions_count = 50
    for step in range(max_steps):
        transitions = generate_positive_transitions(cities, swap_function, num_transitions=transitions_count)
        if not transitions:
            print(" Brak pogarszających przejść.")
            return None
        T = compute_initial_temperature(transitions, chi_0=chi_0)
        print(f" Próba {step+1} ({transitions_count} przejść): T₀ = {T:.4f}")
        if previous_T is not None and abs(T - previous_T) < epsilon_T:
            return T
        previous_T = T
        transitions_count *= 2
    print(" Temperatura się nie ustabilizowała.")
    return previous_T

# --- Simulated Annealing --- #
def simulated_annealing_tsp(cities, T0, swap_function, alpha=0.995, iterations=10000):
    current = list(range(len(cities)))
    random.shuffle(current)
    current_cost = total_distance(current, cities)
    best = current[:]
    best_cost = current_cost
    T = T0
    for i in range(iterations):
        candidate = swap_function(current)
        candidate_cost = total_distance(candidate, cities)
        delta = candidate_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = candidate
            current_cost = candidate_cost
            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost
        T *= alpha
    return best, best_cost

# --- RUN COMPARISON --- #
if __name__ == "__main__":
    cities = load_cities_from_csv("../generated_cities.csv")
    if cities is None:
        exit()

    print("\n--- Porównanie naive 2-OPT vs classic 2-OPT ---")
    for name, swap_func in [("Naive 2-OPT", naive_two_opt), ("Classic 2-OPT", classic_two_opt)]:
        print(f"\n🔧 {name}:")
        T0 = adaptive_temperature_estimation(cities, swap_func)
        if T0 is None:
            continue
        start_time = time.time()
        best_tour, best_cost = simulated_annealing_tsp(cities, T0, swap_func)
        duration = time.time() - start_time
        print(f" Najlepszy dystans = {best_cost:.2f} | ⏱️ Czas = {duration:.2f} s")
