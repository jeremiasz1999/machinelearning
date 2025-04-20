import random
import math

# --- TSP Helpers --- #
def generate_cities(n, grid_size=100):
    return [(random.uniform(0, grid_size), random.uniform(0, grid_size)) for _ in range(n)]

def total_distance(tour, cities):
    dist = 0
    for i in range(len(tour)):
        x1, y1 = cities[tour[i]]
        x2, y2 = cities[tour[(i + 1) % len(tour)]]
        dist += math.hypot(x2 - x1, y2 - y1)
    return dist

def two_opt_swap(tour):
    a, b = sorted(random.sample(range(len(tour)), 2))
    return tour[:a] + tour[a:b+1][::-1] + tour[b+1:]

# --- Initial Temperature Estimation --- #
def generate_positive_transitions(cities, num_transitions=100):
    transitions = []
    for _ in range(num_transitions * 2):
        tour = list(range(len(cities)))
        random.shuffle(tour)
        before = total_distance(tour, cities)
        after_tour = two_opt_swap(tour)
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
            print("⚠️ Zbyt wiele iteracji – możliwa niestabilność")
            break

    return Tn

# --- Simulated Annealing --- #
def simulated_annealing_tsp(cities, T0, alpha=0.995, iterations=10000):
    current = list(range(len(cities)))
    random.shuffle(current)
    current_cost = total_distance(current, cities)
    best = current[:]
    best_cost = current_cost
    T = T0

    for i in range(iterations):
        candidate = two_opt_swap(current)
        candidate_cost = total_distance(candidate, cities)
        delta = candidate_cost - current_cost

        if delta < 0 or random.uniform(0, 1) < math.exp(-delta / T):
            current = candidate
            current_cost = candidate_cost
            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

        T *= alpha

        if i % 1000 == 0:
            print(f"Iteracja {i}: Najlepszy dystans = {best_cost:.2f}")

    return best, best_cost

# --- RUN --- #
if __name__ == "__main__":
    cities = generate_cities(30)
    transitions = generate_positive_transitions(cities, num_transitions=100)

    if not transitions:
        print("⚠️ Nie udało się wygenerować przejść pogarszających.")
    else:
        chi_0 = 0.8
        T0 = compute_initial_temperature(transitions, chi_0=chi_0)
        print(f"\n🎯 Obliczona temperatura początkowa T₀ = {T0:.4f} dla χ₀ = {chi_0}\n")

        best_tour, best_cost = simulated_annealing_tsp(cities, T0)
        print(f"\n✅ Ostateczny wynik: Najlepszy dystans = {best_cost:.2f}")
        print("🧭 Kolejność odwiedzania miast:", best_tour)
