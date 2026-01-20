import csv
import math
import random
import numpy as np
from tqdm import tqdm
from itertools import product

"""zmienne globalne"""
total_distance_calls = 0

def load_cities_from_csv(filename="../generated_cities.csv"):
    cities = []
    try:
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                city_id, x, y = row
                cities.append((float(x), float(y)))
        print(f"Miasta wczytane z: {filename}")
    except FileNotFoundError:
        print(f"Plik {filename} nie znaleziony.")
        return None
    except Exception as e:
        print(f"Błąd wczytywania: {e}")
        return None
    return cities

def total_distance(tour, cities):
    global total_distance_calls
    total_distance_calls += 1
    dist = 0
    n = len(tour)
    for i in range(n):
        x1, y1 = cities[tour[i]]
        x2, y2 = cities[tour[(i + 1) % n]]
        dist += math.hypot(x2 - x1, y2 - y1)
    return dist

def get_total_distance_calls():
    return total_distance_calls

def reset_total_distance_calls():
    global total_distance_calls
    total_distance_calls = 0

def two_opt_swap(tour):
    n = len(tour)
    if n < 4:
        return tour[:]
    a, b = sorted(random.sample(range(n), 2))
    return tour[:a] + tour[a:b+1][::-1] + tour[b+1:]

def simulated_annealing(cities, start_temp=10000, min_temp=1e-4, alpha=0.995, iterations_per_temp=100):
    current_route = list(range(len(cities)))
    random.shuffle(current_route)
    best_route = current_route.copy()
    best_cost = total_distance(current_route, cities)
    temperature = start_temp

    while temperature > min_temp:
        for _ in range(iterations_per_temp):
            new_route = two_opt_swap(current_route)
            current_cost = total_distance(current_route, cities)
            new_cost = total_distance(new_route, cities)
            delta = new_cost - current_cost

            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_route = new_route
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_route = new_route.copy()

        temperature *= alpha

    return best_route, best_cost

def run_classic_sa_for_alpha(cities, alpha, runs=500):
    costs = []
    total_calls_list = []

    for _ in range(runs):
        reset_total_distance_calls()
        _, best_cost = simulated_annealing(
            cities,
            start_temp=10000,
            min_temp=1e-4,
            alpha=alpha,
            iterations_per_temp=100
        )
        calls = get_total_distance_calls()
        costs.append(best_cost)
        total_calls_list.append(calls)

    costs = np.array(costs)
    calls = np.array(total_calls_list)

    min_cost = np.min(costs)
    min_cost_count = np.sum(costs == min_cost)

    return {
        "alpha": round(alpha, 2),
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "min_cost": float(min_cost),
        "min_cost_count": int(min_cost_count),
        "mean_total_calls": float(np.mean(calls))
    }

# --- GŁÓWNY KOD ---
if __name__ == "__main__":
    cities = load_cities_from_csv()
    if cities is None:
        exit()

    # Zakres alpha: 0.80 do 0.99 z krokiem 0.01
    alphas = np.round(np.arange(0.80, 1.00, 0.01), 2)
    runs_per_alpha = 500
    results = []

    print(f"Testowanie klasycznego SA dla {len(alphas)} wartości alpha, po {runs_per_alpha} uruchomień.")

    for alpha in tqdm(alphas, desc="Klasyczny SA"):
        stats = run_classic_sa_for_alpha(cities, alpha, runs=runs_per_alpha)
        results.append(stats)

    # Zapis do CSV
    output_file = "classic_sa_alpha_sweep_500runs.csv"
    fieldnames = ["alpha", "mean_cost", "std_cost", "min_cost", "min_cost_count", "mean_total_calls"]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Wyniki zapisane do: {output_file}")