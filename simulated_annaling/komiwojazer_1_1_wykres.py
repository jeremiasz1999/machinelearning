import csv
import math
import random
import numpy as np
from tqdm import tqdm
import io

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

# --- GŁÓWNY KOD ---
if __name__ == "__main__":
    cities = load_cities_from_csv()
    if cities is None:
        exit()

    alphas = np.round(np.arange(0.80, 1.00, 0.01), 2)
    runs_per_alpha = 500
    results = []

    print(f"Testowanie klasycznego SA dla {len(alphas)} wartości alpha, po {runs_per_alpha} uruchomień.")

    for alpha in tqdm(alphas, desc="Klasyczny SA"):
        costs = []
        total_calls_list = []

        for _ in range(runs_per_alpha):
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

        # === Statystyki ===
        costs_arr = np.array(costs)
        calls_arr = np.array(total_calls_list)

        mean_cost = np.mean(costs_arr)
        std_cost = np.std(costs_arr)
        min_cost = np.min(costs_arr)
        max_cost = np.max(costs_arr)
        min_cost_count = np.sum(costs_arr == min_cost)
        mask = (costs_arr == min_cost)
        sum_calls_for_min_cost = np.sum(calls_arr[mask])
        mean_calls = np.mean(calls_arr)

        q10 = np.percentile(costs_arr, 10)
        q50 = np.percentile(costs_arr, 50)
        q90 = np.percentile(costs_arr, 90)

        epsilon = 0.01 * min_cost  # 1% od optimum
        near_opt_count = np.sum(costs_arr <= min_cost + epsilon)

        if len(costs_arr) > 1:
            corr = np.corrcoef(costs_arr, calls_arr)[0, 1]
        else:
            corr = np.nan

        hist, _ = np.histogram(costs_arr, bins=20, density=True)
        entropy = -np.sum(hist * np.log(hist + 1e-12))

        results.append({
            "alpha": round(alpha, 2),
            "mean_cost": float(mean_cost),
            "std_cost": float(std_cost),
            "min_cost": float(min_cost),
            "min_cost_count": int(min_cost_count),
            "sum_calls_for_min_cost": int(sum_calls_for_min_cost),
            "max_cost": float(max_cost),
            "mean_total_calls": float(mean_calls),
            "q10_cost": float(q10),
            "median_cost": float(q50),
            "q90_cost": float(q90),
            "near_opt_count": int(near_opt_count),
            "cost_call_corr": float(corr) if not np.isnan(corr) else None,
            "entropy": float(entropy),
        })

    # === Zapis do CSV ===
    output_file: io.TextIOWrapper
    with open("classic_sa_alpha_sweep_500runs.csv", "w", newline="", encoding="utf-8") as output_file:
        fieldnames = [
            "alpha", "mean_cost", "std_cost", "min_cost",
            "min_cost_count", "sum_calls_for_min_cost", "max_cost", "mean_total_calls",
            "q10_cost", "median_cost", "q90_cost", "near_opt_count", "cost_call_corr",
            "entropy"
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Wyniki zapisane do: classic_sa_alpha_sweep_500runs.csv")