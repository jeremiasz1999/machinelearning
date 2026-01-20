import random
import math
import csv
#import time
from tqdm import tqdm
#import matplotlib.pyplot as plt
#from random import sample
import numpy as np
from itertools import product
import io
"""zmienne globalne"""
chi_0 = 0.87
total_distance_calls = 0
Tmin = 1e-6


def load_cities_from_csv(filename="generated_cities.csv"):
    """
    Wczytuje współrzędne miast z pliku CSV.

    :param filename: Nazwa pliku CSV.
    :return: Lista miast (każde miasto to krotka (x, y)).
    """
    cities = []
    try:
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Pominięcie nagłówka
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


def total_distance(tour, cities_arr):
    global total_distance_calls
    total_distance_calls += 1
    pts = cities_arr[tour]                    # indeksowanie tablicy NumPy listą/krotką indeksów
    next_pts = np.roll(pts, -1, axis=0)       # przesunięcie o jeden w lewo
    return np.sum(np.hypot(next_pts[:, 0] - pts[:, 0],
                           next_pts[:, 1] - pts[:, 1]))

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

# --- Transition Sampling --- #
def generate_positive_transitions(cities, num_transitions=500):
    transitions = []
    for _ in range(num_transitions):
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

def collect_transitions_at_T1(cities, T1, plateau_iters=500):
    current = list(range(len(cities)))
    random.shuffle(current)
    current_cost = total_distance(current, cities)
    transitions = []

    for _ in range(plateau_iters):
        candidate = two_opt_swap(current)
        candidate_cost = total_distance(candidate, cities)
        delta = candidate_cost - current_cost
        if delta > 0:
            transitions.append((current_cost, candidate_cost))
        if delta < 0 or random.random() < math.exp(-delta / T1):
            current = candidate
            current_cost = candidate_cost
    return transitions

# --- Initial Temperature Estimation --- #
def estimate_acceptance(transitions, T):
    numerator = sum(math.exp(-Emax / T) for (_, Emax) in transitions)
    denominator = sum(math.exp(-Emin / T) for (Emin, _) in transitions)
    return numerator / denominator

def compute_initial_temperature(transitions, chi_0 = chi_0, p=1, epsilon=1e-2, T1=None):
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
            print("Zbyt wiele iteracji")
            break

    return Tn

# --- Adaptive Sampling --- #
def adaptive_temperature_estimation(cities, chi_0 = chi_0, epsilon_T= 2, max_steps=9):
    previous_T = None
    transitions_count = 500

    for step in range(max_steps):
        transitions = generate_positive_transitions(cities, num_transitions=transitions_count)
        if not transitions:
            print(" Brak pogarszających przejść.")
            return None

        T = compute_initial_temperature(transitions, chi_0=chi_0)
        print(f" Próba {step+1} ({transitions_count} przejść): T₀ = {T:.4f}")

        if previous_T is not None and abs(T - previous_T) < epsilon_T:
            return T

        previous_T = T
        transitions_count *= 2

    print("Temperatura się nie ustabilizowała.")
    return previous_T


def simulated_annealing_tsp(cities, T0, alpha= 0.98, iterations=10000):
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

        if delta < 0:
            accept = True
        else:
            exponent = -delta / T
            if exponent < -700:
                accept = False
            else:
                accept = random.random() < math.exp(exponent)

        if accept:
            current = candidate
            current_cost = candidate_cost
            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

        T *= alpha

        #if i % 1000 == 0:
            #print(f"Iteracja {i}: Najlepszy dystans = {best_cost:.2f}")

    return best, best_cost

# --- RUN --- #
if __name__ == "__main__":
    cities = load_cities_from_csv("../generated_cities.csv")
    if cities is None:
        print("Nie udało się wczytać miast.")
        exit()

    cities = np.array(cities)
    alphas = np.round(np.arange(0.80, 1.00, 0.01), 2)
    chi0s = np.round(np.arange(0.80, 1.00, 0.01), 2)
    runs_per_pair = 500
    results = []

    total_combinations = len(alphas) * len(chi0s)
    #print(
    #    f"Plan: {len(alphas)} alfa × {len(chi0s)} chi0 × {runs_per_pair} uruchomień = {total_combinations * runs_per_pair} SA")

    for alpha, chi0 in tqdm(product(alphas, chi0s), total=total_combinations, desc="Eksperyment"):
            #print(f"\n[Alpha={alpha:.2f}, Chi0={chi0:.2f}]")

            # === Krok 1: Estymacja T0 dla danego chi0 ===
            reset_total_distance_calls()
            T1 = adaptive_temperature_estimation(cities, chi_0=chi0)
            if T1 is None:
                #print("  Pomijam — nie udało się obliczyć T1.")
                continue

            plateau_transitions = collect_transitions_at_T1(cities, T1, plateau_iters=500)
            if not plateau_transitions:
                #print("  Brak przejść na plateau — pomijam.")
                continue

            T0 = compute_initial_temperature(plateau_transitions, chi_0=chi0)
            calls_for_T0 = get_total_distance_calls()  # koszt estymacji T0

            # === Krok 2: 500 uruchomień SA ===
            costs = []
            total_calls_list = []

            for run in range(runs_per_pair):
                reset_total_distance_calls()
                _, best_cost = simulated_annealing_tsp(cities, T0, alpha=alpha, iterations=10000)
                calls_in_sa = get_total_distance_calls()
                total_calls = calls_for_T0 + calls_in_sa
                costs.append(best_cost)
                total_calls_list.append(total_calls)


            # === Statystyki ===
            costs_arr = np.array(costs)
            calls_arr = np.array(total_calls_list)
            mean_cost = np.mean(costs_arr)
            std_cost = np.std(costs_arr)
            min_cost = np.min(costs_arr)
            max_cost = np.max(costs_arr)
            min_cost_count = np.sum(costs_arr == min_cost)
            mask = (costs_arr == min_cost)
            sum_calls_for_min_cost = np.sum(calls_arr[mask])  # <-- to jest nowy wskaźnik
            mean_calls = np.mean(calls_arr)
            q10 = np.percentile(costs_arr, 10)
            q50 = np.percentile(costs_arr, 50)
            q90 = np.percentile(costs_arr, 90)
            epsilon = 0.01 * min_cost
            near_opt_count = np.sum(costs_arr <= min_cost + epsilon)
            if len(costs_arr) > 1:
                corr = np.corrcoef(costs_arr, calls_arr)[0, 1]
            else:
                corr = np.nan
            hist, _ = np.histogram(costs_arr, bins=20, density=True)
            entropy = -np.sum(hist * np.log(hist + 1e-12))

            results.append({
                "alpha": round(alpha, 2),
                "chi0": round(chi0, 2),
                "mean_cost": mean_cost,
                "std_cost": std_cost,
                "min_cost": min_cost,
                "min_cost_count": int(min_cost_count),
                "sum_calls_for_min_cost": int(sum_calls_for_min_cost),  # <-- nowe pole
                "max_cost": max_cost,
                "mean_total_calls": mean_calls,
                "q10_cost": q10,
                "median_cost": q50,
                "q90_cost": q90,
                "near_opt_count": int(near_opt_count),
                "cost_call_corr": corr,
                "entropy": entropy,
            })

            #print(f"  Wynik: średnia = {mean_cost:.2f}, koszt = {mean_calls:.0f} wywołań")

    # === Zapis do CSV ===
    output_file: io.TextIOWrapper
    with open("grid_search_alpha_chi0_500runs.csv", "w", newline="", encoding="utf-8") as output_file:
        fieldnames = [
            "alpha", "chi0", "mean_cost", "std_cost", "min_cost",
            "min_cost_count", "sum_calls_for_min_cost", "max_cost", "mean_total_calls",
            "q10_cost", "median_cost", "q90_cost", "near_opt_count", "cost_call_corr",
            "entropy",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n Eksperyment zakończony. Wyniki zapisane do 'grid_search_alpha_chi0_500runs.csv'")



