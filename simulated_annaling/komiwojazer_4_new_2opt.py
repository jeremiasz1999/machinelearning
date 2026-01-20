import random
import math
import csv
import matplotlib.pyplot as plt
from random import sample
chi_0 = 0.87
Tmin = 1e-6

# --- TSP Helpers --- #
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

def total_distance(tour, cities):
    dist = 0
    for i in range(len(tour)):
        x1, y1 = cities[tour[i]]
        x2, y2 = cities[tour[(i + 1) % len(tour)]]
        dist += math.hypot(x2 - x1, y2 - y1)
    return dist

def two_opt_swap(tour):
    n = len(tour)
    if n < 4:
        return tour[:]


    while True:
        a, b = sorted(random.sample(range(n), 2))
        if b > a:
            break


    new_tour = tour[:a] + tour[a:b+1][::-1] + tour[b+1:]
    return new_tour

# --- Transition Sampling --- #
def generate_positive_transitions(cities, num_transitions=500):
    """
    Generuje losowe trasy i wybiera tylko te przejścia, które pogarszają koszt.
    """
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
    """
    Zbiera przejścia pogarszające podczas plateau na temperaturze T1.
    """
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
            print("Zbyt wiele iteracji – możliwa niestabilność")
            break

    return Tn

# --- Adaptive Sampling --- #
def adaptive_temperature_estimation(cities, chi_0 = chi_0, epsilon_T= 1, max_steps=9):
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

# --- Simulated Annealing --- #
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

        if delta < 0 or random.random() < math.exp(-delta / T):
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
    cities = load_cities_from_csv("../generated_cities.csv")
    if cities is None:
        print("Nie udało się wczytać miast. Sprawdź plik 'generated_cities.csv'.")
        exit()

    T1 = adaptive_temperature_estimation(cities, chi_0=chi_0)

    print(f"\nObliczona temperatura początkowa T1 = {T1:.4f} dla χ₀ = {chi_0}\n")

    # Krok 2: zbierz lepsze przejścia podczas plateau w T1
    plateau_transitions = collect_transitions_at_T1(cities, T1, plateau_iters=500)
    if not plateau_transitions:
        print("Nie udało się wygenerować przejść pogarszających na plateau.")
        exit()

    # Krok 3: oblicz dokładniejsze T0 z użyciem zebranych przejść
    T0 = compute_initial_temperature(plateau_transitions, chi_0=chi_0)
    print(f"\nObliczona temperatura początkowa T0 = {T0:.4f} dla χ₀ = {chi_0}\n")



    # 4. Główna faza wyżarzania
    best_tour, best_cost = simulated_annealing_tsp(cities, T0)
    print(f"\nOstateczny wynik: Najlepszy dystans = {best_cost:.2f}")
    print("Kolejność odwiedzania miast:", best_tour)



    # --- Wielokrotne uruchomienie SA w celu analizy statystycznej --- #
    print("\nUruchamiam 10 powtórzeń algorytmu...")

    results = []
    for i in range(500):
        _, cost = simulated_annealing_tsp(cities, T0)
        results.append(cost)

    best_found = min(results)
    count_best = results.count(best_found)

    print(f"\nNajlepszy znaleziony koszt: {best_found:.2f}")
    print(f" Liczba wystąpień tego kosztu w 10 powtórzeniach: {count_best} ({count_best / 10:.2%})")

    # Histogram wyników
    plt.figure(figsize=(10, 6))
    plt.hist(results, bins=20, color='skyblue', edgecolor='black')
    plt.title("Rozkład końcowych kosztów po 200 uruchomieniach SA")
    plt.xlabel("Całkowity dystans")
    plt.ylabel("Liczba wystąpień")
    plt.grid(True)
    plt.axvline(best_found, color='red', linestyle='dashed', linewidth=1.5, label=f"Minimum: {best_found:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.show()
