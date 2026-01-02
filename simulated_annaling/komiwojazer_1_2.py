import csv
import math
import random
import matplotlib.pyplot as plt
from random import sample


def load_cities_from_csv(filename="generated_cities.csv"):
    cities = []
    try:
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                _, x, y = row
                cities.append((float(x), float(y)))
    except:
        return None
    return cities


def total_distance(tour, cities):
    dist = 0.0
    for i in range(len(tour)):
        x1, y1 = cities[tour[i]]
        x2, y2 = cities[tour[(i + 1) % len(tour)]]
        dist += math.hypot(x2 - x1, y2 - y1)
    return dist


def two_opt_gen(tour):
    """
    Generator sąsiedztwa 2-OPT — zwraca wszystkie możliwe permutacje
    powstałe przez odwrócenie segmentu między dwoma punktami.
    """
    n = len(tour)
    i_range = range(2, n)
    for i in sample(i_range, len(i_range)):
        j_range = range(i + 1, n + 1)
        for j in sample(j_range, len(j_range)):
            xn = tour.copy()
            xn = xn[: i - 1] + list(reversed(xn[i - 1 : j])) + xn[j:]
            yield xn

def random_two_opt_neighbor(tour):
    """
    Losowo wybiera jednego sąsiada z generatora two_opt_gen().
    """
    neighbors = list(two_opt_gen(tour))
    if not neighbors:
        return tour[:]  # brak sąsiadów (np. krótka trasa)
    return random.choice(neighbors)


def simulated_annealing(cities, start_temp=10000, min_temp=1e-4, alpha=0.995, iterations_per_temp=100):
    current_route = list(range(len(cities)))
    random.shuffle(current_route)
    best_route = current_route.copy()
    best_cost = total_distance(current_route, cities)
    temperature = start_temp

    while temperature > min_temp:
        for _ in range(iterations_per_temp):
            new_route = random_two_opt_neighbor(current_route)
            current_cost = total_distance(current_route, cities)
            new_cost = total_distance(new_route, cities)
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_route = new_route
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_route = new_route
        temperature *= alpha
    return best_route, best_cost


if __name__ == '__main__':
    cities = load_cities_from_csv("../generated_cities.csv")
    if cities is None:
        exit()

    results = []
    num_runs = 200  # jasno zdefiniuj liczbę uruchomień

    print(f"Uruchamiam {num_runs} niezależnych prób algorytmu Symulowanego Wyżarzania...\n")

    for i in range(num_runs):
        _, cost = simulated_annealing(cities)
        results.append(cost)
        print(f"Próba {i + 1:3d}: długość trasy = {cost:.2f}")

    best_found = min(results)
    count_best = results.count(best_found)

    print(f"\nNajlepszy znaleziony wynik: {best_found:.2f} (uzyskany {count_best} razy)")

    # Histogram wyników
    plt.figure(figsize=(10, 6))
    plt.hist(results, bins=20, color='skyblue', edgecolor='black')
    plt.title(f"Rozkład końcowych kosztów po {num_runs} uruchomieniach SA")
    plt.xlabel("Całkowity dystans")
    plt.ylabel("Liczba wystąpień")
    plt.grid(True)
    plt.axvline(best_found, color='red', linestyle='dashed', linewidth=1.5, label=f"Minimum: {best_found:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.show()