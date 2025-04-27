import csv
import math
import random
import matplotlib.pyplot as plt


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


# Oblicz odległość Euklidesową między dwoma miastami
def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# Oblicz całkowity koszt (długość) trasy
def route_cost(route, cities):
    total = 0.0
    for i in range(len(route)):
        total += distance(cities[route[i]], cities[route[(i + 1) % len(route)]])
    return total


# Wygeneruj sąsiada przez zamianę dwóch losowych miast w trasie
def generate_neighbor(route):
    a, b = random.sample(range(len(route)), 2)
    new_route = route.copy()
    new_route[a], new_route[b] = new_route[b], new_route[a]
    return new_route


# Algorytm symulowanego wyżarzania dla problemu komiwojażera
def simulated_annealing(cities, start_temp=10000, min_temp=1e-4, alpha=0.995, iterations_per_temp=100):
    current_route = list(range(len(cities)))
    random.shuffle(current_route)
    best_route = current_route.copy()
    best_cost = route_cost(current_route, cities)

    temperature = start_temp
    iteration_counter = 0  # Licznik iteracji

    while temperature > min_temp:
        for i in range(iterations_per_temp):
            new_route = generate_neighbor(current_route)
            current_cost = route_cost(current_route, cities)
            new_cost = route_cost(new_route, cities)
            delta = new_cost - current_cost

            # Akceptuj lepsze rozwiązania lub gorsze z pewnym prawdopodobieństwem
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_route = new_route
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_route = new_route

            # Zwiększ licznik iteracji
            iteration_counter += 1

            # Wyświetl najlepszą trasę co 5000 iteracji
            if iteration_counter % 50000 == 0:
                print(f"\nPo {iteration_counter} iteracjach:")
                print(f"Długość trasy: {round(best_cost, 2)}\n")

        # Schładzaj temperaturę
        temperature *= alpha

    return best_route, best_cost


# Główna funkcja programu
if __name__ == '__main__':
    cities = load_cities_from_csv("../generated_cities.csv")
    if cities is None:
        print(" Nie udało się wczytać miast. Sprawdź plik 'generated_cities.csv'.")
        exit()

    best_route, cost = simulated_annealing(cities)
    print("\nOstateczny wynik:")
    print("Najlepsza trasa:", best_route)
    print("Długość trasy:", round(cost, 2))

    results = []
    for i in range(200):
        _, cost = simulated_annealing(cities)
        results.append(cost)

    best_found = min(results)
    count_best = results.count(best_found)

    print(f"\n🏁 Najlepszy znaleziony koszt: {best_found:.2f}")
    print(f"🔁 Liczba wystąpień tego kosztu w 300 powtórzeniach: {count_best} ({count_best / 300:.2%})")

    # Histogram wyników
    plt.figure(figsize=(10, 6))
    plt.hist(results, bins=20, color='skyblue', edgecolor='black')
    plt.title("Rozkład końcowych kosztów po 300 uruchomieniach SA")
    plt.xlabel("Całkowity dystans")
    plt.ylabel("Liczba wystąpień")
    plt.grid(True)
    plt.axvline(best_found, color='red', linestyle='dashed', linewidth=1.5, label=f"Minimum: {best_found:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.show()