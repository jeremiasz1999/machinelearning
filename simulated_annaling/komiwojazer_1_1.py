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



def simulated_annealing(cities, start_temp=10000, min_temp=1e-4, alpha=0.995, iterations_per_temp=100):
    current_route = list(range(len(cities)))
    random.shuffle(current_route)
    best_route = current_route.copy()
    best_cost = total_distance(current_route, cities)

    temperature = start_temp
    iteration_counter = 0

    while temperature > min_temp:
        for i in range(iterations_per_temp):
            new_route = two_opt_swap(current_route)
            current_cost = total_distance(current_route, cities)
            new_cost = total_distance(new_route, cities)
            delta = new_cost - current_cost


            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_route = new_route
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_route = new_route

            iteration_counter += 1

            if iteration_counter % 50000 == 0:
                print(f"\nPo {iteration_counter} iteracjach:")
                print(f"Długość trasy: {round(best_cost, 2)}\n")

        temperature *= alpha

    return best_route, best_cost



if __name__ == '__main__':
    cities = load_cities_from_csv("../generated_cities.csv")
    if cities is None:
        print(" Nie udało się wczytać miast. Sprawdź plik 'generated_cities.csv'.")
        exit()

    best_route, cost = simulated_annealing(cities)
    print("\nOstateczny wynik:")
    print("Najlepsza trasa:", best_route)
    print("Długość trasy:", round(cost, 2))
