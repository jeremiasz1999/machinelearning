import random
import csv

def generate_cities(n, grid_size=100):
    return [(random.uniform(0, grid_size), random.uniform(0, grid_size)) for _ in range(n)]

def save_cities_to_csv(cities, filename="cities.csv"):
    """
    Zapisuje współrzędne miast do pliku CSV.

    :param cities: Lista miast (każde miasto to krotka (x, y)).
    :param filename: Nazwa pliku CSV.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["City", "X", "Y"])  # Nagłówek
        for idx, (x, y) in enumerate(cities):
            writer.writerow([idx, x, y])  # Indeks miasta i jego współrzędne
    print(f"Miasta zostały zapisane do pliku: {filename}")

if __name__ == "__main__":
    cities = generate_cities(30)

 # Zapis miast do pliku CSV
    save_cities_to_csv(cities, filename="C:/Users/JEREMIASZ/PycharmProjects/machinelearning/generated_cities.csv")
