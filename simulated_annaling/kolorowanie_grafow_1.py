import random
import math

"""
Symulowane wyżarzanie dla problemu kolorowania grafu.

:param graph: Słownik reprezentujący graf, gdzie klucz to wierzchołek,
              a wartość to lista sąsiadów.
:param max_iterations: Maksymalna liczba iteracji.
:param initial_temp: Początkowa temperatura.
:param cooling_rate: Współczynnik chłodzenia (0 < cooling_rate < 1).
:return: Słownik z przypisanymi kolorami do wierzchołków i liczbą użytych kolorów.
"""

def simulated_annealing_coloring(graph, max_iterations=10000, initial_temp=1000, cooling_rate=0.99):

    # Inicjalizacja
    vertices = list(graph.keys())
    num_vertices = len(vertices)

    # Przypisz początkowe kolory losowo
    coloring = {v: random.randint(1, num_vertices) for v in vertices}

    # Liczba użytych kolorów
    def count_colors(coloring):
        return len(set(coloring.values()))

    # Funkcja oceny (liczba konfliktów - sąsiednich wierzchołków o tym samym kolorze)
    def evaluate(coloring):
        conflicts = 0
        for vertex, neighbors in graph.items():
            for neighbor in neighbors:
                if coloring[vertex] == coloring[neighbor]:
                    conflicts += 1
        return conflicts // 2  # Każda krawędź jest liczona dwukrotnie

    # Główna pętla symulowanego wyżarzania
    current_temp = initial_temp
    current_conflicts = evaluate(coloring)

    best_coloring = coloring.copy()
    best_conflicts = current_conflicts

    for iteration in range(max_iterations):
        # Wybierz losowy wierzchołek i zmień jego kolor
        vertex = random.choice(vertices)
        old_color = coloring[vertex]
        new_color = random.randint(1, num_vertices)
        coloring[vertex] = new_color

        # Ocena nowego rozwiązania
        new_conflicts = evaluate(coloring)
        delta = new_conflicts - current_conflicts

        # Jeśli nowe rozwiązanie jest lepsze, zaakceptuj je
        if delta < 0 or random.random() < math.exp(-delta / current_temp):
            current_conflicts = new_conflicts
        else:
            # Odrzuć zmianę
            coloring[vertex] = old_color

        # Aktualizacja najlepszego rozwiązania
        if current_conflicts < best_conflicts:
            best_coloring = coloring.copy()
            best_conflicts = current_conflicts

        # Zmniejszenie temperatury
        current_temp *= cooling_rate

        # Warunek stopu
        if best_conflicts == 0:
            break

    # Minimalizacja liczby kolorów
    used_colors = set(best_coloring.values())
    color_map = {color: idx + 1 for idx, color in enumerate(sorted(used_colors))}
    minimized_coloring = {v: color_map[c] for v, c in best_coloring.items()}

    return minimized_coloring, len(used_colors)


# Przykład użycia
if __name__ == "__main__":
    # Reprezentacja grafu jako słownik sąsiedztwa
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'C', 'D'],
        'C': ['A', 'B', 'D'],
        'D': ['B', 'C']
    }

    coloring, num_colors = simulated_annealing_coloring(graph)
    print("Kolory wierzchołków:", coloring)
    print("Liczba użytych kolorów:", num_colors)