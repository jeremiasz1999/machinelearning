import random
import math

def simulated_annealing_pcb_3d(elements, connections, board_dimensions, max_iterations=10000, initial_temp=1000, cooling_rate=0.99):
    """
    Symulowane wyżarzanie dla optymalizacji układu elementów na płytce PCB w 3D.

    :param elements: Lista elementów, gdzie każdy element jest reprezentowany jako [x, y, z].
    :param connections: Lista par połączeń między elementami, np. [(0, 1), (1, 2)].
    :param board_dimensions: Wymiary płytki PCB jako (max_x, max_y, max_z).
    :param max_iterations: Maksymalna liczba iteracji.
    :param initial_temp: Początkowa temperatura.
    :param cooling_rate: Współczynnik chłodzenia (0 < cooling_rate < 1).
    :return: Optymalny układ elementów i całkowita długość połączeń.
    """
    # Inicjalizacja
    def calculate_total_length(positions, connections):
        """Oblicza całkowitą długość połączeń."""
        total_length = 0
        for a, b in connections:
            dist = math.sqrt(
                (positions[a][0] - positions[b][0]) ** 2 +
                (positions[a][1] - positions[b][1]) ** 2 +
                (positions[a][2] - positions[b][2]) ** 2
            )
            total_length += dist
        return total_length

    def is_valid_position(position, board_dimensions):
        """Sprawdza, czy pozycja mieści się w granicach płytki."""
        x, y, z = position
        max_x, max_y, max_z = board_dimensions
        return 0 <= x <= max_x and 0 <= y <= max_y and 0 <= z <= max_z

    def evaluate(positions, connections):
        """Funkcja oceny: suma długości połączeń."""
        return calculate_total_length(positions, connections)

    # Rozpoczęcie od losowego układu
    current_positions = [random_position(board_dimensions) for _ in elements]
    current_score = evaluate(current_positions, connections)

    best_positions = current_positions.copy()
    best_score = current_score

    current_temp = initial_temp

    # Główna pętla symulowanego wyżarzania
    for iteration in range(max_iterations):
        # Wybierz losowy element do przemieszczenia
        idx = random.randint(0, len(elements) - 1)
        old_position = current_positions[idx]
        new_position = perturb_position(old_position, board_dimensions)

        # Oblicz nową wartość funkcji oceny
        current_positions[idx] = new_position
        new_score = evaluate(current_positions, connections)
        delta = new_score - current_score

        # Akceptacja nowego rozwiązania
        if delta < 0 or random.random() < math.exp(-delta / current_temp):
            current_score = new_score
        else:
            # Odrzuć zmianę
            current_positions[idx] = old_position

        # Aktualizacja najlepszego rozwiązania
        if current_score < best_score:
            best_positions = current_positions.copy()
            best_score = current_score

        # Zmniejszenie temperatury
        current_temp *= cooling_rate

        # Warunek stopu
        if current_temp < 1e-5:
            break

    return best_positions, best_score


def random_position(board_dimensions):
    """Generuje losową pozycję w granicach płytki."""
    max_x, max_y, max_z = board_dimensions
    return [
        random.uniform(0, max_x),
        random.uniform(0, max_y),
        random.uniform(0, max_z)
    ]


def perturb_position(position, board_dimensions):
    """Przesuwa pozycję o niewielką losową wartość."""
    max_x, max_y, max_z = board_dimensions
    perturbation_range = 0.1  # Zakres przesunięcia
    new_position = [
        min(max(0, position[0] + random.uniform(-perturbation_range, perturbation_range)), max_x),
        min(max(0, position[1] + random.uniform(-perturbation_range, perturbation_range)), max_y),
        min(max(0, position[2] + random.uniform(-perturbation_range, perturbation_range)), max_z)
    ]
    return new_position


# Przykład użycia
if __name__ == "__main__":
    # Definicja elementów i połączeń
    num_elements = 5
    elements = list(range(num_elements))  # Elementy oznaczone jako 0, 1, 2, ...
    connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]  # Połączenia między elementami

    # Wymiary płytki PCB
    board_dimensions = (10, 10, 2)  # (max_x, max_y, max_z)

    # Uruchomienie algorytmu
    best_positions, best_score = simulated_annealing_pcb_3d(elements, connections, board_dimensions)

    print("Optymalne pozycje elementów:")
    for i, pos in enumerate(best_positions):
        print(f"Element {i}: {pos}")
    print("Całkowita długość połączeń:", best_score)