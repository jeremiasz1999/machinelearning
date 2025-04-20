import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Wynikowe pozycje elementów (przykładowe dane)
positions = {
    0: [2.3, 4.1, 1.0],
    1: [3.5, 5.2, 1.2],
    2: [4.7, 6.3, 1.5],
    3: [5.8, 7.4, 1.8],
    4: [6.9, 8.5, 2.0]
}

# Połączenia między elementami
connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]

# Tworzenie wykresu 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Rysowanie połączeń
for a, b in connections:
    ax.plot(
        [positions[a][0], positions[b][0]],
        [positions[a][1], positions[b][1]],
        [positions[a][2], positions[b][2]],
        color="blue"
    )

# Rysowanie elementów
for idx, pos in positions.items():
    ax.scatter(pos[0], pos[1], pos[2], color="red")
    ax.text(pos[0], pos[1], pos[2], f" {idx}", color="black")

# Etykiety osi
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()