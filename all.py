import numpy as np

mu_0 = 4 * np.pi * 1e-7
L = 5
d = 1


def field(m, r, dx):
    r_norm = np.linalg.norm(r)
    if r_norm == 0:
        return ([0, 0,0 ])
    return (mu_0 / (4 * np.pi)) * (3 * np.dot(m, r) * r / r_norm ** 5 - m / r_norm ** 3)


def field_energy(dx, M):
    Nx = int(L / dx)
    Ny = int(L / dx)
    Nz = int(d / dx)

    positions = [(i * dx, j * dx, k * dx) for i in range(Nx) for j in range(Ny) for k in range(Nz)]
    energy = 0.0

    for pos in positions:
        B_total = np.array([0.0, 0.0, 0.0])
        for pos2 in positions:
            if pos != pos2:
                r_vec = np.array(pos) - np.array(pos2)
                B_total += field(M * dx ** 3, r_vec, dx)

        energy += 0.5 * np.dot(B_total, M) * dx ** 3

    return energy


steps = [d, d / 2, d / 3, d / 4, d / 5]

for step in steps:
    print(f"Энергия для шага dx = {step}:")
    for M_direction in [[1, 0, 0], [0, 0, 1]]:
        M = np.array(M_direction, dtype=np.float64)
        energy = field_energy(step, M)
        print(f"  Ориентация M = {M}: {energy}")
