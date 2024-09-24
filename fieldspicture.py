import numpy as np
import matplotlib.pyplot as plt

def delta(x, y, z, x0, y0, z0, epsilon=0.1):
    return np.where((np.abs(x - x0) < epsilon) &
                    (np.abs(y - y0) < epsilon) &
                    (np.abs(z - z0) < epsilon),
                    1 / (epsilon ** 3), 0)


def B_field(x, y, z, dps, mu_0=1, epsilon=0.1):
    Bx, By, Bz = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)

    for (dx, dy, dz, mx, my, mz) in dps:
        rx = x - dx
        ry = y - dy
        rz = z - dz
        r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)


        mask = (r > 0.1)

        r_hx = rx[mask] / r[mask]
        r_hy = ry[mask] / r[mask]
        r_hz = rz[mask] / r[mask]
        dot_mr = mx * r_hx + my * r_hy + mz * r_hz

        Bx[mask] += mu_0 * (3 * dot_mr * r_hx - mx) / r[mask] ** 3
        By[mask] += mu_0 * (3 * dot_mr * r_hy - my) / r[mask] ** 3
        Bz[mask] += mu_0 * (3 * dot_mr * r_hz - mz) / r[mask] ** 3

        delta_val = delta(x, y, z, dx, dy, dz, epsilon=epsilon)

        delta_mask = (np.abs(rx) < epsilon) & (np.abs(ry) < epsilon) & (np.abs(rz) < epsilon)
        Bx[delta_mask] += (8 * np.pi / 3) * mx * delta_val[delta_mask]
        By[delta_mask] += (8 * np.pi / 3) * my * delta_val[delta_mask]
        Bz[delta_mask] += (8 * np.pi / 3) * mz * delta_val[delta_mask]

    return Bx, By, Bz


def make_dps(L, d, dx, orient):
    dps = []
    half_L = L / 2
    half_d = d / 2
    num_dps_x = int(L / dx)
    num_dps_y = int(L / dx)
    num_dps_z = int(d / dx)


    for i in range(num_dps_x):
        for j in range(num_dps_y):
            for k in range(num_dps_z):
                x_pos = i * dx - half_L
                y_pos = j * dx - half_L
                z_pos = k * dx - half_d

                if orient == 'x':
                    dps.append((x_pos, y_pos, z_pos, 1, 0, 0))
                elif orient == 'y':
                    dps.append((x_pos, y_pos, z_pos, 0, 1, 0))
                elif orient == 'z':
                    dps.append((x_pos, y_pos, z_pos, 0, 0, 1))
    return dps


L = 5.0
d = 1.0
dx = 0.1


x, z = np.meshgrid(np.arange(-4, 4, 0.1), np.arange(-4, 4, 0.1))
y = 0


dps_x = make_dps(L, d, dx, orient='x')
Bx_x, By_x, Bz_x = B_field(x, y * np.ones_like(z), z, dps_x, epsilon=0.1)


dps_z = make_dps(L, d, dx, orient='z')
Bx_z, By_z, Bz_z = B_field(x, y * np.ones_like(z), z, dps_z, epsilon=0.1)


fig, axs = plt.subplots(2, 2, figsize=(12, 10))


axs[0, 0].quiver(x, z, Bx_x, Bz_x, color='b')
axs[0, 0].set_title('M вдоль оси x')
axs[0, 0].set_xlabel('x (м)')
axs[0, 0].set_ylabel('z (м)')

c1 = axs[0, 1].streamplot(x, z, Bx_x, Bz_x, color=np.sqrt(Bx_x ** 2 + Bz_x ** 2), linewidth=1, cmap='viridis')
axs[0, 1].set_title('M вдоль оси x')
axs[0, 1].set_xlabel('x (м)')
axs[0, 1].set_ylabel('z (м)')
fig.colorbar(c1.lines, ax=axs[0, 1], label='|B| (Тесла)')

axs[1, 0].quiver(x, z, Bx_z, Bz_z, color='r')
axs[1, 0].set_title('M вдоль оси z')
axs[1, 0].set_xlabel('x (м)')
axs[1, 0].set_ylabel('z (м)')

c2 = axs[1, 1].streamplot(x, z, Bx_z, Bz_z, color=np.sqrt(Bx_z ** 2 + Bz_z ** 2), linewidth=1, cmap='plasma')
axs[1, 1].set_title('M вдоль оси z')
axs[1, 1].set_xlabel('x (м)')
axs[1, 1].set_ylabel('z (м)')
fig.colorbar(c2.lines, ax=axs[1, 1], label='|B| (Тесла)')

plt.tight_layout()
plt.show()
