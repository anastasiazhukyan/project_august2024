import numpy as np
import matplotlib.pyplot as plt

def B_field(x, z, dps, mu_0=1):
    Bx, Bz = np.zeros_like(x), np.zeros_like(z)

    for (dx, dz, mx, mz) in dps:
        rx = x - dx
        rz = z - dz
        r = np.sqrt(rx ** 2 + rz ** 2 + 1e-9)
        r_hx = rx / r
        r_hz = rz / r
        dot_mr = mx * r_hx + mz * r_hz

        Bx += mu_0 * (3 * dot_mr * r_hx - mx) /r ** 3
        Bz += mu_0 * (3 * dot_mr * r_hz - mz) / r**3

    return Bx, Bz

def make_dps(L, d, dx, orient):
    dps = []
    half_L = L / 2
    num_dps = int(L / dx)
    for i in range(num_dps):
        for j in range(num_dps):
            x_pos = i * dx - half_L
            z_pos = j * dx - half_L
            if orient == 'x':
                dps.append((x_pos, z_pos, 1, 0))
            elif orient == 'z':
                dps.append((x_pos, z_pos, 0, 1))
    return dps

L = 5.0
d = 1.0
dx = 0.1

x, z = np.meshgrid(np.arange(-3, 3, 0.1), np.arange(-3, 3, 0.1))

dps_x = make_dps(L, d, dx, orient='x')
Bx_x, Bz_x = B_field(x, z, dps_x)

dps_z = make_dps(L, d, dx, orient='z')
Bx_z, Bz_z = B_field(x, z, dps_z)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].quiver(x, z, Bx_x, Bz_x, color='b')
axs[0, 0].set_title('M вдоль оси x')
axs[0, 0].set_xlabel('x (м)')
axs[0, 0].set_ylabel('z (м)')

c1 = axs[0, 1].streamplot(x, z, Bx_x, Bz_x, color=np.sqrt(Bx_x**2 + Bz_x**2), linewidth=1, cmap='viridis')
axs[0, 1].set_title('M вдоль оси x')
axs[0, 1].set_xlabel('x (м)')
axs[0, 1].set_ylabel('z (м)')
fig.colorbar(c1.lines, ax=axs[0, 1], label='|B| (Тесла)')

axs[1, 0].quiver(x, z, Bx_z, Bz_z, color='r')
axs[1, 0].set_title('M вдоль оси z')
axs[1, 0].set_xlabel('x (м)')
axs[1, 0].set_ylabel('z (м)')

c2 = axs[1, 1].streamplot(x, z, Bx_z, Bz_z, color=np.sqrt(Bx_z**2 + Bz_z**2), linewidth=1, cmap='plasma')
axs[1, 1].set_title('M вдоль оси z')
axs[1, 1].set_xlabel('x (м)')
axs[1, 1].set_ylabel('z (м)')
fig.colorbar(c2.lines, ax=axs[1, 1], label='|B| (Тесла)')

plt.tight_layout()
plt.show()
