
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DEFINIÇÃO DO EIXO 
# ==========================================
Lambdas = np.linspace(380, 780, 400)

def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

# ==========================================
# 2. FUNÇÃO DO CAROTENO
# ==========================================
def modelar_caroteno(amp, mu_central, sigma):
    # Separa em dois picos próximos para formar a bimodalidade
    pico1 = gauss(Lambdas, amp, mu_central - 15, sigma * 0.7)
    pico2 = gauss(Lambdas, amp * 0.85, mu_central + 15, sigma * 0.7)
    return pico1 + pico2

# ==========================================
# 3. PARÂMETROS E PLOTAGEM ESTÁTICA
# ==========================================
amp_caro = 1.0
mu_caro = 465.0
sigma_caro = 30.0

y_caro = modelar_caroteno(amp_caro, mu_caro, sigma_caro)

fig, ax = plt.subplots(figsize=(8, 5))
fig.canvas.manager.set_window_title('Espectro Estático: Caroteno')

ax.plot(Lambdas, y_caro, color='darkorange', linewidth=3)
ax.set_title('Caroteno ($\mu_a$)', fontweight='bold', fontsize=14)
ax.set_xlabel('Comprimento de Onda (nm)', fontsize=12)
ax.set_ylabel('Absorção', fontsize=12)
ax.set_xlim(380, 780)
ax.set_ylim(0, 1.5)
ax.grid(True, ls="--", alpha=0.5)

plt.show()