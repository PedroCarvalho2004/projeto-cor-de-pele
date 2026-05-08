
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# ==========================================
# 1. DOWNLOAD DOS DADOS REAIS (OMLC)
# ==========================================
print("Descarregando dados reais da Bilirrubina (OMLC)...")
url_bili = "https://omlc.org/spectra/PhotochemCAD/data/041-abs.txt"
req = urllib.request.Request(url_bili, headers={'User-Agent': 'Mozilla/5.0'})

l_real, abs_real = [], []
try:
    texto = urllib.request.urlopen(req).read().decode('utf-8')
    for linha in texto.split('\n'):
        partes = linha.strip().split()
        if len(partes) >= 2:
            try:
                wv = float(partes[0])
                ab = float(partes[1])
                l_real.append(wv)
                abs_real.append(ab)
            except ValueError:
                pass
    print(f"Sucesso! {len(l_real)} pontos reais carregados.")
except Exception as e:
    print(f"Erro ao baixar os dados: {e}")


Lambdas = np.linspace(380, 780, 400)

def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

# ==========================================
# 2. FUNÇÃO DO CAROTENO
# ==========================================


 
y_caro =gauss(Lambdas, 130000, 449, 40) + gauss(Lambdas, -20000, 520, 30)

fig, ax = plt.subplots(figsize=(8, 5))
fig.canvas.manager.set_window_title('Espectro Estático: Caroteno')

if len(l_real) > 0:
    ax.plot(l_real, abs_real, color='lightgray', linewidth=8, label='Dados Reais (OMLC)')

ax.plot(Lambdas, y_caro, color='darkorange', linewidth=3, label="Fitting do caroteno")
ax.set_title('Caroteno ($\mu_a$)', fontweight='bold', fontsize=14)
ax.set_xlabel('Comprimento de Onda (nm)', fontsize=12)
ax.set_ylabel('Absorção', fontsize=12)
ax.set_xlim(380, 780)
ax.set_ylim(0, 300000)
ax.grid(True, ls="--", alpha=0.5)

plt.show()
ax.grid(True, ls="--", alpha=0.5)

plt.show()