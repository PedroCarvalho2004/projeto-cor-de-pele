
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# ==========================================
# 1. DOWNLOAD DOS DADOS REAIS (OMLC)
# ==========================================
print("Descarregando dados reais da Bilirrubina (OMLC)...")
url_bili = "https://omlc.org/spectra/PhotochemCAD/data/119-orig-abs.txt"
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

l_real = np.array(l_real)
abs_real = np.array(abs_real)

# Pega o valor máximo da curva real para referência
max_abs = np.max(abs_real) if len(abs_real) > 0 else 1.0

# ==========================================
# 2. O SEU MODELO MATEMÁTICO (ALTERE AQUI)
# ==========================================
Lambdas = np.linspace(380, 780, 400)

def gauss_asym(x, amp, mu, sig_left, sig_right):
    sigma = np.where(x < mu, sig_left, sig_right)
    sigma = np.where(sigma == 0, 1e-6, sigma) 
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)


# Calcula a curva do seu modelo
y_bili = gauss_asym(Lambdas, 1.85, 456, 43,20)

# ==========================================
# 3. PLOTAGEM ESTÁTICA
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
fig.canvas.manager.set_window_title('Modelagem Estática: Bilirrubina')

# Curva Real (fundo cinzento grosso)
if len(l_real) > 0:
    ax.plot(l_real, abs_real, color='lightgray', linewidth=8, label='Dados Reais (OMLC)')

# O seu Modelo Analítico (linha colorida e fina por cima)
ax.plot(Lambdas, y_bili, color='olive', linewidth=2, label='Seu Modelo Matemático')

ax.set_title('Bilirrubina: Real vs Matemática', fontweight='bold', fontsize=14)
ax.set_xlabel('Comprimento de Onda (nm)', fontsize=12)
ax.set_ylabel('Absorção', fontsize=12)

# Focamos o gráfico apenas na zona azul onde a bilirrubina absorve
ax.set_xlim(380, 600) 
ax.set_ylim(0, max_abs * 1.2)

ax.grid(True, ls="--", alpha=0.5)
ax.legend(fontsize=12)

plt.show()
# %%

