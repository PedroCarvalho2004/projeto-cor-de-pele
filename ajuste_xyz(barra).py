import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider, Button

# ==========================================
# 1. CONFIGURAÇÕES E FUNÇÕES MATEMÁTICAS
# ==========================================
nome_do_arquivo = 'CMF.csv' # <--- COLOQUE O NOME DO SEU ARQUIVO AQUI

def gauss_asym(x, amp, mu, sig_left, sig_right):
    sigma = np.where(x < mu, sig_left, sig_right)
    sigma = np.where(sigma == 0, 1e-6, sigma) 
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

def calcular_r2(y_real, y_pred):
    residuos = y_real - y_pred
    soma_quadrados_residuos = np.sum(residuos**2)
    soma_quadrados_total = np.sum((y_real - np.mean(y_real))**2)
    return 1 - (soma_quadrados_residuos / soma_quadrados_total) if soma_quadrados_total != 0 else 0

# ==========================================
# 2. CARREGAMENTO DINÂMICO DO CSV
# ==========================================
try:
    df = pd.read_csv(nome_do_arquivo)
    colunas_alvo = [c for c in df.columns if c.lower() != 'wavelength']
    idx_coluna = 0
    x_ref = df['Wavelength'].values
    y_ref_real = df[colunas_alvo[idx_coluna]].values
    print(f"Arquivo carregado! Colunas detectadas: {colunas_alvo}")
except:
    print("Erro ao carregar CSV. Gerando dados de teste...")
    x_ref = np.linspace(380, 780, 100)
    y_ref_real = gauss_asym(x_ref, 1, 550, 40, 40)
    colunas_alvo = ['Teste_Y']
    idx_coluna = 0

x_liso = np.linspace(380, 780, 500)

# ==========================================
# 3. INTERFACE GRÁFICA
# ==========================================
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(bottom=0.35, left=0.15)

# Plots iniciais
line_ref, = ax.plot(x_ref, y_ref_real, 'ok', alpha=0.2, label='Referência (CSV)')
line_model, = ax.plot(x_liso, gauss_asym(x_liso, 1, 550, 40, 40), 'r-', linewidth=2, label='Seu Ajuste')

ax.set_title(f"Ajustando: {colunas_alvo[idx_coluna]}")
ax.legend()
ax.grid(True, alpha=0.3)

# --- SLIDERS ---
color = 'lavender'
ax_amp = plt.axes([0.2, 0.22, 0.6, 0.03], facecolor=color)
ax_mu  = plt.axes([0.2, 0.17, 0.6, 0.03], facecolor=color)
ax_sl  = plt.axes([0.2, 0.12, 0.6, 0.03], facecolor=color)
ax_sr  = plt.axes([0.2, 0.07, 0.6, 0.03], facecolor=color)

s_amp = Slider(ax_amp, 'Amplitude', 0.0, 2.2, valinit=1.0)
s_mu  = Slider(ax_mu,  'Pico (mu)', 400, 700, valinit=550)
s_sl  = Slider(ax_sl,  'Sigma Esq', 2.0, 100, valinit=40)
s_sr  = Slider(ax_sr,  'Sigma Dir', 2.0, 100, valinit=40)

# --- BOTÕES ---
ax_next = plt.axes([0.82, 0.27, 0.12, 0.05])
btn_next = Button(ax_next, 'Próxima >>', color='lightgreen')

ax_save = plt.axes([0.02, 0.27, 0.12, 0.05])
btn_save = Button(ax_save, 'Salvar Log', color='lightblue')

# ==========================================
# 4. LÓGICA DE INTERAÇÃO
# ==========================================
def update(val):
    amp, mu, sl, sr = s_amp.val, s_mu.val, s_sl.val, s_sr.val
    line_model.set_ydata(gauss_asym(x_liso, amp, mu, sl, sr))
    
    y_pred = gauss_asym(x_ref, amp, mu, sl, sr)
    r2 = calcular_r2(y_ref_real, y_pred)
    ax.set_title(f"Ajustando: {colunas_alvo[idx_coluna]} | R²: {r2:.4f}")
    fig.canvas.draw_idle()

def mudar_curva(event):
    global idx_coluna, y_ref_real
    idx_coluna = (idx_coluna + 1) % len(colunas_alvo)
    y_ref_real = df[colunas_alvo[idx_coluna]].values
    line_ref.set_ydata(y_ref_real)
    ax.set_ylim(-0.1, max(y_ref_real) * 1.3)
    update(None)

def salvar_log(event):
    print("-" * 30)
    print(f"RESULTADO PARA: {colunas_alvo[idx_coluna]}")
    print(f"Amplitude: {s_amp.val:.4f}")
    print(f"Pico (mu): {s_mu.val:.4f}")
    print(f"Sigma Esq: {s_sl.val:.4f}")
    print(f"Sigma Dir: {s_sr.val:.4f}")
    print("-" * 30)

s_amp.on_changed(update)
s_mu.on_changed(update)
s_sl.on_changed(update)
s_sr.on_changed(update)
btn_next.on_clicked(mudar_curva)
btn_save.on_clicked(salvar_log)

plt.show()