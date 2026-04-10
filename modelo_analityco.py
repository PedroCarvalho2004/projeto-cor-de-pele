
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches
import pandas as pd
from scipy.spatial import ConvexHull
import time

# ==========================================
# 1. DEFINIÇÃO DO EIXO E GAUSSIANAS
# ==========================================
Lambdas = np.linspace(380, 780, 400)
delta_lambda = Lambdas[1] - Lambdas[0]

def gauss_asym(x, amp, mu, sig_left, sig_right):
    sigma = np.where(x < mu, sig_left, sig_right)
    sigma = np.where(sigma == 0, 1e-6, sigma) 
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

# ==========================================
# 2. PROPRIEDADES ÓTICAS: PELE (HÍBRIDO)
# ==========================================
mu_a_hbo2 = (
    gauss(Lambdas, 2750.0, 415, 14) +   
    gauss(Lambdas, 270.0, 542, 11) +    
    gauss(Lambdas, 287.0, 577, 9) +     
    gauss(Lambdas, 70.0,  455, 70) +
    gauss(Lambdas, 4.0,   800, 70)     
)

mu_a_hb = (
    gauss(Lambdas, 2950.0, 434, 14) +   
    gauss(Lambdas, 255.0, 555, 30) +    
    gauss(Lambdas, 50.0,  500, 55) +
    gauss(Lambdas, 10.0,  620, 100)     
)
mu_a_base_derme = 0.008 * np.ones_like(Lambdas)

A_fit, B_fit, C_fit = 7.49, 2.18, 2.41
mu_a_epiderme_fitada = A_fit * (Lambdas/500.0)**(-B_fit) + C_fit

def calcular_espalhamento(S):
    a_cm, f_R, b_Mie = 43.6, 0.41, 0.562
    lambdas_norm = Lambdas / 500.0
    return S * a_cm * (f_R * (lambdas_norm)**(-4) + (1 - f_R) * (lambdas_norm)**(-b_Mie))

# ==========================================
# 3. COLORIMETRIA FÍSICA E ESPAÇO LAB
# ==========================================
T_ilum = 5000.0
c2 = 1.4388e7 
S_ilum = (Lambdas**-5) / (np.exp(c2 / (Lambdas * T_ilum)) - 1.0)

x_bar = (gauss_asym(Lambdas, 0.3483, 439.6, 17.5, 22.9) + 
           gauss_asym(Lambdas, 1.1560, 600.0, 36.5722, 31.8083))

# --- AJUSTE PARA Y_BARRA ---
y_bar = (gauss_asym(Lambdas, 1.0144, 553.8, 39.16, 49.63) )

# --- AJUSTE PARA Z_BARRA ---
z_bar = (gauss_asym(Lambdas, 1.9158, 442.0833, 18.8778, 28.5417))

k_norm = 100.0 / np.sum(S_ilum * y_bar * delta_lambda)

Xn = k_norm * np.sum(S_ilum * x_bar * delta_lambda)
Yn = k_norm * np.sum(S_ilum * y_bar * delta_lambda)
Zn = k_norm * np.sum(S_ilum * z_bar * delta_lambda)

def calcular_cor_XYZ(R_d):
    X = k_norm * np.sum(R_d * S_ilum * x_bar * delta_lambda)
    Y = k_norm * np.sum(R_d * S_ilum * y_bar * delta_lambda)
    Z = k_norm * np.sum(R_d * S_ilum * z_bar * delta_lambda)
    return X, Y, Z

def xyz_para_lab(X, Y, Z):
    def f(t):
        if t > (6.0/29.0)**3:
            return t**(1.0/3.0)
        else:
            return (1.0/3.0) * ((29.0/6.0)**2) * t + 4.0/29.0
            
    L = 116.0 * f(Y/Yn) - 16.0
    a = 500.0 * (f(X/Xn) - f(Y/Yn))
    b = 200.0 * (f(Y/Yn) - f(Z/Zn))
    return float(L), float(a), float(b)

def xyz_para_rgb_monitor(X, Y, Z):
    x, y, z = X/100.0, Y/100.0, Z/100.0
    r =  3.2406 * x - 1.5372 * y - 0.4986 * z
    g = -0.9689 * x + 1.8758 * y + 0.0415 * z
    b_val =  0.0557 * x - 0.2040 * y + 1.0570 * z
    r, g, b_val = np.clip([r, g, b_val], 0, 1)
    
    def gamma(c): return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (c ** (1/2.4)) - 0.055)
    return float(gamma(r)), float(gamma(g)), float(gamma(b_val))

def calcular_reflectancia_bicamada(peso_melanina, d_epi_mm, v_sangue, sat_o2, S=1.0):
    d_epi_cm = d_epi_mm / 10.0 
    mu_a_epi = peso_melanina * mu_a_epiderme_fitada
    T_epi_ida_volta = np.exp(-mu_a_epi * d_epi_cm)**2
    
    c_hbo2, c_hb = v_sangue * sat_o2, v_sangue * (1 - sat_o2)
    mu_a_derme = (c_hbo2 * mu_a_hbo2) + (c_hb * mu_a_hb) + mu_a_base_derme
    mu_s_total = calcular_espalhamento(S)
    
    mu_t_prime = mu_a_derme + mu_s_total
    a_prime = mu_s_total / mu_t_prime
    mu_eff = np.sqrt(3.0 * mu_a_derme * mu_t_prime)
    z0 = 1.0 / mu_t_prime
    D = 1.0 / (3.0 * mu_t_prime)
    A = 3.2
    zb = 2.0 * A * D
    
    R_derme = (a_prime / 2.0) * (np.exp(-mu_eff * z0) + np.exp(-mu_eff * (z0 + 2.0 * zb)))
    return T_epi_ida_volta * R_derme

# ==========================================
# 4. LEITURA DA BASE DE DADOS (Envelope Real)
# ==========================================
tem_dados_reais = False
try:
    nome_arquivo = 'Skin Database 1-92.xlsx'
    
    try:
        df_real = pd.read_excel(nome_arquivo, sheet_name='Database', header=1)
    except:
        df_real = pd.read_excel(nome_arquivo, header=1)
    
    df_real.columns = df_real.columns.astype(str).str.strip()
    df_validos = df_real[['L*', 'b*']].apply(pd.to_numeric, errors='coerce').dropna()
    
    L_real = df_validos['L*'].values
    b_real = df_validos['b*'].values
    
    if len(L_real) > 10:
        pontos_reais = np.column_stack((b_real, L_real))
        from scipy.spatial import ConvexHull
        envelope_real = ConvexHull(pontos_reais)
        tem_dados_reais = True
except Exception as e:
    pass

# ==========================================
# 5. CONFIGURAÇÃO DA INTERFACE GRÁFICA
# ==========================================
fig = plt.figure(figsize=(15, 8))
fig.canvas.manager.set_window_title('Simulador Analítico (Exploração de Espalhamento S)')
plt.subplots_adjust(bottom=0.45)

ax_grafico = plt.axes([0.05, 0.45, 0.25, 0.45])  
ax_cor = plt.axes([0.34, 0.70, 0.08, 0.20])      
ax_banana = plt.axes([0.48, 0.45, 0.48, 0.45])   

ax_cor.axis('off') 

peso_mel_init, d_epi_init, v_sangue_init, sat_o2_init, S_init = 1.0, 0.05, 0.02, 0.70, 1.0

R_init = calcular_reflectancia_bicamada(peso_mel_init, d_epi_init, v_sangue_init, sat_o2_init, S_init)
X_i, Y_i, Z_i = calcular_cor_XYZ(R_init)
L_i, a_i, b_i = xyz_para_lab(X_i, Y_i, Z_i)
rgb_init = xyz_para_rgb_monitor(X_i, Y_i, Z_i)

historico_b = [b_i]
historico_L = [L_i]

linha_grafico, = ax_grafico.plot(Lambdas, R_init, color='maroon', linewidth=2)
ax_grafico.set_title('Reflectância', fontweight='bold', fontsize=11)
ax_grafico.set_ylim(0, 1.0)
ax_grafico.grid(True, ls="--", alpha=0.4)

patch_cor = patches.Rectangle((0, 0), 1, 1, facecolor=rgb_init)
ax_cor.add_patch(patch_cor)
ax_cor.set_title('Cor (5000K)', fontweight='bold', fontsize=10)
texto_lab = fig.text(0.34, 0.50, f"L* = {L_i:.1f}\na* = {a_i:.1f}\nb* = {b_i:.1f}", fontsize=11, family='monospace', fontweight='bold')

# --- Painel 3: A Área da Banana (Sem as bolinhas) ---
if tem_dados_reais:
    vertices = envelope_real.vertices
    vertices_fechados = np.append(vertices, vertices[0])
    ax_banana.fill(pontos_reais[vertices_fechados, 0], pontos_reais[vertices_fechados, 1], color='lightgray', alpha=0.4, label='Envelope Real')
    ax_banana.plot(pontos_reais[vertices_fechados, 0], pontos_reais[vertices_fechados, 1], color='black', linestyle=':', linewidth=2)

rastro_banana, = ax_banana.plot(historico_b, historico_L, color='dimgray', linestyle='-', linewidth=2, alpha=0.7, zorder=4, label='Rastro')
cursor_banana, = ax_banana.plot(b_i, L_i, marker='X', color='black', markersize=16, markeredgecolor='white', label='Paciente Atual', zorder=5)

ax_banana.set_title('Gráfico de Banana Invertido (b* vs L*)', fontweight='bold', fontsize=14)
ax_banana.set_xlabel('b* (Azul -> Amarelo)', fontsize=12)
ax_banana.set_ylabel('L* (Luminosidade: Escuro -> Claro)', fontsize=12)
ax_banana.grid(True, ls="--", alpha=0.6)
ax_banana.legend(fontsize=10)

ax_banana.set_xlim(-5, 50)
ax_banana.set_ylim(0, 100)

# ==========================================
# 6. SLIDERS E ATUALIZAÇÃO
# ==========================================
ax_mel = plt.axes([0.15, 0.35, 0.70, 0.03])
ax_esp = plt.axes([0.15, 0.28, 0.70, 0.03])
ax_san = plt.axes([0.15, 0.21, 0.70, 0.03])
ax_sat = plt.axes([0.15, 0.14, 0.70, 0.03])
ax_S   = plt.axes([0.15, 0.07, 0.70, 0.03])

slider_mel = Slider(ax_mel, 'Peso Melanina', 0.1, 8.0, valinit=peso_mel_init, color='saddlebrown')
slider_esp = Slider(ax_esp, 'Espes. Epiderme (mm)', 0.007, 0.50, valinit=d_epi_init, color='grey')
slider_san = Slider(ax_san, 'Fraç. Sangue', 0.001, 0.15, valinit=v_sangue_init, color='red')
slider_sat = Slider(ax_sat, 'Saturação O2', 0.0, 1.0, valinit=sat_o2_init, color='blue')
slider_S   = Slider(ax_S,   'Espalhamento (S)', 0.1, 3.0, valinit=S_init, color='orange')

ultima_atualizacao = time.time()

def atualizar(val):
    global ultima_atualizacao
    agora = time.time()
    
    if agora - ultima_atualizacao < 0.05:
        return
    ultima_atualizacao = agora

    novo_R = calcular_reflectancia_bicamada(slider_mel.val, slider_esp.val, slider_san.val, slider_sat.val, slider_S.val)
    nX, nY, nZ = calcular_cor_XYZ(novo_R)
    nL, na, nb = xyz_para_lab(nX, nY, nZ)
    nRGB = xyz_para_rgb_monitor(nX, nY, nZ)
    
    linha_grafico.set_ydata(novo_R)
    patch_cor.set_facecolor(nRGB)
    texto_lab.set_text(f"L* = {nL:.1f}\na* = {na:.1f}\nb* = {nb:.1f}")
    
    if ((nb - historico_b[-1])**2 + (nL - historico_L[-1])**2)**0.5 > 0.5:
        historico_b.append(nb)
        historico_L.append(nL)
        if len(historico_b) > 200:
            historico_b.pop(0)
            historico_L.pop(0)
        rastro_banana.set_xdata(historico_b)
        rastro_banana.set_ydata(historico_L)
    
    cursor_banana.set_xdata([nb])
    cursor_banana.set_ydata([nL])
    
    fig.canvas.draw_idle()

slider_mel.on_changed(atualizar)
slider_esp.on_changed(atualizar)
slider_san.on_changed(atualizar)
slider_sat.on_changed(atualizar)
slider_S.on_changed(atualizar)

plt.show()  