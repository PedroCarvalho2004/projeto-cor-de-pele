# %%
%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ==========================================
# 1. DEFINIÇÃO DO EIXO E GAUSSIANAS
# ==========================================
Lambdas = np.linspace(380, 780, 400)

def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

# ==========================================
# 2. PROPRIEDADES ÓTICAS: DERME E EPIDERME
# ==========================================
# --- Derme: Sangue (Modelos Analíticos Ajustados para Força Real) ---
mu_a_hbo2_analitico = (
    gauss(Lambdas, 2750.0, 415, 14) +   
    gauss(Lambdas, 270.0, 542, 11) +    
    gauss(Lambdas, 287.0, 577, 9) +     
    gauss(Lambdas, 70.0,  455,70) +
    # --- A correção do Vermelho ---
    gauss(Lambdas, 4.0,  800, 70)     # Cauda longa e baixa do oxigenado
)

mu_a_hb_analitico = (
    gauss(Lambdas, 2950.0, 434, 14) +   
    gauss(Lambdas, 255.0, 555, 30) +    
    gauss(Lambdas, 50.0,  500, 55) +
    # --- A correção do Vermelho ---
    gauss(Lambdas, 10.0,  620, 100)     # Cauda longa desoxigenada
)
mu_a_base_derme = 0.008 * np.ones_like(Lambdas)

# --- Epiderme: A Sua Equação Fittada (Modelo Empírico) ---
A_fit = 7.49 
B_fit = 4.18
C_fit = 2.41
mu_a_epiderme_fitada = A_fit * (Lambdas/500.0)**(-B_fit) + C_fit

# --- Espalhamento (Equação 12) ---
def calcular_espalhamento(S):
    a_cm, f_R, b_Mie = 43.6, 0.41, 0.562
    lambdas_norm = Lambdas / 500.0
    return S * a_cm * (f_R * (lambdas_norm)**(-4) + (1 - f_R) * (lambdas_norm)**(-b_Mie))

# ==========================================
# 3. MOTOR DO MODELO BICAMADA
# ==========================================
def calcular_reflectancia_bicamada(peso_melanina, d_epi_mm, v_sangue, sat_o2, S=1.0):
    
    # --- EPIDERME ---
    d_epi_cm = d_epi_mm / 10.0 
    mu_a_epi = peso_melanina * mu_a_epiderme_fitada
    T_epi_ida_volta = np.exp(-mu_a_epi * d_epi_cm)**2
    
    # --- DERME ---
    c_hbo2 = v_sangue * sat_o2
    c_hb = v_sangue * (1 - sat_o2)
    
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
# 4. CONFIGURAÇÃO DA INTERFACE GRÁFICA (GUI)
# ==========================================
fig, ax = plt.subplots(figsize=(11, 8))
plt.subplots_adjust(bottom=0.45) 

# Valores iniciais 
peso_mel_init = 1.0
d_epi_init = 0.05
v_sangue_init = 0.02
sat_o2_init = 0.70

# Desenhando a curva inicial
y_init = calcular_reflectancia_bicamada(peso_mel_init, d_epi_init, v_sangue_init, sat_o2_init)
linha_grafico, = ax.plot(Lambdas, y_init, color='maroon', linewidth=3)

ax.set_title('Simulador Analítico (Gaussianas Sangue + Melanina Fittada)', fontsize=14, fontweight='bold')
ax.set_xlabel('Comprimento de Onda (nm)', fontsize=12)
ax.set_ylabel('Reflectância Difusa Total ($R_d$)', fontsize=12)
ax.set_ylim(0, 1.0)
ax.grid(True, which="both", ls="--", alpha=0.4)

# ==========================================
# 5. CRIANDO OS SLIDERS DE CONTROLO
# ==========================================
ax_mel = plt.axes([0.15, 0.30, 0.70, 0.03])
ax_esp = plt.axes([0.15, 0.23, 0.70, 0.03])
ax_san = plt.axes([0.15, 0.16, 0.70, 0.03])
ax_sat = plt.axes([0.15, 0.09, 0.70, 0.03])

# O slider da melanina agora reflete o Peso (0.1 a 5.0) para bater com o modelo .txt
slider_mel = Slider(ax_mel, 'Peso Melanina', 0.1, 5.0, valinit=peso_mel_init, color='saddlebrown')
slider_esp = Slider(ax_esp, 'Espes. Epiderme (mm)', 0.007, 0.50, valinit=d_epi_init, color='grey')
slider_san = Slider(ax_san, 'Fraç. Sangue', 0.001, 0.15, valinit=v_sangue_init, color='red')
slider_sat = Slider(ax_sat, 'Saturação $O_2$', 0.0, 1.0, valinit=sat_o2_init, color='blue')

# ==========================================
# 6. FUNÇÃO DE ATUALIZAÇÃO EM TEMPO REAL
# ==========================================
def atualizar(val):
    novo_y = calcular_reflectancia_bicamada(slider_mel.val, slider_esp.val, slider_san.val, slider_sat.val)
    linha_grafico.set_ydata(novo_y)
    fig.canvas.draw_idle()

slider_mel.on_changed(atualizar)
slider_esp.on_changed(atualizar)
slider_san.on_changed(atualizar)
slider_sat.on_changed(atualizar)

plt.show()
# %%
# %%



























%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd

# ==========================================
# 1. CARREGAMENTO DOS DADOS REAIS E EIXO
# ==========================================
Lambdas = np.linspace(380, 780, 400)

try:
    # O skiprows=2 ignora as duas linhas de texto do cabeçalho do seu arquivo
    dados_sangue = np.loadtxt('sangue.txt', skiprows=2)
    lambdas_txt = dados_sangue[:, 0]
    eps_hbo2_txt = dados_sangue[:, 1]
    eps_hb_txt = dados_sangue[:, 2]
    print("Dados de sangue carregados com sucesso!")
except Exception as e:
    print(f"Erro ao carregar 'sangue.txt': {e}")
    # Vetores vazios caso o arquivo não seja encontrado
    lambdas_txt = Lambdas
    eps_hbo2_txt = np.zeros_like(Lambdas)
    eps_hb_txt = np.zeros_like(Lambdas)

# Interpolando os dados da tabela para os 400 pontos exatos do nosso eixo
eps_hbo2_interp = np.interp(Lambdas, lambdas_txt, eps_hbo2_txt)
eps_hb_interp = np.interp(Lambdas, lambdas_txt, eps_hb_txt)

# ==========================================
# 2. PROPRIEDADES ÓTICAS: DERME E EPIDERME
# ==========================================
# --- Conversão Física do Sangue ---
conc_molar_sangue = 150.0 / 64500.0  # ~0.00233 M (Mols por Litro de sangue)
fator_conversao = np.log(10) * conc_molar_sangue

mu_a_hbo2 = eps_hbo2_interp * fator_conversao
mu_a_hb = eps_hb_interp * fator_conversao

mu_a_base_derme = 0.008 * np.ones_like(Lambdas)

# --- Epiderme ---
# Utilizando a sua Equação Fittada do Artigo (Sem Gaussianas aqui também!)
A_fit = 7.49 
B_fit = 4.18
C_fit = 2.41
mu_a_epiderme_fitada = A_fit * (Lambdas/500.0)**(-B_fit) + C_fit

# --- Espalhamento ---
def calcular_espalhamento(S):
    a_cm, f_R, b_Mie = 43.6, 0.41, 0.562
    lambdas_norm = Lambdas / 500.0
    return S * a_cm * (f_R * (lambdas_norm)**(-4) + (1 - f_R) * (lambdas_norm)**(-b_Mie))

# ==========================================
# 3. MOTOR DO MODELO BICAMADA
# ==========================================
def calcular_reflectancia_bicamada(peso_melanina, d_epi_mm, v_sangue, sat_o2, S=1.0):
    
    # --- EPIDERME ---
    d_epi_cm = d_epi_mm / 10.0 
    mu_a_epi = peso_melanina * mu_a_epiderme_fitada
    T_epi_ida_volta = np.exp(-mu_a_epi * d_epi_cm)**2
    
    # --- DERME ---
    c_hbo2 = v_sangue * sat_o2
    c_hb = v_sangue * (1 - sat_o2)
    
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
# 4. CONFIGURAÇÃO DA INTERFACE GRÁFICA (GUI)
# ==========================================
fig, ax = plt.subplots(figsize=(11, 8))
plt.subplots_adjust(bottom=0.45) 

# Valores biológicos iniciais
peso_mel_init = 1.0
d_epi_init = 0.05
v_sangue_init = 0.02
sat_o2_init = 0.70

y_init = calcular_reflectancia_bicamada(peso_mel_init, d_epi_init, v_sangue_init, sat_o2_init)
linha_grafico, = ax.plot(Lambdas, y_init, color='maroon', linewidth=3)

ax.set_title('Simulador Ótico da Pele (Dados Empíricos Reais)', fontsize=14, fontweight='bold')
ax.set_xlabel('Comprimento de Onda (nm)', fontsize=12)
ax.set_ylabel('Reflectância Difusa Total ($R_d$)', fontsize=12)
ax.set_ylim(0, 1.0)
ax.grid(True, which="both", ls="--", alpha=0.4)

# ==========================================
# 5. CRIANDO OS SLIDERS
# ==========================================
ax_mel = plt.axes([0.15, 0.30, 0.70, 0.03])
ax_esp = plt.axes([0.15, 0.23, 0.70, 0.03])
ax_san = plt.axes([0.15, 0.16, 0.70, 0.03])
ax_sat = plt.axes([0.15, 0.09, 0.70, 0.03])

# O limite de "Peso Melanina" aqui escala a curva do seu artigo (1.0 = valor do artigo)
slider_mel = Slider(ax_mel, 'Peso Melanina', 0.1, 5.0, valinit=peso_mel_init, color='saddlebrown')
slider_esp = Slider(ax_esp, 'Espes. Epiderme (mm)', 0.007, 0.50, valinit=d_epi_init, color='grey')
slider_san = Slider(ax_san, 'Fraç. Sangue', 0.001, 0.15, valinit=v_sangue_init, color='red')
slider_sat = Slider(ax_sat, 'Saturação $O_2$', 0.0, 1.0, valinit=sat_o2_init, color='blue')

# ==========================================
# 6. ATUALIZAÇÃO EM TEMPO REAL
# ==========================================
def atualizar(val):
    novo_y = calcular_reflectancia_bicamada(slider_mel.val, slider_esp.val, slider_san.val, slider_sat.val)
    linha_grafico.set_ydata(novo_y)
    fig.canvas.draw_idle()

slider_mel.on_changed(atualizar)
slider_esp.on_changed(atualizar)
slider_san.on_changed(atualizar)
slider_sat.on_changed(atualizar)

plt.show()
# %%
