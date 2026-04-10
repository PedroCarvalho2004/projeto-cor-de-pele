
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. EIXO DE COMPRIMENTOS DE ONDA
# ==========================================
Lambdas = np.linspace(380, 780, 400)

# ==========================================
# 2. CARREGANDO E CONVERTENDO OS DADOS REAIS (TXT)
# ==========================================
try:
    dados_sangue = np.loadtxt('sangue.txt', skiprows=2)
    lambdas_txt = dados_sangue[:, 0]
    eps_hbo2_txt = dados_sangue[:, 1]
    eps_hb_txt = dados_sangue[:, 2]
except Exception as e:
    print("Erro ao carregar o arquivo:", e)
    lambdas_txt = Lambdas
    eps_hbo2_txt = np.zeros_like(Lambdas)
    eps_hb_txt = np.zeros_like(Lambdas)

# Interpolando para o nosso eixo de 400 pontos
eps_hbo2_interp = np.interp(Lambdas, lambdas_txt, eps_hbo2_txt)
eps_hb_interp = np.interp(Lambdas, lambdas_txt, eps_hb_txt)

# Conversão físico-química para mu_a (cm^-1) assumindo sangue 100% puro (150 g/L)
conc_molar = 150.0 / 64500.0
fator_conversao = np.log(10) * conc_molar

mu_a_hbo2_real = eps_hbo2_interp * fator_conversao
mu_a_hb_real = eps_hb_interp * fator_conversao

# ==========================================
# 3. O SEU MODELO ANALÍTICO (PARA VOCÊ MODELAR)
# ==========================================
def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

# ---> ÁREA DE MODELAGEM <---
# Ajuste os valores de (amp, mu, sigma) aqui para encaixar com a curva real!

mu_a_hbo2_analitico = (
    gauss(Lambdas, 2750.0, 415, 14) +   
    gauss(Lambdas, 270.0, 542, 11) +    
    gauss(Lambdas, 287.0, 577, 9) +     
    gauss(Lambdas, 70.0,  455, 70) +
    gauss(Lambdas, 4.0,   800, 70)     
)   

mu_a_hb_analitico = (
    gauss(Lambdas, 2950.0, 434, 14) +   
    gauss(Lambdas, 255.0, 555, 30) +    
    gauss(Lambdas, 50.0,  500, 55) +
    gauss(Lambdas, 10.0,  620, 100)     
)

# ==========================================
# 4. PLOTANDO PARA COMPARAÇÃO VISUAL
# ==========================================
# Criando dois gráficos lado a lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Gráfico 1: Sangue Oxigenado (HbO2) ---
ax1.plot(Lambdas, mu_a_hbo2_real, color='black', linestyle='--', linewidth=3, label='Dados Reais (TXT)', alpha=0.6)
ax1.plot(Lambdas, mu_a_hbo2_analitico, color='red', linestyle='-', linewidth=2, label='Seu Modelo Analítico')

ax1.set_title('Sangue Oxigenado ($HbO_2$)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Comprimento de Onda (nm)', fontsize=12)
ax1.set_ylabel(r'Coeficiente de Absorção $\mu_a$ ($cm^{-1}$)', fontsize=12)
ax1.legend()
ax1.grid(True, which="both", ls="--", alpha=0.4)
ax1.set_yscale('log') # Usando escala log para ver os picos pequenos e grandes juntos

# --- Gráfico 2: Sangue Desoxigenado (Hb) ---
ax2.plot(Lambdas, mu_a_hb_real, color='black', linestyle='--', linewidth=3, label='Dados Reais (TXT)', alpha=0.6)
ax2.plot(Lambdas, mu_a_hb_analitico, color='blue', linestyle='-', linewidth=2, label='Seu Modelo Analítico')

ax2.set_title('Sangue Desoxigenado (Hb)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Comprimento de Onda (nm)', fontsize=12)
ax2.legend()
ax2.grid(True, which="both", ls="--", alpha=0.4)
ax2.set_yscale('log')

plt.tight_layout()
plt.show()