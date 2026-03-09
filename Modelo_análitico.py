# %%
%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DEFINIÇÃO DOS CROMÓFOROS E ABSORÇÃO
# ==========================================
Lambdas = np.linspace(380, 780, 400)

def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)
# Absorção do Sangue (Seus modelos ajustados)

mu_a_hbo2 = (

    gauss(Lambdas, 102.0, 415, 14) +   

    gauss(Lambdas, 9, 542, 11) +    

    gauss(Lambdas, 10.0, 577, 9) +    

    gauss(Lambdas, 2.5,  455, 85)     

)

mu_a_hb = (

    gauss(Lambdas, 111.0, 434, 14) +  

    gauss(Lambdas, 10, 555, 30) +   

    gauss(Lambdas, 1.8, 500, 55)      

)


# Absorção da Melanina (Modelo analítico padrão)
mu_a_melanina = (Lambdas / 500.0)**(-3.33)

# Absorção da linha de base da pele
mu_a_base = 0.000025 * np.ones_like(Lambdas)

# ==========================================
# 2. EQUAÇÃO DE ESPALHAMENTO (EQUAÇÃO 12 DA LITERATURA)
# ==========================================
def calcular_espalhamento(S):
    # Parâmetros da Derme (referência do projeto)
    a_cm = 43.6       # Amplitude base em 500 nm (cm^-1)
    f_R = 0.41        # Fração de Rayleigh
    b_Mie = 0.562     # Potência de Mie
    
    # Normalização do comprimento de onda
    lambdas_norm = Lambdas / 500.0
    
    # Componentes separados para visualização
    mu_s_rayleigh = S * a_cm * f_R * (lambdas_norm)**(-4)
    mu_s_mie = S * a_cm * (1 - f_R) * (lambdas_norm)**(-b_Mie)
    
    # Equação 12 completa
    mu_s_total = mu_s_rayleigh + mu_s_mie
    return mu_s_total, mu_s_rayleigh, mu_s_mie

# ==========================================
# 3. EQUAÇÃO DE FARRELL (REFLECTÂNCIA)
# ==========================================
def calcular_reflectancia_farrell(c_melanina, fracao_sangue, saturacao_o2, S):
    
    # 1. Absorção Total (mu_a) em cm^-1
    c_hbo2 = fracao_sangue * saturacao_o2
    c_hb = fracao_sangue * (1 - saturacao_o2)
    mu_a_total = (c_melanina * mu_a_melanina) + (c_hbo2 * mu_a_hbo2) + (c_hb * mu_a_hb) + mu_a_base
    
    # 2. Espalhamento Reduzido Total (mu_s') em cm^-1
    mu_s_total, _, _ = calcular_espalhamento(S)
    
    # 3. Parâmetros da Teoria de Difusão
    mu_t_prime = mu_a_total + mu_s_total
    a_prime = mu_s_total / mu_t_prime
    mu_eff = np.sqrt(3.0 * mu_a_total * mu_t_prime)
    
    z0 = 1.0 / mu_t_prime
    D = 1.0 / (3.0 * mu_t_prime)
    
    # Fator de reflexão interna para tecido biológico (n ~= 1.4)
    A = 3.2
    zb = 2.0 * A * D
    
    # 4. Solução de Farrell
    R_d = (a_prime / 2.0) * (np.exp(-mu_eff * z0) + np.exp(-mu_eff * (z0 + 2.0 * zb)))
    
    return R_d

# ==========================================
# 4. PLOTAGEM DOS RESULTADOS
# ==========================================
plt.figure(figsize=(12, 8))

# --- Subplot 1: Validação do Espalhamento (Equação 12) ---

S_base = 1.0
mu_s_tot, mu_s_ray, mu_s_mie = calcular_espalhamento(S_base)



# --- Subplot 2: Reflectância de Farrell ---

v_sangue = 0.02
sat_o2 = 0.70

Rd_clara = calcular_reflectancia_farrell(0.5, v_sangue, sat_o2, S_base)
Rd_morena = calcular_reflectancia_farrell(2.5, v_sangue, sat_o2, S_base)
Rd_negra = calcular_reflectancia_farrell(8.0, v_sangue, sat_o2, S_base)

plt.plot(Lambdas, Rd_clara, label='Pele Clara', color='lightcoral', linewidth=2)
plt.plot(Lambdas, Rd_morena, label='Pele Morena', color='saddlebrown', linewidth=2)
plt.plot(Lambdas, Rd_negra, label='Pele Negra', color='maroon', linewidth=2)

plt.title('Reflectância Difusa (Farrell)', fontsize=12)
plt.xlabel('Comprimento de Onda (nm)')
plt.ylabel('Reflectância ($R_d$)')
plt.ylim(0, 0.8)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()

plt.tight_layout()
plt.show()
# %%
