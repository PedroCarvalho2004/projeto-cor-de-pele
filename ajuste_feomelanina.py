import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================
# 1. DADOS DA LITERATURA PARA FEOMELANINA
# ==========================================
# Pontos representativos do decaimento abrupto da Feomelanina
# Substitua pelos seus dados ex-vivo caso possua medidas específicas
x_dados = np.array([400, 450, 500, 550, 600, 650, 700])
y_dados = np.array([100.0, 45.0, 20.0, 9.0, 4.0, 1.8, 0.8])

# ==========================================
# 2. MODELO FÍSICO (Lei de Potência)
# ==========================================
def modelo_absorcao_feomelanina(lambdas, a, b, c):
    """
    a: Amplitude da absorção em 500 nm
    b: Fator de decaimento (potência) - esperado ser maior que o da Eumelanina
    c: Absorção basal residual
    """
    return a * (lambdas / 500.0)**(-b) + c

# ==========================================
# 3. EXECUTANDO O FITTING
# ==========================================
parametros_otimizados, matriz_covariancia = curve_fit(
    modelo_absorcao_feomelanina, 
    x_dados, 
    y_dados, 
    p0=[20.0, 5.0, 0.0], # Chute inicial adaptado para queda brusca
    maxfev=20000
)

A_fit = parametros_otimizados[0]
B_fit = parametros_otimizados[1]
C_fit = parametros_otimizados[2]

print("=== EQUAÇÃO DA FEOMELANINA ===")
print(f"mu_a_feo(lambda) = {A_fit:.3f} * (lambda/500)^(-{B_fit:.3f}) + {C_fit:.3f} cm^-1")

# ==========================================
# 4. PLOTAGEM DO RESULTADO
# ==========================================
Lambdas_continuo = np.linspace(380, 780, 400)
y_fit_continuo = modelo_absorcao_feomelanina(Lambdas_continuo, A_fit, B_fit, C_fit)

plt.figure(figsize=(10, 6))

plt.scatter(x_dados, y_dados, color='orange', s=80, label='Dados de Referência (Feomelanina)', zorder=5)
plt.plot(Lambdas_continuo, y_fit_continuo, color='red', linewidth=3, label='Fitting Analítico')

plt.title('Ajuste de Curva: Feomelanina', fontsize=14, fontweight='bold')
plt.xlabel('Comprimento de Onda (nm)', fontsize=12)
plt.ylabel(r'Coeficiente de Absorção $\mu_a$ relativa ($cm^{-1}$)', fontsize=12)

texto_equacao = f"$\\mu_a = {A_fit:.2f} \\cdot (\\lambda/500)^{{-{B_fit:.2f}}} + {C_fit:.2f}$"
plt.text(500, 60, texto_equacao, fontsize=12, color='darkred', bbox=dict(facecolor='white', alpha=0.8))

plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(fontsize=12)
plt.show()