
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================
# 1. OS SEUS DADOS EXPERIMENTAIS DO ARTIGO
# ==========================================
# Eixo X: Comprimento de onda dos Lasers (nm)
x_dados = np.array([476, 514, 532, 633])

# Eixo Y: Coeficiente de Absorção real (cm^-1)
y_dados = np.array([11.70, 8.58, 8.65, 5.16])

# ==========================================
# 2. DEFININDO O MODELO FÍSICO
# ==========================================
def modelo_absorcao_tecido(lambdas, a, b, c):
    """
    a: Amplitude da absorção em 500 nm
    b: Fator de decaimento (potência)
    c: Absorção basal residual (linha de base)
    """
    return a * (lambdas / 500.0)**(-b) + c

# ==========================================
# 3. EXECUTANDO O FITTING
# ==========================================
# p0 é o "chute inicial" (Amplitude=10, Potência=3, Base=1)
parametros_otimizados, matriz_covariancia = curve_fit(
    modelo_absorcao_tecido, 
    x_dados, 
    y_dados, 
    p0=[10.0, 3.0, 1.0],
    maxfev=10000 # Aumentando as tentativas para não dar erro
)

A_fit = parametros_otimizados[0]
B_fit = parametros_otimizados[1]
C_fit = parametros_otimizados[2]

print("=== SUA NOVA EQUAÇÃO DO ARTIGO ===")
print(f"mu_a(lambda) = {A_fit:.3f} * (lambda/500)^(-{B_fit:.3f}) + {C_fit:.3f} cm^-1")

# ==========================================
# 4. PLOTANDO O RESULTADO
# ==========================================
# Eixo X contínuo para desenhar a linha do modelo
Lambdas_continuo = np.linspace(400, 700, 200)
y_fit_continuo = modelo_absorcao_tecido(Lambdas_continuo, A_fit, B_fit, C_fit)

plt.figure(figsize=(10, 6))

# Plotando os dados reais (bolinhas pretas)
plt.scatter(x_dados, y_dados, color='black', s=80, label='Dados Ex-Vivo do Artigo', zorder=5)

# Plotando a curva matemática descoberta (linha vermelha)
plt.plot(Lambdas_continuo, y_fit_continuo, color='red', linewidth=3, label='Fitting Analítico (Lei de Potência)')

plt.title('Ajuste de Curva: Dados Experimentais da Literatura', fontsize=14)
plt.xlabel('Comprimento de Onda (nm)', fontsize=12)
plt.ylabel(r'Coeficiente de Absorção Total $\mu_a$ ($cm^{-1}$)', fontsize=12)

# Adicionando um texto no gráfico com a sua equação
texto_equacao = f"$\\mu_a = {A_fit:.2f} \\cdot (\\lambda/500)^{{-{B_fit:.2f}}} + {C_fit:.2f}$"
plt.text(550, 10, texto_equacao, fontsize=12, color='darkred', bbox=dict(facecolor='white', alpha=0.8))

plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(fontsize=12)
plt.show()  