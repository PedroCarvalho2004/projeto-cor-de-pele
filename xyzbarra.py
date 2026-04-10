import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# 1. CONFIGURAÇÕES E FUNÇÕES
# ==========================================
nome_do_arquivo = 'CMF.csv' # Garanta que o arquivo está na mesma pasta

def gauss_asym(x, amp, mu, sig_left, sig_right):
    """Gaussiana assimétrica para o ajuste fino."""
    sigma = np.where(x < mu, sig_left, sig_right)
    # Evita divisão por zero
    sigma = np.where(sigma == 0, 1e-6, sigma) 
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

def calcular_r2(y_real, y_pred):
    """Calcula a nota do ajuste (0 a 1)."""
    residuos = y_real - y_pred
    soma_quadrados_residuos = np.sum(residuos**2)
    soma_quadrados_total = np.sum((y_real - np.mean(y_real))**2)
    return 1 - (soma_quadrados_residuos / soma_quadrados_total)

# ==========================================
# 2. CARREGANDO DADOS REAIS
# ==========================================
try:
    df = pd.read_csv(nome_do_arquivo)
    x_ref = df['Wavelength'].values
    y_real_x = df['x_bar'].values
    y_real_y = df['y_bar'].values
    y_real_z = df['z_bar'].values
    print("Dados de referência carregados!")
except Exception as e:
    print(f"Erro ao carregar o CSV: {e}")
    exit()

# Eixo para o gráfico ficar lisinho
Lambdas = np.linspace(380, 780, 500)

# ==========================================
# 3. O SEU MODELO (COLOQUE SEUS VALORES AQUI)
# ==========================================

# --- AJUSTE PARA X_BARRA ---
# Se usar mais de um lobo, basta somar: gauss_asym(...) + gauss_asym(...)
x_barra = (gauss_asym(Lambdas, 0.3483, 439.6, 17.5, 22.9) + 
           gauss_asym(Lambdas, 1.1560, 600.0, 36.5722, 31.8083))

# --- AJUSTE PARA Y_BARRA ---
y_barra = (gauss_asym(Lambdas, 1.0144, 553.8, 39.16, 49.63) )

# --- AJUSTE PARA Z_BARRA ---
z_barra = (gauss_asym(Lambdas, 1.9158, 442.0833, 18.8778, 28.5417))

# ==========================================
# 4. CÁLCULO DA ACURÁCIA (R²)
# ==========================================
# Recalculando o modelo exatamente nos pontos do CSV para comparar
x_pred = (gauss_asym(x_ref, 0.3483, 439.6, 17.5, 22.9) + 
           gauss_asym(x_ref, 1.1560, 553.8, 39.15, 49.64))

y_pred = (gauss_asym(x_ref, 1.0144, 553.8, 39.16, 49.63) )

z_pred = (gauss_asym(x_ref, 1.9158, 553.3333, 38.3417, 49.3667))

print("\n--- RESULTADOS DO SEU FIT ---")
print(f"R² X_barra: {calcular_r2(y_real_x, x_pred):.4f}")
print(f"R² Y_barra: {calcular_r2(y_real_y, y_pred):.4f}")
print(f"R² Z_barra: {calcular_r2(y_real_z, z_pred):.4f}")

# ==========================================
# 5. PLOTAGEM FINAL
# ==========================================
plt.figure(figsize=(12, 7))

# Plot dos Dados Reais (Tracejado/Transparente)
plt.plot(x_ref, y_real_x, 'r--', alpha=0.3, label='X Real')
plt.plot(x_ref, y_real_y, 'g--', alpha=0.3, label='Y Real')
plt.plot(x_ref, y_real_z, 'b--', alpha=0.3, label='Z Real')

# Plot do Seu Modelo (Sólido)
plt.plot(Lambdas, x_barra, 'r', linewidth=2, label='Seu Modelo X')
plt.plot(Lambdas, y_barra, 'g', linewidth=2, label='Seu Modelo Y')
plt.plot(Lambdas, z_barra, 'b', linewidth=2, label='Seu Modelo Z')

plt.title('Comparação: Dados CVRL vs. Seu Modelo Ajustado', fontsize=14)
plt.xlabel('Comprimento de Onda (nm)')
plt.ylabel('Sensibilidade')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()