# %%
import numpy as np 
import matplotlib.pyplot as plt




# %%
# paramentros livres:
Lambdas = np.linspace(380,780,400) #comprimento de onda de 380 a 780, step 1

#parametros da melanina: mu_a = c1*(lambda)^-c2
eumel_c1= 1.7e11
eumel_c2= 3.48

# Parâmetros da Linha de Base (Pele sem sangue e sem melanina)
# Fórmula de Jacques: mu_a = c3 + c4 * exp(-(lambda - c5) / c6)
base_c3 = 0.00244
base_c4 = 0.853
base_c5 = 154.0
base_c6 = 66.2

#funções:

#mu_a da eumelanina 
mu_a_eumel= eumel_c1*(Lambdas**-eumel_c2)

#mua_a de base
mu_a_base = base_c3 + base_c4 * np.exp(-(Lambdas - base_c5) / base_c6)

# 2.3 Sangue (Aproximado por Gaussianas para modelar os picos)
def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

# O Sangue Oxigenado (HbO2) tem um pico gigante em 415nm e dois menores em 542 e 577nm
mu_a_hbo2 = gauss(Lambdas, 25.0, 415, 18) + gauss(Lambdas, 1.3, 542, 10) + gauss(Lambdas, 1.25, 577, 10)

# O Sangue Desoxigenado (Hb) tem pico em 430nm e um pico mais largo em 555nm
mu_a_hb = gauss(Lambdas, 18.0, 430, 18) + gauss(Lambdas, 1.1, 555, 14)

# ==========================================
# 3. PLOTANDO OS ESPECTROS
# ==========================================
plt.figure(figsize=(10, 6))

# Desenhando cada curva
plt.plot(Lambdas, mu_a_eumel, label='Melanina', color='saddlebrown', linewidth=2.5)
plt.plot(Lambdas, mu_a_hbo2, label='Sangue Oxigenado (HbO2)', color='red', linewidth=2)
plt.plot(Lambdas, mu_a_hb, label='Sangue Desoxigenado (Hb)', color='blue', linestyle='--', linewidth=2)
plt.plot(Lambdas, mu_a_base, label='Linha de Base', color='gray', linewidth=2)

# Estética do Gráfico
plt.title('Espectros de Absorção dos Constituintes da Pele', fontsize=14)
plt.xlabel('Comprimento de Onda (nm)', fontsize=12)
plt.ylabel('Coeficiente de Absorção $\mu_a$ ($mm^{-1}$)', fontsize=12)

# DICA DE OURO: Usamos escala logarítmica no eixo Y porque o sangue absorve 
# MUITO mais luz em 415nm do que no resto do espectro. Se não fosse log, 
# não daria para ver a melanina direita!
plt.yscale('log') 

plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()
plt.show()









# %% plot somente da melanina com a absorção de base
plt.plot(Lambdas, mu_a_base, label='Linha de Base', color='gray', linewidth=2)
plt.plot(Lambdas, mu_a_eumel, label='Melanina', color='saddlebrown', linewidth=2.5)
plt.yscale('log') 

plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()
plt.show()






# %% plot somente da hemoglobina
plt.plot(Lambdas, mu_a_hbo2, label='Sangue Oxigenado (HbO2)', color='red', linewidth=2)
plt.plot(Lambdas, mu_a_hb, label='Sangue Desoxigenado (Hb)', color='blue', linestyle='--', linewidth=2)
plt.yscale('log') 

plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()
plt.show()
# %%
# %%
import urllib.request
import re
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. BUSCANDO OS DADOS DIRETO DO OMLC
# ==========================================
url = "https://omlc.org/spectra/hemoglobin/summary.html"
print("Baixando dados reais do sangue do site OMLC...")

# Acessa a página (fingindo ser um navegador para não ser bloqueado)
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
response = urllib.request.urlopen(req)
html_content = response.read().decode('utf-8')

# ==========================================
# 2. EXTRAINDO A TABELA COM REGEX
# ==========================================
# O script "caça" linhas que tenham 3 números separados por espaços (lambda, HbO2, Hb)
padrao = r'^\s*(\d{3,4})\s+([\d\.]+)\s+([\d\.]+)'
linhas_tabela = re.findall(padrao, html_content, re.MULTILINE)

# Converte o texto que puxamos da internet para uma matriz matemática do Numpy
matriz = np.array(linhas_tabela, dtype=float)

# Separa as 3 colunas (Dados Brutos do Site)
lambdas_brutos = matriz[:, 0]       # nm
eps_hbo2_bruto = matriz[:, 1]       # cm^-1/M
eps_hb_bruto = matriz[:, 2]         # cm^-1/M

# ==========================================
# 3. A CONVERSÃO FÍSICA PARA O NOSSO MODELO
# ==========================================
# O "Fator Mágico" (2.303 * 0.00233 / 10) que deduzimos para Sangue Puro em mm^-1
fator_mm = 0.000536

mu_a_hbo2_puro = eps_hbo2_bruto * fator_mm
mu_a_hb_puro = eps_hb_bruto * fator_mm

# ==========================================
# 4. PLOTANDO O GRÁFICO (IGUAL AO DO SITE)
# ==========================================
plt.figure(figsize=(10, 6))

# Plotando as duas curvas (agora na nossa unidade oficial de mm^-1)
plt.plot(lambdas_brutos, mu_a_hbo2_puro, color='blue', linewidth=2.5, label='Sangue Oxigenado (HbO2)')
plt.plot(lambdas_brutos, mu_a_hb_puro, color='red', linestyle='--', linewidth=2.5, label='Sangue Desoxigenado (Hb)')

# Estética: Escala logarítmica e zoom igual ao do OMLC (250 a 1000 nm)
plt.yscale('log')
plt.xlim(250, 1000)

# Títulos
plt.title('Coeficiente de Absorção do Sangue Puro ($100\%$ de volume)', fontsize=14)
plt.xlabel('Comprimento de Onda (nm)', fontsize=12)
plt.ylabel('Coeficiente de Absorção $\mu_a$ ($mm^{-1}$)', fontsize=12)

# Grades e Legendas
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(shadow=True, fontsize=11)

print("Gráfico gerado com sucesso!")
plt.show()

# %%
import urllib.request
import re
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PUXANDO DADOS REAIS DO OMLC (Apenas para servir de guia visual)
# ==========================================
url = "https://omlc.org/spectra/hemoglobin/summary.html"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
html_content = urllib.request.urlopen(req).read().decode('utf-8')

linhas_tabela = re.findall(r'^\s*(\d{3,4})\s+([\d\.]+)\s+([\d\.]+)', html_content, re.MULTILINE)
matriz = np.array(linhas_tabela, dtype=float)

mask = (matriz[:, 0] >= 400) & (matriz[:, 0] <= 650)
lambdas = matriz[mask, 0]
mu_a_hbo2_real = matriz[mask, 1] * 0.000536  # Convertendo para mm^-1
mu_a_hb_real = matriz[mask, 2] * 0.000536    # Convertendo para mm^-1

# ==========================================
# 2. OS SEUS PARÂMETROS MANUAIS (Altere os valores aqui!)
# ==========================================

# SANGUE DESOXIGENADO (Hb) - 2 Gaussianas
# [Amplitude, Centro (nm), Largura]
hb_amp1, hb_centro1, hb_larg1 = 57.0, 430.0, 18.0  # Pico principal (Banda de Soret)
hb_amp2, hb_centro2, hb_larg2 = 6.0,  555.0, 20.0  # Pico secundário

# SANGUE OXIGENADO (HbO2) - 3 Gaussianas
# [Amplitude, Centro (nm), Largura]
hbo2_amp1, hbo2_centro1, hbo2_larg1 = 68.0, 414.0, 16.0  # Pico principal
hbo2_amp2, hbo2_centro2, hbo2_larg2 = 7.5,  542.0, 12.0  # Primeiro morrinho (Banda Q)
hbo2_amp3, hbo2_centro3, hbo2_larg3 = 7.8,  576.0, 12.0  # Segundo morrinho (Banda Q)

# ==========================================
# 3. CONSTRUINDO AS SUAS CURVAS MATEMÁTICAS
# ==========================================
def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

# Somando as gaussianas manuais
minha_curva_hb = gauss(lambdas, hb_amp1, hb_centro1, hb_larg1) + \
                 gauss(lambdas, hb_amp2, hb_centro2, hb_larg2)

minha_curva_hbo2 = gauss(lambdas, hbo2_amp1, hbo2_centro1, hbo2_larg1) + \
                   gauss(lambdas, hbo2_amp2, hbo2_centro2, hbo2_larg2) + \
                   gauss(lambdas, hbo2_amp3, hbo2_centro3, hbo2_larg3)

# ==========================================
# 4. PLOTANDO O GRÁFICO
# ==========================================
plt.figure(figsize=(10, 6))

# A "Sombra" dos dados reais (Bolinhas transparentes)
plt.scatter(lambdas, mu_a_hbo2_real, color='lightcoral', s=15, label='Referência OMLC (HbO2)', alpha=0.3)
plt.scatter(lambdas, mu_a_hb_real, color='lightblue', s=15, label='Referência OMLC (Hb)', alpha=0.3)

# As SUAS curvas construídas à mão (Linhas fortes)
plt.plot(lambdas, minha_curva_hbo2, color='red', linewidth=2.5, label='Minha Curva HbO2')
plt.plot(lambdas, minha_curva_hb, color='blue', linestyle='--', linewidth=2.5, label='Minha Curva Hb')

plt.yscale('log')
plt.title('Modelo Manual de Absorção do Sangue', fontsize=14)
plt.xlabel('Comprimento de Onda (nm)', fontsize=12)
plt.ylabel('Absorção $\mu_a$ ($mm^{-1}$)', fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()

plt.show()
# %%
# %%
import urllib.request
import re
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PUXANDO DADOS REAIS DO OMLC (Como sombra)
# ==========================================
url = "https://omlc.org/spectra/hemoglobin/summary.html"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
html_content = urllib.request.urlopen(req).read().decode('utf-8')

linhas_tabela = re.findall(r'^\s*(\d{3,4})\s+([\d\.]+)\s+([\d\.]+)', html_content, re.MULTILINE)
matriz = np.array(linhas_tabela, dtype=float)

# Pegando a faixa inteira do site (250 a 1000 nm) para o visual ficar igual
lambdas = matriz[:, 0]
mu_a_hbo2_real = matriz[:, 1] * 0.000536  
mu_a_hb_real = matriz[:, 2] * 0.000536    

# ==========================================
# 2. OS SEUS PARÂMETROS MANUAIS
# ==========================================
# SANGUE DESOXIGENADO (Hb)
hb_amp1, hb_centro1, hb_larg1 = 290.0, 430.0, 13.0
hb_amp2, hb_centro2, hb_larg2 = 27.0,  555.0, 35.0
base_hb = 0.15  # <--- O "chão" para salvar o logaritmo

# SANGUE OXIGENADO (HbO2)
hbo2_amp1, hbo2_centro1, hbo2_larg1 = 290.0, 414.0, 13.0
hbo2_amp2, hbo2_centro2, hbo2_larg2 = 50,  560.0,30.0
hbo2_amp3, hbo2_centro3, hbo2_larg3 = 27,  560.0, 12.0
base_hbo2 = 0.15 # <--- O "chão" para salvar o logaritmo
# LINHA DE BASE

linha_base=10
#================================
#equação de reta para subtrair
#================================
coef_angular_a=0.00
coef_linear_b=0
minha_reta = (coef_angular_a * lambdas) + coef_linear_b

# ==========================================
# 3. CONSTRUINDO AS SUAS CURVAS MATEMÁTICAS
# ==========================================
def gauss(x, amp, mu, sigma):
    return (amp * np.exp(-0.5 * ((x - mu) / sigma)**2))

# Adicionamos a 'base' para a curva não cair para o infinito no gráfico log
minha_curva_hb = base_hb + \
                 gauss(lambdas, hb_amp1, hb_centro1, hb_larg1) + \
                 gauss(lambdas, hb_amp2, hb_centro2, hb_larg2) - \
                 minha_reta +0.5

minha_curva_hbo2 = base_hbo2 + \
                   gauss(lambdas, hbo2_amp1, hbo2_centro1, hbo2_larg1) + \
                   gauss(lambdas, hbo2_amp2, hbo2_centro2, hbo2_larg2) - \
                   gauss(lambdas, hbo2_amp3, hbo2_centro3, hbo2_larg3) - \
                    +0.05

# ==========================================
# 4. PLOTANDO O GRÁFICO (Estilo OMLC)
# ==========================================
plt.figure(figsize=(10, 6))

# Dados reais como linhas grossas transparentes no fundo
plt.plot(lambdas, mu_a_hbo2_real, color='lightcoral', linewidth=5, label='Real OMLC (HbO2)', alpha=0.4)
plt.plot(lambdas, mu_a_hb_real, color='lightblue', linewidth=5, label='Real OMLC (Hb)', alpha=0.4)

# Suas gaussianas por cima
plt.plot(lambdas, minha_curva_hbo2, color='red', linewidth=2, label='Minha Curva HbO2')
plt.plot(lambdas, minha_curva_hb, color='blue', linestyle='--', linewidth=2, label='Minha Curva Hb')

plt.yscale('log')
plt.xlim(250, 1000)   # Limites do site
plt.ylim(0.01, 500)   # Ajuste do eixo Y para mostrar bem o fundo e o topo

plt.title('Modelo Manual de Absorção do Sangue (Visão Completa)', fontsize=14)
plt.xlabel('Comprimento de Onda (nm)', fontsize=12)
plt.ylabel('Absorção $\mu_a$ ($mm^{-1}$)', fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()

plt.show()

# %%
