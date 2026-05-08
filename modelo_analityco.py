import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import dearpygui.dearpygui as dpg

# ==========================================
# 1 a 3. FÍSICA E MATEMÁTICA (MANTIDO INTACTO)
# ==========================================
Lambdas = np.linspace(380, 780, 400)
delta_lambda = Lambdas[1] - Lambdas[0]

def gauss_asym(x, amp, mu, sig_left, sig_right):
    sigma = np.where(x < mu, sig_left, sig_right)
    sigma = np.where(sigma == 0, 1e-6, sigma) 
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

mu_a_hbo2 = gauss(Lambdas, 2750.0, 415, 14) + gauss(Lambdas, 270.0, 542, 11) + gauss(Lambdas, 287.0, 577, 9) + gauss(Lambdas, 70.0,  455, 70) + gauss(Lambdas, 4.0,   800, 70)
mu_a_hb = gauss(Lambdas, 2950.0, 434, 14) + gauss(Lambdas, 255.0, 555, 30) + gauss(Lambdas, 50.0,  500, 55) + gauss(Lambdas, 10.0,  620, 100)
mu_a_base_derme = 0.008 * np.ones_like(Lambdas)

A_fit, B_fit, C_fit = 7.49, 2.18, 2.41
mu_a_epiderme_fitada = A_fit * (Lambdas/500.0)**(-B_fit) + C_fit

def calcular_espalhamento(S):
    a_cm, f_R, b_Mie = 43.6, 0.41, 0.562
    lambdas_norm = Lambdas / 500.0
    return S * a_cm * (f_R * (lambdas_norm)**(-4) + (1 - f_R) * (lambdas_norm)**(-b_Mie))

T_ilum = 5000.0
c2 = 1.4388e7 
S_ilum = (Lambdas**-5) / (np.exp(c2 / (Lambdas * T_ilum)) - 1.0)
x_bar = gauss_asym(Lambdas, 0.3483, 439.6, 17.5, 22.9) + gauss_asym(Lambdas, 1.1560, 600.0, 36.5722, 31.8083)
y_bar = gauss_asym(Lambdas, 1.0144, 553.8, 39.16, 49.63)
z_bar = gauss_asym(Lambdas, 1.9158, 442.0833, 18.8778, 28.5417)

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
    def f(t): return t**(1.0/3.0) if t > (6.0/29.0)**3 else (1.0/3.0) * ((29.0/6.0)**2) * t + 4.0/29.0
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
    return float(gamma(r)*255), float(gamma(g)*255), float(gamma(b_val)*255)

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
b_envelope = []
L_envelope = []

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
        envelope_real = ConvexHull(pontos_reais)
        
        vertices = envelope_real.vertices
        # Fechamos o polígono repetindo o primeiro vértice no final
        vertices_fechados = np.append(vertices, vertices[0])
        
        # Guardamos as listas prontas para o DearPyGui
        b_envelope = pontos_reais[vertices_fechados, 0].tolist()
        L_envelope = pontos_reais[vertices_fechados, 1].tolist()
        tem_dados_reais = True
        
except Exception as e:
    print(f"Aviso: Base de dados não encontrada ou erro na leitura ({e}). O envelope não será desenhado.")


# ==========================================
# 5. VARIÁVEIS GLOBAIS DE ESTADO DA UI
# ==========================================
historico_b = []
historico_L = []

# ==========================================
# 6. FUNÇÃO DE ATUALIZAÇÃO (CALLBACK)
# ==========================================
def atualizar_simulacao(sender, app_data):
    mel = dpg.get_value("slider_mel")
    esp = dpg.get_value("slider_esp")
    san = dpg.get_value("slider_san")
    sat = dpg.get_value("slider_sat")
    s_val = dpg.get_value("slider_S")
    
    R = calcular_reflectancia_bicamada(mel, esp, san, sat, s_val)
    X, Y, Z = calcular_cor_XYZ(R)
    L, a, b = xyz_para_lab(X, Y, Z)
    r, g, b_cor = xyz_para_rgb_monitor(X, Y, Z)
    
    dpg.set_value("linha_reflectancia", [Lambdas.tolist(), R.tolist()])
    dpg.set_value("texto_lab", f"L* = {L:.1f} | a* = {a:.1f} | b* = {b:.1f}")
    dpg.configure_item("retangulo_cor", fill=[r, g, b_cor, 255])
    
    global historico_b, historico_L
    if not historico_b or ((b - historico_b[-1])**2 + (L - historico_L[-1])**2)**0.5 > 0.1:
        historico_b.append(b)
        historico_L.append(L)
        # Ajuste o número abaixo caso queira o rastro mais longo ou mais curto
        if len(historico_b) > 15:
            historico_b.pop(0)
            historico_L.pop(0)
            
    dpg.set_value("linha_rastro", [historico_b, historico_L])
    dpg.set_value("ponto_atual", [[b], [L]])

# ==========================================
# 7. CONSTRUÇÃO DA INTERFACE COM DEARPYGUI
# ==========================================
dpg.create_context()

with dpg.window(label="Simulador Analítico de Pele", width=1200, height=800):
    
    with dpg.group(horizontal=True):
        
        # --- PAINEL ESQUERDO ---
        with dpg.child_window(width=600):
            dpg.add_text("Propriedades Ópticas", color=[255, 200, 0])
            
            dpg.add_slider_float(label="Peso Melanina", tag="slider_mel", default_value=1.0, min_value=0.1, max_value=8.0, callback=atualizar_simulacao)
            dpg.add_slider_float(label="Espes. Epiderme (mm)", tag="slider_esp", default_value=0.05, min_value=0.007, max_value=0.50, callback=atualizar_simulacao)
            dpg.add_slider_float(label="Fraç. Sangue", tag="slider_san", default_value=0.02, min_value=0.001, max_value=0.15, callback=atualizar_simulacao)
            dpg.add_slider_float(label="Saturação O2", tag="slider_sat", default_value=0.70, min_value=0.0, max_value=1.0, callback=atualizar_simulacao)
            dpg.add_slider_float(label="Espalhamento (S)", tag="slider_S", default_value=1.0, min_value=0.1, max_value=3.0, callback=atualizar_simulacao)
            
            dpg.add_spacing(count=5)
            dpg.add_separator()
            dpg.add_spacing(count=5)
            
            with dpg.group(horizontal=True):
                with dpg.drawlist(width=100, height=100):
                    dpg.draw_rectangle(pmin=[0, 0], pmax=[100, 100], color=[255, 255, 255], fill=[255, 200, 150, 255], tag="retangulo_cor")
                dpg.add_text("Valores CIELAB:", tag="texto_lab")

            dpg.add_spacing(count=10)
            
            with dpg.plot(label="Reflectância", height=350, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Comprimento de Onda (nm)")
                with dpg.plot_axis(dpg.mvYAxis, label="R"):
                    dpg.set_axis_limits(dpg.last_item(), 0, 1.0)
                    # CORRIGIDO: Removido o argumento "color="
                    dpg.add_line_series([], [], label="Espectro Simulado", tag="linha_reflectancia")

        # --- PAINEL DIREITO ---
        with dpg.child_window():
            with dpg.plot(label="Gráfico de Banana Invertido (b* vs L*)", height=-1, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="b* (Azul -> Amarelo)")
                dpg.set_axis_limits(dpg.last_item(), -5, 50)
                
                with dpg.plot_axis(dpg.mvYAxis, label="L* (Escuro -> Claro)"):
                    dpg.set_axis_limits(dpg.last_item(), 0, 100)
                    
                    if tem_dados_reais:
                        # CORRIGIDO: Removido o argumento "color="
                        dpg.add_line_series(b_envelope, L_envelope, label="Envelope Real (Dados)", tag="linha_envelope")
                    
                    # CORRIGIDO: Removido o argumento "color="
                    dpg.add_line_series([], [], label="Rastro", tag="linha_rastro")
                    dpg.add_scatter_series([], [], label="Paciente Atual", tag="ponto_atual")

# Chamada inicial
atualizar_simulacao(None, None)

dpg.create_viewport(title='Simulador Analítico (DearPyGui)', width=1220, height=820)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()