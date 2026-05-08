import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import dearpygui.dearpygui as dpg

# ==========================================
# 1. CONSTANTES E FÍSICA DOS CROMÓFOROS
# ==========================================
Lambdas = np.linspace(380, 780, 400)
delta_lambda = Lambdas[1] - Lambdas[0]

def gauss_asym(x, amp, mu, sig_left, sig_right):
    sigma = np.where(x < mu, sig_left, sig_right)
    sigma = np.where(sigma == 0, 1e-6, sigma) 
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

# Absorção base: Hemoglobina, Derme Base e Melanina
mu_a_hbo2 = gauss(Lambdas, 2750.0, 415, 14) + gauss(Lambdas, 270.0, 542, 11) + gauss(Lambdas, 287.0, 577, 9) + gauss(Lambdas, 70.0,  455, 70) + gauss(Lambdas, 4.0,   800, 70)
mu_a_hb = gauss(Lambdas, 2950.0, 434, 14) + gauss(Lambdas, 255.0, 555, 30) + gauss(Lambdas, 50.0,  500, 55) + gauss(Lambdas, 10.0,  620, 100)
mu_a_base_derme = 0.008 * np.ones_like(Lambdas)
mu_a_melanina_fit = 7.49 * (Lambdas/500.0)**(-2.18) + 2.41

# Novos Cromóforos com Proteção Física (mu_a nunca negativo)
curva_caro_raw = gauss(Lambdas, 130000, 449, 40) + gauss(Lambdas, -20000, 520, 30)
mu_a_caroteno_uM = np.maximum(0, curva_caro_raw) * (2.303 * 1e-6)
mu_a_bilirrubina_base = gauss_asym(Lambdas, 1.85, 456, 43, 20)

def calcular_espalhamento_derme(S):
    # Parâmetros clássicos de Jacques para a derme: Rayleigh + Mie
    a_cm, f_R, b_Mie = 43.6, 0.41, 0.562
    lambdas_norm = Lambdas / 500.0
    return S * a_cm * (f_R * (lambdas_norm)**(-4) + (1 - f_R) * (lambdas_norm)**(-b_Mie))

# ==========================================
# 2. COLORIMETRIA CIE 1931
# ==========================================
T_ilum = 5000.0
c2 = 1.4388e7 
S_ilum = (Lambdas**-5) / (np.exp(c2 / (Lambdas * T_ilum)) - 1.0)
x_bar = gauss_asym(Lambdas, 0.3483, 439.6, 17.5, 22.9) + gauss_asym(Lambdas, 1.1560, 600.0, 36.5722, 31.8083)
y_bar = gauss_asym(Lambdas, 1.0144, 553.8, 39.16, 49.63)
z_bar = gauss_asym(Lambdas, 1.9158, 442.0833, 18.8778, 28.5417)

k_norm = 100.0 / np.sum(S_ilum * y_bar * delta_lambda)
Xn = k_norm * np.sum(S_ilum * x_bar * delta_lambda)
Yn = 100.0
Zn = k_norm * np.sum(S_ilum * z_bar * delta_lambda)

def xyz_para_lab(X, Y, Z):
    def f(t): return t**(1.0/3.0) if t > 0.008856 else 7.787*t + 16.0/116.0
    L = 116.0 * f(Y/Yn) - 16.0
    a = 500.0 * (f(X/Xn) - f(Y/Yn))
    b = 200.0 * (f(Y/Yn) - f(Z/Zn))
    return float(L), float(a), float(b)

def xyz_para_rgb(X, Y, Z):
    x, y, z = X/100.0, Y/100.0, Z/100.0
    r =  3.2406*x - 1.5372*y - 0.4986*z
    g = -0.9689*x + 1.8758*y + 0.0415*z
    b =  0.0557*x - 0.2040*y + 1.0570*z
    rgb = np.clip([r, g, b], 0, 1)
    rgb = np.where(rgb <= 0.0031308, 12.92*rgb, 1.055*(rgb**(1.0/2.4)) - 0.055)
    return [int(c*255) for c in rgb]

# ==========================================
# 3. MODELO DE REFLECTÂNCIA DE BICAMADA
# ==========================================
def calcular_reflectancia(peso_mel, d_mm, v_sangue, sat_o2, S_derme, caro_uM, bili):
    d_cm = d_mm / 10.0
    
    # 1. Interface Física (Reflexão de Fresnel e Saunderson)
    n_pele = 1.4
    R_sp = ((n_pele - 1)**2) / ((n_pele + 1)**2) # Aprox 0.0277
    R_int = 0.55
    
    # 2. Epiderme (Apenas Filtro Transmissivo)
    mu_a_epi = (peso_mel * mu_a_melanina_fit) + (caro_uM * mu_a_caroteno_uM)
    T_epi_ida_volta = np.exp(-mu_a_epi * d_cm * 2.0)
    
    # 3. Derme (Difusão e Espalhamento)
    c_hbo2 = v_sangue * sat_o2
    c_hb = v_sangue * (1.0 - sat_o2)
    mu_a_d = (c_hbo2 * mu_a_hbo2) + (c_hb * mu_a_hb) + (bili * mu_a_bilirrubina_base) + mu_a_base_derme
    mu_s_d = calcular_espalhamento_derme(S_derme)
    
    mu_tr = mu_a_d + mu_s_d
    a_p = mu_s_d / mu_tr
    mu_eff = np.sqrt(3.0 * mu_a_d * mu_tr)
    D = 1.0 / (3.0 * mu_tr)
    z0 = 1.0 / mu_tr
    zb = 2.0 * 3.2 * D
    
    R_derme = (a_p / 2.0) * (np.exp(-mu_eff * z0) + np.exp(-mu_eff * (z0 + 2.0 * zb)))
    
    # 4. Combinação e Correção Final
    R_difusa = T_epi_ida_volta * R_derme
    R_total = R_sp + ((1.0 - R_sp) * (1.0 - R_int) * R_difusa) / (1.0 - R_int * R_difusa)
    return R_total

# ==========================================
# 4. CARREGAMENTO DE DADOS REAIS
# ==========================================
tem_dados = False
b_hull, L_hull = [], []

try:
    nome_arquivo = 'Skin Database 1-92.xlsx'
    # Tenta ler a aba 'Database' primeiro, se falhar, lê a padrão
    try:
        df = pd.read_excel(nome_arquivo, sheet_name='Database', header=1)
    except:
        df = pd.read_excel(nome_arquivo, header=1)
        
    df.columns = df.columns.astype(str).str.strip()
    
    # Extrai as colunas garantindo que sejam numéricas
    pts = df[['b*', 'L*']].apply(pd.to_numeric, errors='coerce').dropna().values
    
    if len(pts) > 5:
        hull = ConvexHull(pts)
        # Fecha o polígono conectando o último ponto ao primeiro
        v = np.append(hull.vertices, hull.vertices[0])
        b_hull, L_hull = pts[v, 0].tolist(), pts[v, 1].tolist()
        tem_dados = True
        print("✅ Base de dados real (Envelope) carregada com sucesso!")
        
except Exception as e:
    print(f"❌ Aviso: Falha ao carregar a base de dados. O envelope não será desenhado. Erro: {e}")
# ==========================================
# 5. VARIÁVEIS DE ESTADO E CALLBACK DA UI
# ==========================================
hist_b, hist_L = [], []

def atualizar_simulacao(sender, app_data):
    mel = dpg.get_value("slider_mel")
    esp = dpg.get_value("slider_esp")
    san = dpg.get_value("slider_san")
    sat = dpg.get_value("slider_sat")
    sd = dpg.get_value("slider_sd")
    caro = dpg.get_value("slider_caro")
    bili = dpg.get_value("slider_bili")
    
    R = calcular_reflectancia(mel, esp, san, sat, sd, caro, bili)
    
    X = k_norm * np.sum(R * S_ilum * x_bar * delta_lambda)
    Y = k_norm * np.sum(R * S_ilum * y_bar * delta_lambda)
    Z = k_norm * np.sum(R * S_ilum * z_bar * delta_lambda)
    
    L, a, b = xyz_para_lab(X, Y, Z)
    rgb = xyz_para_rgb(X, Y, Z)
    
    dpg.set_value("line_r", [Lambdas.tolist(), R.tolist()])
    dpg.set_value("txt_lab", f"L* = {L:.1f} | a* = {a:.1f} | b* = {b:.1f}")
    dpg.configure_item("rect_color", fill=rgb + [255])
    
    global hist_b, hist_L
    if not hist_b or ((b - hist_b[-1])**2 + (L - hist_L[-1])**2)**0.5 > 0.1:
        hist_b.append(b)
        hist_L.append(L)
        if len(hist_b) > 15:
            hist_b.pop(0)
            hist_L.pop(0)
            
    dpg.set_value("line_trace", [hist_b, hist_L])
    dpg.set_value("scat_now", [[b], [L]])

# ==========================================
# 6. CONSTRUÇÃO DA INTERFACE COM DEARPYGUI
# ==========================================
dpg.create_context()

with dpg.window(label="Simulador de Cor de Pele - FAMB/USP", width=1200, height=850):
    with dpg.group(horizontal=True):
        
        # --- PAINEL ESQUERDO ---
        with dpg.child_window(width=580):
            dpg.add_text("Parâmetros Fisiológicos", color=[255, 200, 0])
            dpg.add_slider_float(label="Peso Melanina", tag="slider_mel", default_value=1.0, min_value=0.1, max_value=8.0, callback=atualizar_simulacao)
            dpg.add_slider_float(label="Espes. Epiderme (mm)", tag="slider_esp", default_value=0.05, min_value=0.007, max_value=0.15, callback=atualizar_simulacao)
            dpg.add_slider_float(label="Fraç. Sangue", tag="slider_san", default_value=0.02, min_value=0.001, max_value=0.15, callback=atualizar_simulacao)
            dpg.add_slider_float(label="Saturação O2", tag="slider_sat", default_value=0.70, min_value=0.0, max_value=1.0, callback=atualizar_simulacao)
            dpg.add_slider_float(label="Espalhamento Derme", tag="slider_sd", default_value=1.0, min_value=0.1, max_value=3.0, callback=atualizar_simulacao)
            dpg.add_slider_float(label="Caroteno (uM)", tag="slider_caro", default_value=2.0, min_value=0.0, max_value=50.0, callback=atualizar_simulacao)
            dpg.add_slider_float(label="Bilirrubina (Derme)", tag="slider_bili", default_value=0.0, min_value=0.0, max_value=5.0, callback=atualizar_simulacao)
            
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            with dpg.group(horizontal=True):
                with dpg.drawlist(width=100, height=100):
                    dpg.draw_rectangle([0, 0], [100, 100], tag="rect_color", fill=[255, 200, 150, 255])
                dpg.add_text("Valores CIELAB", tag="txt_lab")
            
            dpg.add_spacer(height=10)
            
            with dpg.plot(label="Espectro de Reflectância", height=320, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Comprimento de Onda (nm)")
                with dpg.plot_axis(dpg.mvYAxis, label="R", tag="y_axis_r"):
                    dpg.set_axis_limits("y_axis_r", 0, 1.0)
                    # Alocação segura com zeros para evitar SystemError
                    dpg.add_line_series(Lambdas.tolist(), [0.0]*len(Lambdas), label="Espectro Simulado", tag="line_r")

        # --- PAINEL DIREITO ---
        with dpg.child_window():
            with dpg.plot(label="Espaço de Cor (b* vs L*)", height=-1, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="b* (Azul -> Amarelo)")
                dpg.set_axis_limits(dpg.last_item(), -5, 50)
                
                with dpg.plot_axis(dpg.mvYAxis, label="L* (Escuro -> Claro)"):
                    dpg.set_axis_limits(dpg.last_item(), 0, 100)
                    if tem_dados:
                        # Envelope real desenhado sem o argumento color
                        dpg.add_line_series(b_hull, L_hull, label="Envelope Real (Dados)", tag="line_hull")
                    
                    # Alocação segura com dummy data
                    dpg.add_line_series([0.0], [0.0], label="Rastro", tag="line_trace")
                    dpg.add_scatter_series([0.0], [0.0], label="Paciente Atual", tag="scat_now")

# Inicialização forçada para preencher os gráficos no primeiro frame
atualizar_simulacao(None, None)

dpg.create_viewport(title='Simulador Analítico de Pele V2.0', width=1220, height=820)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()