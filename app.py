"""
================================================================================
ODE Odyssey 🔬 - Le Compagnon Interactif des Équations Différentielles en MPSI/PCSI
Programmes couverts : Électrocinétique (RC, RL, RLC), Mécanique du Point (Glycérol,
Pendule Simple, 2 Ressorts), Énergétique (Double Puits de Landau).
Stack : Streamlit, SymPy, SciPy, Plotly, NumPy
================================================================================
"""

import streamlit as st
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------------------
# Configuration Streamlit
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="ODE Odyssey - Labo Prépa MPSI / PCSI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# CSS Custom : Flashcards colorées, typographie scientifique et badges de cours
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    /* Headers de Flashcards avec barres colorées */
    .card-header-blue {
        border-left: 5px solid #00d2ff;
        padding-left: 12px;
        margin-bottom: 10px;
    }
    .card-header-green {
        border-left: 5px solid #2ecc71;
        padding-left: 12px;
        margin-bottom: 10px;
    }
    .card-header-purple {
        border-left: 5px solid #9b59b6;
        padding-left: 12px;
        margin-bottom: 10px;
    }
    .card-header-orange {
        border-left: 5px solid #e67e22;
        padding-left: 12px;
        margin-bottom: 10px;
    }
    .badge-cours {
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# SÉLECTION DU SYSTÈME PARMI LES GRANDS CLASSIQUES DU COURS DE MPSI
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ Le Labo de Prépa")
    st.caption("Sélectionnez directement une situation vue en cours ou en TD :")

    categorie = st.selectbox(
        "Chapitre du cours :",
        [
            "🔌 1. Électrocinétique (RC, RL, RLC)",
            "💧 2. Dynamique Newtonienne (Chute dans le Glycérol)",
            "⏳ 3. Oscillateurs (Pendule Simple & 2 Ressorts)",
            "⚡ 4. Énergétique (Double Puits de Landau)"
        ]
    )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # CATEGORIE 1 : ÉLECTROCINÉTIQUE
    # --------------------------------------------------------------------------
    if categorie.startswith("🔌"):
        choix_modele = st.selectbox(
            "Exercice de cours :",
            [
                "Circuit RC Série (Charge d'un condensateur, 1er ordre)",
                "Circuit RL Série (Établissement du courant, 1er ordre)",
                "Circuit RLC Série (Oscillations libres amorties, 2nd ordre)"
            ]
        )

        if "RC" in choix_modele:
            type_sys = "RC"
            R_val = st.number_input("Résistance R (kΩ)", min_value=0.1, max_value=500.0, value=10.0, step=1.0) * 1e3
            C_val = st.number_input("Capacité C (µF)", min_value=0.1, max_value=1000.0, value=100.0, step=10.0) * 1e-6
            E_gen = st.number_input("Tension du générateur E (V)", value=5.0, step=0.5)
            u0 = st.number_input("Tension initiale u_C(0) (V)", value=0.0, step=0.5)
            tau_phys = R_val * C_val
            t_max = st.slider("Durée d'observation (s)", min_value=float(0.5*tau_phys), max_value=float(7.0*tau_phys), value=float(5.0*tau_phys))
            nom_var, unite_var = "Tension u_C", "V"
            nom_vit, unite_vit = "Dérivée du_C/dt", "V/s"

        elif "RL" in choix_modele:
            type_sys = "RL"
            R_val = st.number_input("Résistance R (Ω)", min_value=1.0, max_value=1000.0, value=50.0, step=5.0)
            L_val = st.number_input("Inductance L (mH)", min_value=1.0, max_value=5000.0, value=200.0, step=10.0) * 1e-3
            E_gen = st.number_input("Tension générateur E (V)", value=10.0, step=1.0)
            i0 = st.number_input("Courant initial i(0) (A)", value=0.0, step=0.05)
            tau_phys = L_val / R_val
            t_max = st.slider("Durée d'observation (s)", min_value=float(0.5*tau_phys), max_value=float(7.0*tau_phys), value=float(5.0*tau_phys))
            nom_var, unite_var = "Intensité i", "A"
            nom_vit, unite_vit = "Dérivée di/dt", "A/s"

        else: # RLC
            type_sys = "RLC"
            L_val = st.number_input("Inductance L (H)", min_value=0.01, max_value=5.0, value=0.1, step=0.05)
            C_val = st.number_input("Capacité C (µF)", min_value=0.5, max_value=500.0, value=20.0, step=5.0) * 1e-6
            R_crit = 2.0 * np.sqrt(L_val / C_val)
            st.info(f"💡 Résistance critique R_c = 2√(L/C) = {R_crit:.1f} Ω")
            R_val = st.number_input("Résistance R (Ω)", min_value=0.0, max_value=1500.0, value=20.0, step=5.0)
            u0 = st.number_input("Tension initiale u_C(0) (V)", value=5.0, step=0.5)
            i0 = st.number_input("Courant initial i(0) (A)", value=0.0, step=0.05)
            w0_phys = 1.0 / np.sqrt(L_val * C_val)
            T0_phys = 2.0 * np.pi / w0_phys
            t_max = st.slider("Durée d'observation (s)", min_value=float(0.005), max_value=float(10*T0_phys), value=float(6*T0_phys))
            nom_var, unite_var = "Tension u_C", "V"
            nom_vit, unite_vit = "Courant i / C (V/s)", "V/s"

    # --------------------------------------------------------------------------
    # CATEGORIE 2 : DYNAMIQUE NEWTONIENNE (CHUTE DANS LE GLYCÉROL)
    # --------------------------------------------------------------------------
    elif categorie.startswith("💧"):
        type_sys = "GLYCEROL"
        st.caption("Une bille d'acier chute dans une éprouvette de glycérol (frottement fluide de Stokes f = -6πηR v).")
        r_bille_mm = st.number_input("Rayon de la bille r (mm)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
        r_bille = r_bille_mm * 1e-3
        rho_acier = 7800.0   # kg/m^3
        rho_glyc = 1260.0    # kg/m^3
        eta_glyc = 1.412     # Pa.s (viscosité dynamique du glycérol à 20°C)
        g_acc = 9.81
        
        # Calculs physiques automatiques
        V_bille = (4.0/3.0) * np.pi * (r_bille**3)
        m_bille = rho_acier * V_bille
        Poids_app = (rho_acier - rho_glyc) * V_bille * g_acc  # Poids - Poussée d'Archimède
        h_stokes = 6.0 * np.pi * eta_glyc * r_bille
        v_lim_theo = Poids_app / h_stokes
        tau_glyc = m_bille / h_stokes

        st.metric("Vitesse limite théorique", f"{v_lim_theo*100:.2f} cm/s")
        st.metric("Temps caractéristique τ", f"{tau_glyc*1000:.1f} ms")

        v0_init = st.number_input("Vitesse initiale v(0) (m/s)", value=0.0, step=0.02)
        t_max = st.slider("Durée de chute (s)", min_value=float(2*tau_glyc), max_value=float(10*tau_glyc), value=float(5*tau_glyc), step=float(tau_glyc/5))
        nom_var, unite_var = "Vitesse v", "m/s"
        nom_vit, unite_vit = "Accélération a", "m/s²"

    # --------------------------------------------------------------------------
    # CATEGORIE 3 : OSCILLATEURS (PENDULE SIMPLE & 2 RESSORTS)
    # --------------------------------------------------------------------------
    elif categorie.startswith("⏳"):
        choix_modele = st.selectbox(
            "Système oscillant :",
            [
                "Pendule Simple (Comparaison linéaire θ vs non-linéaire sin θ)",
                "Masse entre deux ressorts identiques (Oscillateur harmonique)"
            ]
        )
        if "Pendule" in choix_modele:
            type_sys = "PENDULE"
            L_pend = st.number_input("Longueur de la tige L (m)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
            m_pend = st.number_input("Masse de la masselotte m (kg)", min_value=0.05, max_value=5.0, value=0.5, step=0.05)
            c_frott = st.number_input("Frottement de l'air c (N·s/m)", min_value=0.0, max_value=1.5, value=0.08, step=0.01)
            theta0_deg = st.slider("Angle initial θ(0) (en degrés)", min_value=5, max_value=175, value=45, step=5)
            theta0 = np.radians(theta0_deg)
            omega0_init = st.number_input("Vitesse angulaire initiale θ'(0) (rad/s)", value=0.0, step=0.1)
            g_acc = 9.81
            w0_pend = np.sqrt(g_acc / L_pend)
            t_max = st.slider("Durée d'observation (s)", min_value=2.0, max_value=30.0, value=12.0, step=1.0)
            nom_var, unite_var = "Élongation angulaire θ", "rad"
            nom_vit, unite_vit = "Vitesse angulaire θ'", "rad/s"

        else: # Deux ressorts
            type_sys = "DEUX_RESSORTS"
            m_res = st.number_input("Masse centrale m (kg)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            k_res = st.number_input("Raideur de chaque ressort k (N/m)", min_value=1.0, max_value=100.0, value=15.0, step=1.0)
            c_res = st.number_input("Frottement c (N·s/m)", min_value=0.0, max_value=5.0, value=0.3, step=0.1)
            st.caption("Deux ressorts en parallèle ou opposés confèrent une raideur équivalente k_eq = 2k.")
            y0_res = st.number_input("Déplacement initial y(0) (cm)", value=10.0, step=1.0) * 1e-2
            v0_res = st.number_input("Vitesse initiale y'(0) (m/s)", value=0.0, step=0.1)
            t_max = st.slider("Durée d'observation (s)", min_value=2.0, max_value=25.0, value=12.0, step=1.0)
            nom_var, unite_var = "Position y", "m"
            nom_vit, unite_vit = "Vitesse y'", "m/s"

    # --------------------------------------------------------------------------
    # CATEGORIE 4 : ÉNERGÉTIQUE (PUITS DE LANDAU)
    # --------------------------------------------------------------------------
    else:
        type_sys = "LANDAU"
        st.caption("Puits double potentiel V(x) = -a/2 x² + b/4 x⁴ (Classique pour illustrer la bifurcation et les oscillations autour des deux équilibres stables).")
        a_landau = st.number_input("Paramètre de barrière a", min_value=0.5, max_value=10.0, value=4.0, step=0.5)
        b_landau = st.number_input("Paramètre de confinement b", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
        gamma_l = st.number_input("Frottement fluide γ", min_value=0.0, max_value=2.0, value=0.2, step=0.05)
        x_eq_theo = np.sqrt(a_landau / b_landau)
        st.info(f"💡 Équilibres stables : x = ±{x_eq_theo:.2f} | Équilibre instable : x = 0")
        x0_landau = st.slider("Position initiale x(0)", min_value=-2.5, max_value=2.5, value=0.2, step=0.1)
        v0_landau = st.slider("Vitesse initiale v(0)", min_value=-3.0, max_value=3.0, value=1.0, step=0.1)
        t_max = st.slider("Durée d'observation (s)", min_value=5.0, max_value=40.0, value=20.0, step=1.0)
        nom_var, unite_var = "Position x", "u.a."
        nom_vit, unite_vit = "Vitesse x'", "u.a./s"

    t_eval = np.linspace(0, t_max, 1200)


# ==============================================================================
# RÉSOLUTION NUMÉRIQUE ET SYMBOLIQUE SPÉCIFIQUE
# ==============================================================================
t_s = sp.Symbol('t', real=True, positive=True)

if type_sys == "RC":
    # u_C' + u_C/tau = E/tau
    tau_num = R_val * C_val
    sol_num_y = E_gen + (u0 - E_gen) * np.exp(-t_eval / tau_num)
    sol_num_v = -(u0 - E_gen) / tau_num * np.exp(-t_eval / tau_num)
    
    # Formule analytique propre
    u_sym = sp.Function('u')(t_s)
    sol_exacte_latex = f"E + (u_0 - E) e^{{-t/\\tau}} = {E_gen:.1f} + ({u0 - E_gen:.1f}) e^{{-t / {tau_num:.3f}}}"

elif type_sys == "RL":
    # i' + (R/L) i = E/L
    tau_num = L_val / R_val
    i_inf = E_gen / R_val
    sol_num_y = i_inf + (i0 - i_inf) * np.exp(-t_eval / tau_num)
    sol_num_v = -(i0 - i_inf) / tau_num * np.exp(-t_eval / tau_num)
    sol_exacte_latex = f"\\frac{{E}}{{R}} + \\left(i_0 - \\frac{{E}}{{R}}\\right) e^{{-t/\\tau}} = {i_inf:.2f} + ({i0 - i_inf:.2f}) e^{{-t / {tau_num:.3f}}}"

elif type_sys == "GLYCEROL":
    # m v' = Poids_app - h v => v' + v/tau = v_lim/tau
    sol_num_y = v_lim_theo + (v0_init - v_lim_theo) * np.exp(-t_eval / tau_glyc)
    sol_num_v = -(v0_init - v_lim_theo) / tau_glyc * np.exp(-t_eval / tau_glyc)
    sol_exacte_latex = f"v_{{\\lim}} + (v_0 - v_{{\\lim}}) e^{{-t/\\tau}} = {v_lim_theo:.3f} + ({v0_init - v_lim_theo:.3f}) e^{{-t / {tau_glyc:.4f}}}"

elif type_sys == "RLC":
    w0_sq = 1.0 / (L_val * C_val)
    w0 = np.sqrt(w0_sq)
    lambd = R_val / (2.0 * L_val)
    delta_r = lambd**2 - w0_sq
    Q_val = w0 / (2.0 * lambd) if lambd > 0 else np.inf

    def rlc_derivs(t, state):
        u, du = state
        return [du, -2*lambd*du - w0_sq*u]

    res_ivp = solve_ivp(rlc_derivs, (0, t_max), [u0, -i0/C_val], t_eval=t_eval, method='RK45')
    sol_num_y = res_ivp.y[0]
    sol_num_v = res_ivp.y[1]

    if delta_r < -1e-9:
        w_p = np.sqrt(-delta_r)
        sol_exacte_latex = f"e^{{-{lambd:.2f} t}} \\left( {u0:.2f} \\cos({w_p:.2f} t) + A_2 \\sin({w_p:.2f} t) \\right)"
    elif abs(delta_r) <= 1e-9:
        sol_exacte_latex = f"({u0:.2f} + B t) e^{{-{lambd:.2f} t}}"
    else:
        sol_exacte_latex = f"A e^{{-{lambd - np.sqrt(delta_r):.2f} t}} + B e^{{-{lambd + np.sqrt(delta_r):.2f} t}}"

elif type_sys == "PENDULE":
    w0 = np.sqrt(g_acc / L_pend)
    gamma_p = c_frott / m_pend

    # 1. Résolution exacte non-linéaire (SciPy RK45)
    def pendule_derivs(t, s):
        th, dth = s
        return [dth, -gamma_p*dth - (w0**2)*np.sin(th)]

    res_ivp = solve_ivp(pendule_derivs, (0, t_max), [theta0, omega0_init], t_eval=t_eval, method='RK45')
    sol_num_y = res_ivp.y[0]
    sol_num_v = res_ivp.y[1]

    # 2. Modèle linéaire pour comparaison
    lambd = gamma_p / 2.0
    w_p = np.sqrt(w0**2 - lambd**2) if w0 > lambd else 1.0
    sol_lineaire = theta0 * np.exp(-lambd * t_eval) * np.cos(w_p * t_eval)
    sol_exacte_latex = f"\\theta(t) \\approx \\theta_0 e^{{-\\lambda t}} \\cos(\\omega t) = {theta0:.2f} e^{{-{lambd:.2f} t}} \\cos({w_p:.2f} t)"

elif type_sys == "DEUX_RESSORTS":
    k_eq = 2.0 * k_res
    w0 = np.sqrt(k_eq / m_res)
    lambd = c_res / (2.0 * m_res)
    delta_r = lambd**2 - w0**2

    def deux_res_derivs(t, s):
        pos, vel = s
        return [vel, -2*lambd*vel - (w0**2)*pos]

    res_ivp = solve_ivp(deux_res_derivs, (0, t_max), [y0_res, v0_res], t_eval=t_eval, method='RK45')
    sol_num_y = res_ivp.y[0]
    sol_num_v = res_ivp.y[1]
    w_p = np.sqrt(w0**2 - lambd**2) if w0 > lambd else 1.0
    sol_exacte_latex = f"y(t) = y_0 e^{{-\\lambda t}} \\cos(\\omega t) = {y0_res*100:.1f} e^{{-{lambd:.2f} t}} \\cos({w_p:.2f} t) \\text{{ [cm]}}"

else: # LANDAU
    # x'' + gamma x' - a x + b x^3 = 0
    def landau_derivs(t, s):
        pos, vel = s
        return [vel, -gamma_l*vel + a_landau*pos - b_landau*(pos**3)]

    res_ivp = solve_ivp(landau_derivs, (0, t_max), [x0_landau, v0_landau], t_eval=t_eval, method='RK45')
    sol_num_y = res_ivp.y[0]
    sol_num_v = res_ivp.y[1]
    sol_exacte_latex = "\\text{Non-linéaire (Équation de Duffing/Landau sans solution analytique élémentaire)}"


# ==============================================================================
# EN-TÊTE PRINCIPAL ET BADGES
# ==============================================================================
st.title("⚡ ODE Odyssey &middot; Le Solveur Pédagogique MPSI")
st.markdown(
    f"<div class='badge-cours' style='background-color:#1b4f72; color:#aed6f1; border:1px solid #3498db;'>"
    f"🎓 Chapitre sélectionné : {categorie}</div>",
    unsafe_allow_html=True
)

# Onglets principaux
tab_fc, tab_3d, tab_2d, tab_cours = st.tabs([
    "🗂️ Flashcards Interactives du Cours",
    "🌌 Espace des Phases 3D",
    "📈 Chronogrammes 2D & Portraits de Phase",
    "📚 Synthèse & Fiches Mémo de Concours"
])


# ==============================================================================
# ONGLET 1 : FLASHCARDS DU COURS DE PRÉPA
# ==============================================================================
with tab_fc:
    st.subheader("📚 Les Flashcards Clés à Retenir")
    st.info("💡 Les formules sont désormais compilées avec le moteur KaTeX officiel de la prépa : nettes, centrées et sans code parasite !")

    col1, col2 = st.columns(2)

    with col1:
        # FLASHCARD 1
        with st.container(border=True):
            st.markdown("<div class='card-header-blue'><h4 style='margin:0; color:#00d2ff;'>📌 Flashcard 1 : L'Équation Différentielle Canonique</h4></div>", unsafe_allow_html=True)
            if type_sys == "RC":
                st.latex(r"\frac{\mathrm{d}u_C}{\mathrm{d}t} + \frac{1}{\tau} u_C(t) = \frac{E}{\tau} \quad \text{avec} \quad \tau = RC")
                st.markdown(f"**Application numérique :** $\\tau = RC = {tau_num:.3f} \\text{{ s}}$")
                st.markdown("---")
                st.markdown("**🔍 Signification des lettres :**")
                st.markdown(f"• **$u_C(t)$** : Tension aux bornes du condensateur ({unite_var})\n\n"
                            f"• **$R = {R_val/1e3:.1f} \\text{{ k}}\Omega$** : Résistance (limite le passage des électrons)\n\n"
                            f"• **$C = {C_val*1e6:.1f} \\;\\mu\\text{{F}}$** : Capacité (réservoir de charges)\n\n"
                            f"• **$\\tau = {tau_num:.3f} \\text{{ s}}$** : Constante de temps caractéristique")

            elif type_sys == "RL":
                st.latex(r"\frac{\mathrm{d}i}{\mathrm{d}t} + \frac{1}{\tau} i(t) = \frac{E}{L} \quad \text{avec} \quad \tau = \frac{L}{R}")
                st.markdown(f"**Application numérique :** $\\tau = \\frac{{L}}{{R}} = {tau_num*1000:.1f} \\text{{ ms}}$")
                st.markdown("---")
                st.markdown("**🔍 Signification des lettres :**")
                st.markdown(f"• **$i(t)$** : Intensité dans la bobine ({unite_var})\n\n"
                            f"• **$L = {L_val*1e3:.1f} \\text{{ mH}}$** : Inductance (inertie face aux variations de courant)\n\n"
                            f"• **$R = {R_val:.1f} \\;\Omega$** : Résistance dissipative\n\n"
                            f"• **$\\tau = {tau_num*1000:.1f} \\text{{ ms}}$** : Temps d'établissement")

            elif type_sys == "GLYCEROL":
                st.latex(r"m \frac{\mathrm{d}v}{\mathrm{d}t} = (\rho_{\text{acier}} - \rho_{\text{glyc}}) V g - 6\pi\eta r v")
                st.latex(r"\frac{\mathrm{d}v}{\mathrm{d}t} + \frac{1}{\tau} v = \frac{v_{\lim}}{\tau}")
                st.markdown(f"**Vitesse limite :** $v_{{\\lim}} = {v_lim_theo*100:.2f} \\text{{ cm/s}} \\quad | \\quad \\tau = {tau_glyc*1000:.1f} \\text{{ ms}}$")
                st.markdown("---")
                st.markdown("**🔍 Les 3 forces du bilan PFD :**")
                st.markdown("• **Poids $\vec{P}$** : Force motrice vers le bas\n\n"
                            "• **Poussée d'Archimède $\vec{\Pi}_A$** : Force vers le haut (liquide déplacé)\n\n"
                            "• **Frottement de Stokes $\vec{f} = -6\pi\eta r \vec{v}$** : Résistance visqueuse")

            elif type_sys == "PENDULE":
                st.latex(r"\frac{\mathrm{d}^2 \theta}{\mathrm{d}t^2} + \frac{c}{m} \frac{\mathrm{d}\theta}{\mathrm{d}t} + \frac{g}{L} \sin\theta = 0")
                st.latex(r"\text{Petites oscillations : } \ddot{\theta} + \omega_0^2 \theta = 0 \quad (\omega_0 = \sqrt{g/L})")
                st.markdown(f"**Pulsation propre :** $\\omega_0 = \\sqrt{{g/L}} = {w0_pend:.2f} \\text{{ rad/s}} \\quad | \\quad T_0 = {2*np.pi/w0_pend:.2f} \\text{{ s}}$")
                st.markdown("---")
                st.markdown("**🔍 Remarque d'oral :**")
                st.markdown("• L'approximation linéaire $\sin\\theta \\approx \\theta$ est valable pour $\\theta \\le 20^\\circ$.\n\n"
                            "• Aux grands angles, la période réelle $T$ grandit (formule de Borda).")

            elif type_sys == "DEUX_RESSORTS":
                st.latex(r"m \frac{\mathrm{d}^2 y}{\mathrm{d}t^2} + c \frac{\mathrm{d}y}{\mathrm{d}t} + 2k y = 0")
                st.latex(r"\omega_0 = \sqrt{\frac{2k}{m}}")
                st.markdown("---")
                st.markdown("**🔍 Pourquoi $2k$ ?**")
                st.markdown("• Quand la masse se déplace de $y$, le ressort 1 est étiré (force $-ky$).\n\n"
                            "• Le ressort 2 est comprimé de $y$ et repousse la masse (force $-ky$).\n\n"
                            "• Force totale : $F = -2ky \\implies k_{\\text{eq}} = 2k$.")

            elif type_sys == "RLC":
                st.latex(r"\frac{\mathrm{d}^2 u_C}{\mathrm{d}t^2} + \frac{\omega_0}{Q} \frac{\mathrm{d}u_C}{\mathrm{d}t} + \omega_0^2 u_C = 0")
                st.latex(rf"\omega_0 = {w0:.1f} \text{{ rad/s}} \quad | \quad Q = {Q_val:.2f} \quad | \quad R_c = {R_crit:.1f} \;\Omega")

            else: # LANDAU
                st.latex(r"\frac{\mathrm{d}^2 x}{\mathrm{d}t^2} + \gamma \frac{\mathrm{d}x}{\mathrm{d}t} - a x + b x^3 = 0")
                st.latex(r"E_p(x) = -\frac{a}{2} x^2 + \frac{b}{4} x^4")
                st.markdown(f"**Équilibres stables :** $x = \\pm \\sqrt{{a/b}} = \\pm {x_eq_theo:.2f}$ | **Instable :** $x = 0$")

        # FLASHCARD 2
        with st.container(border=True):
            st.markdown("<div class='card-header-green'><h4 style='margin:0; color:#2ecc71;'>🎯 Flashcard 2 : Propriétés Mathématiques & Stabilité</h4></div>", unsafe_allow_html=True)
            if type_sys in ["RC", "RL", "GLYCEROL"]:
                st.latex(r"y(t) = y_\infty + (y_0 - y_\infty) e^{-\frac{t}{\tau}}")
                st.markdown("• **Théorème de Cauchy-Lipschitz :** Avec une condition initiale fixée, la solution est unique.\n\n"
                            "• **Pas d'oscillation possible** dans un système du premier ordre linéaire autonome.\n\n"
                            "• **Stabilité :** Le système converge toujours vers son état permanent car $\\tau > 0$.")
            elif type_sys == "PENDULE":
                st.latex(r"\Delta' = \lambda^2 - \omega_0^2 < 0 \implies r_{1,2} = -\lambda \pm i\omega")
                st.markdown("• **Espace des phases :** Les trajectoires spiralent vers $(0,0)$ si amortissement, ou forment des ellipses fermées si $c=0$.\n\n"
                            "• **Séparatrice :** À $\\theta = \\pm \\pi$, il s'agit d'un point col (équilibre instable).")
            else:
                st.latex(r"r^2 + 2\lambda r + \omega_0^2 = 0 \implies \Delta' = \lambda^2 - \omega_0^2")
                st.markdown("• **Si $\\Delta' < 0$ :** Régime pseudo-périodique oscillant amorti.\n\n"
                            "• **Si $\\Delta' = 0$ :** Régime critique (temps de réponse minimal).\n\n"
                            "• **Si $\\Delta' > 0$ :** Régime apériodique (sur-amorti).")

    with col2:
        # FLASHCARD 3
        with st.container(border=True):
            st.markdown("<div class='card-header-purple'><h4 style='margin:0; color:#9b59b6;'>✨ Flashcard 3 : La Solution Analytique du Cours</h4></div>", unsafe_allow_html=True)
            st.markdown("*Formule exacte pour votre copie ou votre colle :*")
            if type_sys == "RC":
                st.latex(r"u_C(t) = E + (u_0 - E) e^{-\frac{t}{\tau}}")
                st.latex(rf"u_C(t) = {E_gen:.1f} + ({u0 - E_gen:.1f}) e^{{-\frac{{t}}{{{tau_num:.3f}}}}}")
            elif type_sys == "RL":
                st.latex(r"i(t) = \frac{E}{R} + \left(i_0 - \frac{E}{R}\right) e^{-\frac{t}{\tau}}")
                st.latex(rf"i(t) = {i_inf:.2f} + ({i0 - i_inf:.2f}) e^{{-\frac{{t}}{{{tau_num:.4f}}}}}")
            elif type_sys == "GLYCEROL":
                st.latex(r"v(t) = v_{\lim} + (v_0 - v_{\lim}) e^{-\frac{t}{\tau}}")
                st.latex(rf"v(t) = {v_lim_theo:.3f} + ({v0_init - v_lim_theo:.3f}) e^{{-\frac{{t}}{{{tau_glyc:.4f}}}}}")
            elif type_sys == "PENDULE":
                st.latex(r"\theta(t) \approx \theta_0 e^{-\lambda t} \cos(\omega t)")
                st.latex(rf"\theta(t) \approx {theta0:.2f} e^{{-{lambd:.2f} t}} \cos({w_p:.2f} t)")
            elif type_sys == "DEUX_RESSORTS":
                st.latex(rf"y(t) = {y0_res*100:.1f} e^{{-{lambd:.2f} t}} \cos({w_p:.2f} t) \quad \text{{[cm]}}")
            elif type_sys == "RLC":
                if delta_r < -1e-9:
                    st.latex(rf"u_C(t) = e^{{-{lambd:.2f} t}} \left[ {u0:.2f} \cos({w_p:.2f} t) + A_2 \sin({w_p:.2f} t) \right]")
                elif abs(delta_r) <= 1e-9:
                    st.latex(rf"u_C(t) = ({u0:.2f} + B t) e^{{-{lambd:.2f} t}}")
                else:
                    st.latex(r"u_C(t) = A e^{r_1 t} + B e^{r_2 t}")
            else:
                st.latex(r"\text{Résolution numérique RK45 haute précision}")

            st.markdown("---")
            st.markdown("**💡 Les 3 pourcentages à connaître par cœur :**")
            st.markdown("• À **$t = \\tau$** : **63,2 %** du régime transitoire achevé.\n\n"
                        "• À **$t = 3\\tau$** : **95,0 %** (le système est stabilisé).\n\n"
                        "• À **$t = 5\\tau$** : **99,3 %** (régime permanent quasiment atteint).")

        # FLASHCARD 4
        with st.container(border=True):
            st.markdown("<div class='card-header-orange'><h4 style='margin:0; color:#e67e22;'>📐 Flashcard 4 : Méthode de la Tangente à l'Origine</h4></div>", unsafe_allow_html=True)
            st.latex(r"\text{Pente initiale : } \left. \frac{\mathrm{d}y}{\mathrm{d}t} \right|_{t=0} = \frac{y_\infty - y_0}{\tau}")
            st.markdown("---")
            st.markdown("**🎯 Comment l'utiliser sur une copie de concours :**")
            st.markdown("1. Tracez l'asymptote horizontale finale $y = y_\\infty$.\n\n"
                        "2. Tracez la tangente à la courbe au point de départ $(t=0, y=y_0)$.\n\n"
                        "3. L'intersection de cette tangente et de l'asymptote se projette sur l'axe des temps **exactement en $t = \\tau$** !")


# ==============================================================================
# ONGLET 2 : VISUALISATION 3D DANS L'ESPACE DES PHASES
# ==============================================================================
with tab_3d:
    st.subheader(f"🌌 Trajectoire 3D dans l'Espace des Phases Étendu $(t, {nom_var}, {nom_vit})$")
    st.caption("Pivotez avec la souris. La courbe 3D se déroule dans le temps sans jamais s'auto-croiser.")

    t_cursor = st.slider("⏱️ Déplacez le curseur pour inspecter un instant précis t* :", 0.0, float(t_max), float(min(1.0, t_max)), step=float(t_max/100))
    idx_cur = int(np.argmin(np.abs(t_eval - t_cursor)))
    y_c, v_c = sol_num_y[idx_cur], sol_num_v[idx_cur]

    y_min, y_max = np.min(sol_num_y), np.max(sol_num_y)
    v_min, v_max = np.min(sol_num_v), np.max(sol_num_v)
    pad_y = (y_max - y_min) * 0.15 if y_max != y_min else 1.0
    pad_v = (v_max - v_min) * 0.15 if v_max != v_min else 1.0

    fig_3d = go.Figure()

    # Trajectoire 3D
    fig_3d.add_trace(go.Scatter3d(
        x=t_eval, y=sol_num_y, z=sol_num_v,
        mode='lines',
        name='Trajectoire d\'état',
        line=dict(color=t_eval, colorscale='Turbo', width=6, colorbar=dict(title="Temps (s)", thickness=15))
    ))

    # Marqueur initial
    fig_3d.add_trace(go.Scatter3d(
        x=[0], y=[sol_num_y[0]], z=[sol_num_v[0]],
        mode='markers+text',
        name='État initial',
        text=["🟢 Départ (t=0)"],
        textposition="top center",
        marker=dict(size=8, color='#00ff88', symbol='diamond')
    ))

    # Marqueur curseur
    fig_3d.add_trace(go.Scatter3d(
        x=[t_cursor], y=[y_c], z=[v_c],
        mode='markers+text',
        name='Point inspecté',
        text=[f"📍 {y_c:.2f} {unite_var} | {v_c:.2f} {unite_vit}"],
        textposition="top right",
        marker=dict(size=10, color='#ff007f', symbol='circle')
    ))

    # Ombres au sol et au mur
    fig_3d.add_trace(go.Scatter3d(
        x=t_eval, y=sol_num_y, z=np.full_like(t_eval, v_min - pad_v),
        mode='lines', name=f'Ombre {nom_var}(t)',
        line=dict(color='rgba(150, 150, 150, 0.35)', width=3, dash='dot')
    ))
    fig_3d.add_trace(go.Scatter3d(
        x=np.full_like(t_eval, t_max), y=sol_num_y, z=sol_num_v,
        mode='lines', name=f'Ombre Portrait de Phase',
        line=dict(color='rgba(255, 165, 0, 0.45)', width=3, dash='dot')
    ))

    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(title='Temps t (s)', backgroundcolor='#161922'),
            yaxis=dict(title=f'{nom_var} ({unite_var})', backgroundcolor='#161922'),
            zaxis=dict(title=f"{nom_vit} ({unite_vit})", backgroundcolor='#161922'),
            camera=dict(eye=dict(x=1.7, y=-1.7, z=1.2)),
            aspectratio=dict(x=1.6, y=1, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600
    )
    st.plotly_chart(fig_3d, width='stretch')


# ==============================================================================
# ONGLET 3 : CHRONOGRAMMES 2D ET PORTRAITS DE PHASE
# ==============================================================================
with tab_2d:
    col_2d1, col_2d2 = st.columns(2)

    with col_2d1:
        st.subheader(f"📈 Évolution Temporelle {nom_var}(t)")
        fig_time = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)

        fig_time.add_trace(go.Scatter(
            x=t_eval, y=sol_num_y, mode='lines', name=f'{nom_var}(t)',
            line=dict(color='#00d2ff', width=2.5)
        ), row=1, col=1)

        # Si pendule, on superpose le modèle linéaire pour voir le décrochage aux grands angles
        if type_sys == "PENDULE":
            fig_time.add_trace(go.Scatter(
                x=t_eval, y=sol_lineaire, mode='lines', name='Modèle Linéaire (sin θ ≈ θ)',
                line=dict(color='#e74c3c', width=1.5, dash='dash')
            ), row=1, col=1)

        fig_time.add_trace(go.Scatter(
            x=t_eval, y=sol_num_v, mode='lines', name=f'{nom_vit}(t)',
            line=dict(color='#2ecc71', width=2)
        ), row=2, col=1)

        fig_time.update_yaxes(title_text=f"{nom_var} ({unite_var})", row=1, col=1)
        fig_time.update_yaxes(title_text=f"{nom_vit} ({unite_vit})", row=2, col=1)
        fig_time.update_xaxes(title_text="Temps t (s)", row=2, col=1)
        fig_time.update_layout(height=480, margin=dict(l=40, r=20, t=20, b=20), hovermode='x unified')
        st.plotly_chart(fig_time, width='stretch')

    with col_2d2:
        st.subheader(f"🌀 Portrait de Phase 2D ({nom_var}, {nom_vit})")
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=sol_num_y, y=sol_num_v, mode='lines', name='Trajectoire de phase',
            line=dict(color='#f39c12', width=2.5)
        ))
        fig_p.add_trace(go.Scatter(
            x=[sol_num_y[0]], y=[sol_num_v[0]], mode='markers', name='Départ (t=0)',
            marker=dict(size=10, color='#00ff88', symbol='diamond')
        ))
        fig_p.add_trace(go.Scatter(
            x=[sol_num_y[-1]], y=[sol_num_v[-1]], mode='markers', name='Attracteur / Équilibre',
            marker=dict(size=8, color='red', symbol='x')
        ))
        fig_p.update_layout(
            xaxis=dict(title=f"{nom_var} ({unite_var})"),
            yaxis=dict(title=f"{nom_vit} ({unite_vit})"),
            height=480,
            margin=dict(l=40, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_p, width='stretch')


# ==============================================================================
# ONGLET 4 : SYNTHÈSE DES FICHES DE RÉVISION DU PROGRAMME MPSI/PCSI
# ==============================================================================
with tab_cours:
    st.subheader("📖 Fiches Récapitulatives du Premier Semestre de Prépa")
    
    st.markdown("""
    ### 1. Classification par Ordre des Équations Différentielles

    | Système de Cours | Ordre | Équation Canonique | Temps ou Pulsation Caractéristique |
    | :--- | :---: | :--- | :--- |
    | **Circuit RC** | **1er** | $\\frac{\\mathrm{d}u_C}{\\mathrm{d}t} + \\frac{1}{\\tau} u_C = \\frac{E}{\\tau}$ | $\\tau = RC$ |
    | **Circuit RL** | **1er** | $\\frac{\\mathrm{d}i}{\\mathrm{d}t} + \\frac{1}{\\tau} i = \\frac{E}{L}$ | $\\tau = \\frac{L}{R}$ |
    | **Chute Glycérol (Stokes)** | **1er** | $\\frac{\\mathrm{d}v}{\\mathrm{d}t} + \\frac{1}{\\tau} v = \\frac{v_{\\lim}}{\\tau}$ | $\\tau = \\frac{m}{6\\pi\\eta r}, \\quad v_{\\lim} = \\frac{(m - \\rho_{fl}V)g}{6\\pi\\eta r}$ |
    | **Circuit RLC** | **2nd** | $\\frac{\\mathrm{d}^2 u_C}{\\mathrm{d}t^2} + \\frac{\\omega_0}{Q} \\frac{\\mathrm{d}u_C}{\\mathrm{d}t} + \\omega_0^2 u_C = 0$ | $\\omega_0 = \\frac{1}{\\sqrt{LC}}, \\quad Q = \\frac{1}{R}\\sqrt{\\frac{L}{C}}$ |
    | **Masse-Ressort** | **2nd** | $\\frac{\\mathrm{d}^2 y}{\\mathrm{d}t^2} + 2\\lambda \\frac{\\mathrm{d}y}{\\mathrm{d}t} + \\omega_0^2 y = 0$ | $\\omega_0 = \\sqrt{\\frac{k}{m}}, \\quad Q = \\frac{\\sqrt{mk}}{c}$ |
    | **Deux Ressorts** | **2nd** | $m \\ddot{y} + c \\dot{y} + 2k y = 0$ | $\\omega_0 = \\sqrt{\\frac{2k}{m}}$ |
    | **Pendule Simple** | **2nd** | $\\ddot{\\theta} + \\frac{c}{m}\\dot{\\theta} + \\frac{g}{L} \\sin\\theta = 0$ | $\\omega_0 = \\sqrt{\\frac{g}{L}}, \\quad T_0 = 2\\pi\\sqrt{\\frac{L}{g}}$ |

    ---

    ### 2. Le mot sur les autres thèmes mentionnés :
    - **🔬 Optique géométrique (Lentilles minces) :** En optique géométrique classique, il ne s'agit pas d'équations différentielles mais de **relations algébriques de conjugaison** (Descartes : $\\frac{1}{\\overline{A'A'}} - \\frac{1}{\\overline{OA}} = \\frac{1}{f'}$ ou Newton). L'optique ondulatoire (interférences de Young, diffraction) sera abordée plus tard dans l'année !
    - **⚛️ Le Carotène et la Mécanique Quantique :** C'est le fameux modèle du puits quantique 1D de Kuhn ! Les électrons délocalisés du carotène vérifient l'équation de Schrödinger indépendante du temps :
      $$-\\frac{\\hbar^2}{2m} \\frac{\\mathrm{d}^2 \\psi}{\\mathrm{d}x^2} = E \\psi \\implies \\frac{\\mathrm{d}^2 \\psi}{\\mathrm{d}x^2} + k^2 \\psi = 0$$
      C'est mathématiquement identique à un oscillateur harmonique sans amortissement, avec des conditions aux limites $\\psi(0) = \\psi(L) = 0$ menant à la quantification des niveaux d'énergie !
    - **🚀 La Fusée à eau / Éjection de gaz (Tsiolkovski) :** La masse $m(t)$ varie au cours du temps, ce qui donne $m(t)\\frac{\\mathrm{d}v}{\\mathrm{d}t} = -v_e \\frac{\\mathrm{d}m}{\\mathrm{d}t} - mg$.
    """)
