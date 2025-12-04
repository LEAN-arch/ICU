import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import welch, find_peaks
from scipy.stats import norm, multivariate_normal, wasserstein_distance, linregress, chi2, ks_2samp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import grangercausalitytests
import time
import warnings

# Suppress warnings for cleaner UI
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="TITAN | IRIDIUM COMMAND",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬"
)

class CONFIG:
    COLORS = {
        "bg": "#f8fafc", "card": "#ffffff", "text": "#0f172a", "muted": "#64748b",
        "crit": "#dc2626", "warn": "#d97706", "ok": "#059669",
        "info": "#2563eb", "hemo": "#0891b2", "resp": "#7c3aed", 
        "ai": "#be185d", "drug": "#4f46e5", "anomaly": "#db2777",
        "ext": "#d946ef", "spc": "#059669"
    }
    
    # Physics Constants
    ATM_PRESSURE = 760.0; H2O_PRESSURE = 47.0; R_QUOTIENT = 0.8; MAX_PAO2 = 600.0
    HB_CONVERSION = 1.34; LAC_PROD_THRESH = 330.0; LAC_CLEAR_RATE = 0.05; VCO2_CONST = 130
    
    # L9 Advanced Constants
    BARO_SETPOINT = 85.0; COMPARTMENT_K12 = 0.5; COMPARTMENT_K21 = 0.2
    EC50_NOREPI = 0.5; EMAX_NOREPI = 3000.0
    GFR_BASE = 100.0; AUTOREG_LOWER = 60.0; AUTOREG_UPPER = 150.0
    
    # Drug PK
    DRUG_PK = {
        'norepi': {'svr': 2500.0, 'map': 120.0, 'co': 0.8, 'tau': 2.0, 'tol': 1440.0}, 
        'vaso':   {'svr': 4000.0, 'map': 150.0, 'co': -0.2, 'tau': 5.0, 'tol': 2880.0}, 
        'dobu':   {'svr': -600.0, 'map': 5.0, 'co': 4.5, 'hr': 25.0, 'tau': 3.0, 'tol': 720.0}, 
        'bb':     {'svr': 50.0, 'map': -15.0, 'co': -2.0, 'hr': -35.0, 'tau': 4.0, 'tol': 5000.0}
    }
    
    # SPC & QA Limits
    MAP_LSL = 65.0; MAP_USL = 110.0; CUSUM_H = 4.0; CUSUM_K = 0.5; MEWMA_LAMBDA = 0.3

STYLING = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Roboto+Mono:wght@500;700&display=swap');
    .stApp {{ background-color: {CONFIG.COLORS['bg']}; color: {CONFIG.COLORS['text']}; font-family: 'Inter', sans-serif; }}
    
    div[data-testid="stMetric"] {{ background-color: {CONFIG.COLORS['card']}; border: 1px solid {CONFIG.COLORS['muted']}33; border-radius: 6px; padding: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
    div[data-testid="stMetric"] label {{ color: {CONFIG.COLORS['muted']}; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ font-family: 'Roboto Mono', monospace; font-size: 1.4rem; font-weight: 700; }}
    
    .zone-header {{ font-size: 0.85rem; font-weight: 900; color: {CONFIG.COLORS['text']}; text-transform: uppercase; border-bottom: 2px solid {CONFIG.COLORS['info']}33; margin: 25px 0 10px 0; letter-spacing: 0.05em; }}
    .status-banner {{ padding: 15px; border-radius: 8px; background: {CONFIG.COLORS['card']}; border-left: 6px solid {CONFIG.COLORS['ai']}; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); display: flex; justify-content: space-between; align-items: center; }}
    .clinical-hint {{ font-size: 0.75rem; color: {CONFIG.COLORS['muted']}; background: #f1f5f9; padding: 8px; border-radius: 4px; margin-top: 5px; border-left: 3px solid {CONFIG.COLORS['info']}; }}
    
    .crit-pulse {{ animation: pulse-red 2s infinite; color: {CONFIG.COLORS['crit']}; }}
    @keyframes pulse-red {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} 100% {{ opacity: 1; }} }}
</style>
"""

# ==========================================
# 2. UTILS
# ==========================================
class Utils:
    _rng = np.random.default_rng(42)
    @staticmethod
    def set_seed(seed): Utils._rng = np.random.default_rng(seed)
    @staticmethod
    def brownian_bridge(n, start, end, volatility, noise_type='pink'):
        t = np.linspace(0, 1, n)
        dW = Utils._rng.normal(0, np.sqrt(1/n), n)
        B = start + np.cumsum(dW) - t * (np.cumsum(dW)[-1] - (end - start))
        if noise_type == 'pink': noise = np.convolve(Utils._rng.normal(0, 0.5, n), np.ones(8)/8, mode='same')
        elif noise_type == 'periodic': noise = np.sin(np.linspace(0, n/4, n)) * 2.0
        elif noise_type == 'white': noise = Utils._rng.normal(0, 0.1, n)
        else: noise = np.zeros(n)
        return B + (noise * volatility)

# ==========================================
# 3. PHYSIOLOGY ENGINES
# ==========================================
class Physiology:
    class Autonomic:
        @staticmethod
        def generate(mins, p, is_paced, vent_mode):
            if is_paced: hr = Utils.brownian_bridge(mins, p['hr'][0], p['hr'][0], 0.1, 'white')
            elif vent_mode == 'Control (AC)': hr = Utils.brownian_bridge(mins, p['hr'][0], p['hr'][1], 1.5, 'periodic')
            else: hr = Utils.brownian_bridge(mins, p['hr'][0], p['hr'][1], 1.5, 'pink')
            
            map_r = np.maximum(Utils.brownian_bridge(mins, p['map'][0], p['map'][1], 1.2, 'pink'), 20.0)
            ci = np.maximum(Utils.brownian_bridge(mins, p['ci'][0], p['ci'][1], 0.2, 'pink'), 0.5)
            svri = np.maximum(Utils.brownian_bridge(mins, p['svri'][0], p['svri'][1], 100.0, 'pink'), 100.0)
            rr = np.maximum(Utils.brownian_bridge(mins, 16, 28, 2.0, 'pink'), 4.0)
            return hr, map_r, ci, svri, rr

    class PKPD:
        @staticmethod
        def apply(map_b, ci_b, hr_b, svri_b, drugs, mins):
            t = np.arange(mins)
            e_map, e_ci, e_hr, e_svr = np.zeros(mins), np.zeros(mins), np.zeros(mins), np.zeros(mins)
            for d, dose in drugs.items():
                if dose <= 0: continue
                pk = CONFIG.DRUG_PK.get(d)
                if not pk: continue
                k = (1 - np.exp(-t/pk['tau'])) * np.exp(-t/pk['tol'])
                if 'map' in pk: e_map += dose * pk['map'] * k
                if 'co' in pk: e_ci += dose * pk['co'] * k
                if 'hr' in pk: e_hr += dose * pk['hr'] * k
                if 'svr' in pk: e_svr += dose * pk['svr'] * k
            return np.maximum(map_b+e_map, 10), np.maximum(ci_b+e_ci, 0.1), np.maximum(hr_b+e_hr, 20), np.maximum(svri_b+e_svr, 50)

    class Respiratory:
        @staticmethod
        def exchange(fio2, rr, shunt, peep, mins, copd):
            vd_vt = np.clip(0.3 * copd + (0.1 if copd>1.0 else 0), 0.1, 0.8)
            va = np.maximum((rr * 0.5) * (1 - vd_vt), 0.5)
            paco2 = (CONFIG.VCO2_CONST * 0.863) / va
            p_ideal = np.clip((fio2 * (CONFIG.ATM_PRESSURE - CONFIG.H2O_PRESSURE)) - (paco2 / CONFIG.R_QUOTIENT), 0, CONFIG.MAX_PAO2)
            pao2 = p_ideal * (1 - (shunt * np.exp(-0.08 * peep)))
            spo2 = 100 * (np.maximum(pao2, 0.1)**3 / (np.maximum(pao2, 0.1)**3 + 26**3))
            return pao2, paco2, spo2, np.full(mins, vd_vt)

    class Metabolic:
        @staticmethod
        def calculate(ci, hb, spo2, mins, stress):
            do2i = ci * hb * CONFIG.HB_CONVERSION * (spo2/100) * 10
            vo2i = np.full(mins, 125.0 * stress)
            o2er = vo2i / np.maximum(do2i, 1.0)
            lactate = np.zeros(mins); lac = 1.0
            prod = np.where((do2i < CONFIG.LAC_PROD_THRESH) | (o2er > 0.5), 0.2, 0.0)
            for i in range(mins):
                lac = max(0.6, lac + prod[i] - (CONFIG.LAC_CLEAR_RATE * (ci[i]/2.5)))
                lactate[i] = lac
            return do2i, vo2i, o2er, lactate

class DeepPhysiology:
    """Level 9 Advanced Modules."""
    class Baroreflex:
        @staticmethod
        def compute(map_val):
            symp = 1.0 / (1.0 + np.exp(0.1 * (map_val - CONFIG.BARO_SETPOINT)))
            hr_reflex = 60 + (60 * symp)
            svr_reflex = 800 + (1500 * symp)
            return hr_reflex, svr_reflex, symp
    class Chemoreceptor:
        @staticmethod
        def compute(paco2, pao2):
            drive_co2 = np.maximum(0, paco2 - 45.0) * 2.0
            drive_o2 = 2000 / (pao2 + 0.1) if np.mean(pao2) < 60 else 0
            return np.clip(12 + drive_co2 + (drive_o2 * 0.1), 0, 45)
    class Renal:
        @staticmethod
        def compute(map_val, ci):
            gfr = CONFIG.GFR_BASE * (map_val / CONFIG.AUTOREG_LOWER) if map_val < CONFIG.AUTOREG_LOWER else CONFIG.GFR_BASE
            uo = (gfr * 0.01 * 60) * (ci / 2.5)
            return gfr, uo

class AdvancedPKPD:
    @staticmethod
    def two_compartment(dose_array, mins):
        c1 = np.zeros(mins); c2 = np.zeros(mins); dt=1.0
        for t in range(1, mins):
            dc1 = dose_array[t] - (0.1*c1[t-1]) - (CONFIG.COMPARTMENT_K12*c1[t-1]) + (CONFIG.COMPARTMENT_K21*c2[t-1])
            dc2 = (CONFIG.COMPARTMENT_K12*c1[t-1]) - (CONFIG.COMPARTMENT_K21*c2[t-1])
            c1[t] = c1[t-1] + dc1*dt; c2[t] = c2[t-1] + dc2*dt
        return c1, c2
    @staticmethod
    def hill_effect(conc, emax, ec50, gamma=1.5):
        return (emax * (conc**gamma)) / (ec50**gamma + conc**gamma)

# ==========================================
# 4. ANALYTICS, SPC, QA & AI
# ==========================================
class Analytics:
    @staticmethod
    def signal_forensics(ts, is_paced):
        arr = np.array(ts)
        if is_paced or np.std(arr) < 0.5: return "EXTERNAL: PACEMAKER", 99, "Zero Variance (Quartz Precision)"
        if np.max(np.abs(np.gradient(arr))) > 5.0: return "EXTERNAL: INFUSION", 90, "Non-Physiologic Step Change"
        f, Pxx = welch(arr, fs=1/60)
        entropy = -np.sum((Pxx/np.sum(Pxx)) * np.log2((Pxx/np.sum(Pxx)) + 1e-12))
        if entropy < 1.5: return "EXTERNAL: VENTILATOR", 85, "Periodic Entrainment"
        return "INTERNAL: AUTONOMIC", 80, "Fractal Pink Noise"

    @staticmethod
    def bayes_shock(row):
        means = {"Cardiogenic": [1.8, 2800], "Distributive": [5.0, 800], "Hypovolemic": [2.0, 3000], "Stable": [3.2, 1900]}
        covs = {"Cardiogenic": [[0.5, -100], [-100, 150000]], "Distributive": [[1.0, -200], [-200, 100000]],
                "Hypovolemic": [[0.4, -50], [-50, 200000]], "Stable": [[0.6, -150], [-150, 150000]]}
        scores = {}; total = 0
        x = [row['CI'], row['SVRI']]
        for k, m in means.items():
            try: scores[k] = multivariate_normal.pdf(x, m, covs[k]); total += scores[k]
            except: scores[k] = 0
        return {k: (v/total)*100 for k, v in scores.items()} if total > 1e-9 else {k:25.0 for k in means}

    @staticmethod
    def rl_advisor(row, drugs):
        if row['MAP'] < 65:
            if row['CI'] < 2.2: return "Titrate Dobutamine", 88
            else: return "Increase Norepi", 90
        return "Maintain", 99

    @staticmethod
    def detect_anomalies(df):
        df['anomaly'] = IsolationForest(contamination=0.05).fit_predict(df[['MAP','CI','SVRI']].fillna(0))
        return df

    @staticmethod
    def monte_carlo_forecast(df, target='MAP', n_sims=50):
        curr, hist = df[target].iloc[-1], df[target].iloc[-30:]
        vol = max(np.std(hist) if len(hist)>1 else 1.0, 0.5)
        paths = np.array([curr + np.cumsum(np.random.normal(0, vol, 30)) for _ in range(n_sims)])
        return np.percentile(paths, 10, 0), np.percentile(paths, 50, 0), np.percentile(paths, 90, 0)

    @staticmethod
    def spc_multivariate(df):
        X = df[['MAP', 'CI', 'SVRI']].to_numpy()
        try:
            lw = LedoitWolf().fit(X[:60]) 
            diff = X - lw.location_
            t2 = np.sum(np.dot(diff, lw.precision_) * diff, axis=1)
            pca = PCA(2).fit(StandardScaler().fit_transform(X))
            X_recon = pca.inverse_transform(pca.transform(StandardScaler().fit_transform(X)))
            spe = np.sum((StandardScaler().fit_transform(X) - X_recon)**2, axis=1)
            return t2, spe
        except: return np.zeros(len(X)), np.zeros(len(X))
    
    @staticmethod
    def inverse_centroids(df):
        try:
            if len(df)<10: return ["Data Insufficient"]
            sc = StandardScaler()
            X = sc.fit_transform(df[['CI','SVRI','Lactate']].fillna(0))
            ctrs = sc.inverse_transform(KMeans(3, random_state=42, n_init=10).fit(X).cluster_centers_)
            return [f"C{i+1}: CI={c[0]:.1f}, SVR={c[1]:.0f}" for i,c in enumerate(ctrs)]
        except: return ["Calc Error"]

class QualityAssurance:
    @staticmethod
    def calculate_cpk(data, usl, lsl):
        mean = np.mean(data); std = np.std(data)
        if std == 0: return 0
        return min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
    @staticmethod
    def get_subgroups(data, size=5):
        n = len(data); n_trim = n - (n % size)
        if n_trim == 0: return data.reshape(1, -1)
        return data[:n_trim].reshape(-1, size)
    @staticmethod
    def simulate_noisy_sensor(true_data, bias=5, noise_std=8):
        return true_data + bias + np.random.normal(0, noise_std, len(true_data))
    @staticmethod
    def calc_ewma(data, lam=0.2):
        ewma = np.zeros_like(data); ewma[0] = data[0]
        for i in range(1, len(data)): ewma[i] = lam * data[i] + (1 - lam) * ewma[i-1]
        return ewma
    @staticmethod
    def calc_cusum(data, k=0.5):
        z = (data - np.mean(data)) / (np.std(data) if np.std(data)>0 else 1)
        cp, cm = np.zeros_like(data), np.zeros_like(data)
        for i in range(1, len(data)):
            cp[i] = max(0, z[i] - k + cp[i-1]); cm[i] = max(0, -k - z[i] + cm[i-1])
        return cp, cm
    @staticmethod
    def check_westgard(data):
        mean, std = np.mean(data), np.std(data)
        if std == 0: return []
        z = (data - mean) / std
        violations = []
        if len(z)>0 and abs(z[-1]) > 3: violations.append("1-3s (Random)")
        if len(z)>1 and abs(z[-1]) > 2 and abs(z[-2]) > 2: violations.append("2-2s (Systematic)")
        return violations
    @staticmethod
    def mewma(df, lam=0.3):
        X = df[['MAP', 'CI']].values
        mu = np.mean(X, axis=0); sigma = np.cov(X.T)
        try: inv_sigma = np.linalg.pinv(sigma)
        except: inv_sigma = np.eye(2)
        z = np.zeros_like(X); t2 = []
        for i in range(len(X)):
            z[i] = lam * (X[i] - mu) + (1 - lam) * (z[i-1] if i > 0 else 0)
            t2.append(z[i].T @ inv_sigma @ z[i])
        return np.array(t2)

class DeepAnalytics:
    @staticmethod
    def kalman_filter(data):
        x = np.mean(data); P = 1.0; Q = 1e-5; R = 0.1**2; estimates = []
        for m in data:
            x = x + (P+Q)/(P+Q+R)*(m-x); P = (1-(P+Q)/(P+Q+R))*(P+Q)
            estimates.append(x)
        return np.array(estimates)
    @staticmethod
    def granger(df):
        try:
            res = grangercausalitytests(df[['MAP', 'Lactate']].diff().dropna(), maxlag=[3], verbose=False)
            return True, res[3][0]['ssr_ftest'][1]
        except: return False, 1.0
    @staticmethod
    def autoencoder(df):
        X = StandardScaler().fit_transform(df[['MAP', 'CI', 'SVRI', 'HR']].fillna(0))
        model = MLPRegressor(hidden_layer_sizes=(8, 4, 8), random_state=42, max_iter=200).fit(X, X)
        return np.mean((X - model.predict(X))**2, axis=1)

class ForecastingEngine:
    @staticmethod
    def fit_predict(data, steps=30):
        try: hw = ExponentialSmoothing(data, trend='add').fit().forecast(steps)
        except: hw = np.zeros(steps)
        return hw

# ==========================================
# 5. PATIENT SIMULATOR (ORCHESTRATOR)
# ==========================================
class PatientSimulator:
    def __init__(self, mins=360):
        self.mins = mins
        self.t = np.arange(mins)
        
    def run(self, case_id, drugs, fluids, bsa, peep, is_paced, vent_mode):
        cases = {
            "65M Post-CABG": {'ci':(2.4, 1.8), 'map':(75, 55), 'svri':(2000, 1600), 'hr':(85, 95), 'shunt':0.10, 'copd':1.0, 'vo2_stress': 1.0},
            "24F Septic Shock": {'ci':(3.5, 5.5), 'map':(65, 45), 'svri':(1200, 500), 'hr':(110, 140), 'shunt':0.15, 'copd':1.0, 'vo2_stress': 1.4},
            "82M HFpEF Sepsis": {'ci':(2.2, 1.8), 'map':(85, 55), 'svri':(2400, 1800), 'hr':(80, 110), 'shunt':0.20, 'copd':1.5, 'vo2_stress': 1.1},
            "50M Trauma": {'ci':(3.0, 1.5), 'map':(70, 40), 'svri':(2500, 3200), 'hr':(100, 150), 'shunt':0.05, 'copd':1.0, 'vo2_stress': 0.9}
        }
        p = cases[case_id]
        seed = len(case_id)+42
        Utils.set_seed(seed)
        
        hr, map_r, ci_r, svri_r, rr = Physiology.Autonomic.generate(self.mins, p, is_paced, vent_mode)
        ppv = (20 if "Trauma" in case_id else 12) + (np.sin(self.t/8)*4)
        ci_fluid = (fluids/500) * (0.4 if np.mean(ppv)>13 else 0.05)
        
        map_f, ci_f, hr_f, svri_f = Physiology.PKPD.apply(map_r, ci_r+ci_fluid, hr, svri_r, drugs, self.mins)
        pao2, paco2, spo2, vd_vt = Physiology.Respiratory.exchange(drugs['fio2'], rr, p['shunt'], peep, self.mins, p['copd'])
        hb = 8.0 if "Trauma" in case_id else 12.0
        do2i, vo2i, o2er, lactate = Physiology.Metabolic.calculate(ci_f, hb, spo2, self.mins, p['vo2_stress'])
        
        df = pd.DataFrame({
            "Time": self.t, "HR": hr_f, "MAP": map_f, "CI": ci_f, "SVRI": svri_f,
            "CO": ci_f * bsa, "SVR": svri_f / bsa,
            "Lactate": lactate, "SpO2": spo2, "PaO2": pao2, "PaCO2": paco2, "RR": rr,
            "DO2I": do2i, "VO2I": vo2i, "O2ER": o2er, "Vd/Vt": vd_vt,
            "CPO": (map_f * (ci_f * bsa)) / 451
        }).fillna(0)
        return df

class PatientSimulatorL9(PatientSimulator):
    def run_deep(self, case_id, drugs, fluids, bsa, peep, is_paced, vent_mode):
        df = self.run(case_id, drugs, fluids, bsa, peep, is_paced, vent_mode)
        baro_hr, baro_svr, symp = DeepPhysiology.Baroreflex.compute(df['MAP'].values)
        df['HR'] = (df['HR'] * 0.7) + (baro_hr * 0.3)
        df['SVRI'] = (df['SVRI'] * 0.8) + (baro_svr * 0.2)
        df['Symp_Tone'] = symp
        gfr, uo = [], []
        for m, c in zip(df['MAP'], df['CI']):
            g, u = DeepPhysiology.Renal.compute(m, c)
            gfr.append(g); uo.append(u)
        df['GFR'] = gfr; df['UrineOutput'] = uo
        c1_ne, _ = AdvancedPKPD.two_compartment(np.full(self.mins, drugs['norepi']), self.mins)
        df['Norepi_C1'] = c1_ne
        df['MEWMA'] = QualityAssurance.mewma(df)
        df['Recon_Error'] = DeepAnalytics.autoencoder(df)
        df['MAP_Kalman'] = DeepAnalytics.kalman_filter(df['MAP'].values)
        return df

# ==========================================
# 6. VISUALIZATION LAYER (L9)
# ==========================================
class Viz:
    @staticmethod
    def spark(data, color, key):
        fig = px.line(x=np.arange(len(data)), y=data)
        fig.update_traces(line_color=color, line_width=2)
        fig.update_layout(showlegend=False, xaxis_visible=False, yaxis_visible=False, margin=dict(l=0,r=0,t=0,b=0), height=35)
        return fig

    @staticmethod
    def bayes(probs, key):
        fig = go.Figure(go.Bar(x=list(probs.values()), y=list(probs.keys()), orientation='h', marker=dict(color=[CONFIG.COLORS['hemo'], CONFIG.COLORS['crit'], CONFIG.COLORS['warn'], CONFIG.COLORS['ok']])))
        fig.update_layout(margin=dict(l=0,r=0,t=25,b=0), height=120, xaxis=dict(title="Probability [%]", range=[0, 100]), title="Bayesian State Estimation")
        return fig

    @staticmethod
    def attractor_3d(df, key):
        r = df.iloc[-60:]
        fig = go.Figure(go.Scatter3d(x=r['CPO'], y=r['SVRI'], z=r['Lactate'], mode='lines+markers', marker=dict(size=3, color=r.index, colorscale='Viridis'), line=dict(width=2)))
        fig.update_layout(scene=dict(xaxis_title='Power [W]', yaxis_title='SVRI [dyn·s]', zaxis_title='Lac [mM]'), margin=dict(l=0,r=0,b=0,t=30), height=250, title="3D Trajectory")
        return fig

    @staticmethod
    def chaos(df, source, key):
        hr = np.maximum(df['HR'].iloc[-120:], 1.0); rr = 60000 / hr
        c = CONFIG.COLORS['ext'] if "EXTERNAL" in source else 'teal'
        fig = go.Figure(go.Scatter(x=rr.iloc[:-1], y=rr.iloc[1:], mode='markers', marker=dict(color=c, size=4, opacity=0.6)))
        fig.update_layout(title=f"Chaos: {source}", height=200, margin=dict(l=20,r=20,t=30,b=20), xaxis_title="RR(n) [ms]", yaxis_title="RR(n+1) [ms]")
        return fig

    @staticmethod
    def spectral(df, key):
        data = df['HR'].iloc[-120:].to_numpy()
        f, Pxx = welch(data, fs=1/60)
        fig = px.line(x=f, y=Pxx)
        fig.add_vline(x=0.04, line_dash="dot", annotation_text="LF"); fig.add_vline(x=0.15, line_dash="dot", annotation_text="HF")
        fig.update_layout(title="Spectral HRV", height=200, margin=dict(l=20,r=20,t=30,b=20), xaxis_title="Hz", yaxis_title="Power")
        return fig

    @staticmethod
    def hemodynamic_profile(df, key):
        r = df.iloc[-60:]
        fig = go.Figure()
        fig.add_hline(y=2000, line_dash="dot", annotation_text="Vaso"); fig.add_vline(x=2.2, line_dash="dot", annotation_text="Low Flow")
        fig.add_trace(go.Scatter(x=r['CI'], y=r['SVRI'], mode='markers', marker=dict(color=r.index, colorscale='Viridis'), name="State"))
        fig.update_layout(title="Pump vs Pipes", height=250, margin=dict(l=20,r=20,t=30,b=20), xaxis_title="CI [L/min/m²]", yaxis_title="SVRI [dyn·s]")
        return fig

    @staticmethod
    def phase_space(df, key):
        r = df.iloc[-60:]
        fig = go.Figure()
        fig.add_shape(type="rect", x0=0, x1=0.6, y0=2, y1=15, fillcolor="rgba(255,0,0,0.1)", line_width=0)
        fig.add_trace(go.Scatter(x=r['CPO'], y=r['Lactate'], mode='lines+markers', marker=dict(color=r.index, colorscale='Bluered'), name="Traj"))
        fig.update_layout(title="Coupling", height=250, margin=dict(l=20,r=20,t=30,b=20), xaxis_title="Power [W]", yaxis_title="Lactate [mM]")
        return fig

    @staticmethod
    def vq_scatter(df, key):
        fig = px.scatter(df.iloc[-60:], x="PaO2", y="SpO2", color="PaCO2", color_continuous_scale="Bluered")
        fig.update_layout(title="V/Q Status", height=250, margin=dict(l=20,r=20,t=30,b=20), xaxis_title="PaO2", yaxis_title="SpO2")
        return fig

    @staticmethod
    def spc_charts(df, key):
        data = df['MAP'].to_numpy()
        xbar = np.mean(QualityAssurance.get_subgroups(data), axis=1)
        fig = make_subplots(rows=1, cols=3, subplot_titles=("X-Bar", "R-Chart", "I-MR"))
        fig.add_trace(go.Scatter(y=xbar, mode='lines+markers'), row=1, col=1)
        fig.add_hline(y=np.mean(xbar)+3*np.std(xbar), line_color='red', row=1, col=1)
        fig.add_trace(go.Scatter(y=np.ptp(QualityAssurance.get_subgroups(data), axis=1), mode='lines+markers'), row=1, col=2)
        fig.add_trace(go.Scatter(y=np.abs(np.diff(data)), mode='lines'), row=1, col=3)
        fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=20), title="Statistical Process Control")
        return fig

    @staticmethod
    def method_comp(df, key):
        true = df['MAP'].iloc[-120:].to_numpy(); noise = QualityAssurance.simulate_noisy_sensor(true)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Bland-Altman", "Regression"))
        diff = true - noise
        fig.add_trace(go.Scatter(x=(true+noise)/2, y=diff, mode='markers'), row=1, col=1)
        fig.add_hline(y=np.mean(diff)+1.96*np.std(diff), line_dash='dot', line_color='red', row=1, col=1)
        fig.add_trace(go.Scatter(x=true, y=noise, mode='markers'), row=1, col=2)
        fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=20), title="Sensor Validation")
        return fig

    @staticmethod
    def cpk_tol(df, key):
        d = df['MAP'].iloc[-120:]; cpk = QualityAssurance.calculate_cpk(d, CONFIG.MAP_USL, CONFIG.MAP_LSL)
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Cpk={cpk:.2f}", "Tolerance"))
        fig.add_trace(go.Scatter(x=np.linspace(40,130,100), y=norm.pdf(np.linspace(40,130,100), d.mean(), d.std()), fill='tozeroy'), row=1, col=1)
        fig.add_vline(x=65, line_color='red', row=1, col=1); fig.add_vline(x=110, line_color='red', row=1, col=1)
        fig.add_trace(go.Scatter(y=d, mode='lines'), row=1, col=2)
        fig.add_hrect(y0=65, y1=110, fillcolor='green', opacity=0.1, row=1, col=2)
        fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=20), title="Process Capability")
        return fig

    @staticmethod
    def forecast(df, p10, p50, p90, key):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(30), y=df['MAP'].iloc[-30:], name="History", line=dict(color='black')))
        fx = np.arange(30, 60)
        fig.add_trace(go.Scatter(x=np.concatenate([fx, fx[::-1]]), y=np.concatenate([p90, p10[::-1]]), fill='toself', fillcolor='rgba(0,0,255,0.2)', line=dict(width=0), name="CI"))
        fig.add_trace(go.Scatter(x=fx, y=p50, line=dict(dash='dot', color='blue'), name="Median"))
        fig.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20), title="Monte Carlo Forecast")
        return fig

    @staticmethod
    def counterfactual(df, df_b, key):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['MAP'].iloc[-60:], name="Rx", line=dict(color=CONFIG.COLORS['ok'])))
        fig.add_trace(go.Scatter(y=df_b['MAP'].iloc[-60:], name="No Rx", line=dict(dash='dot', color=CONFIG.COLORS['crit'])))
        fig.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20), title="Counterfactual")
        return fig

    @staticmethod
    def mspc(t2, spe, key):
        fig = make_subplots(rows=1, cols=2, subplot_titles=("T²", "SPE"))
        fig.add_trace(go.Scatter(y=t2), row=1, col=1); fig.add_hline(y=chi2.ppf(0.99, 3), line_color='red', row=1, col=1)
        fig.add_trace(go.Scatter(y=spe), row=1, col=2)
        fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=20), title="Multivariate SPC")
        return fig

    @staticmethod
    def adv_control(df, key):
        d = df['MAP'].to_numpy()
        ewma = QualityAssurance.calc_ewma(d); cp, cm = QualityAssurance.calc_cusum(d)
        violations = QualityAssurance.check_westgard(d)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("EWMA", "CUSUM"))
        fig.add_trace(go.Scatter(y=d, line=dict(color='gray'), name="Raw"), row=1, col=1)
        fig.add_trace(go.Scatter(y=ewma, line=dict(color='blue'), name="EWMA"), row=1, col=1)
        # CUSUM simplified visualization
        fig.add_trace(go.Scatter(y=np.cumsum(d - np.mean(d)), name="CUSUM"), row=1, col=2)
        fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=20), title=f"Adv Control")
        return fig

    @staticmethod
    def adv_forecast(df, key):
        try: hw = ForecastingEngine.fit_predict(df['MAP'].iloc[-60:].to_numpy())
        except: hw = np.zeros(30)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(60), y=df['MAP'].iloc[-60:], name="Hx", line=dict(color='black')))
        fig.add_trace(go.Scatter(x=np.arange(60,90), y=hw, name="ETS", line=dict(dash='dot', color='green')))
        fig.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20), title="Advanced Forecasting (ETS)")
        return fig

    @staticmethod
    def wasserstein(df, key):
        e = df['MAP'].iloc[:60]; l = df['MAP'].iloc[-60:]
        d = wasserstein_distance(e, l)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=e, opacity=0.5, name="Baseline"))
        fig.add_trace(go.Histogram(x=l, opacity=0.5, name="Current"))
        fig.update_layout(height=200, margin=dict(l=10,r=10,t=30,b=20), title=f"Dist Shift (W={d:.1f})", barmode='overlay')
        return fig
    
    # L9 New Visuals
    @staticmethod
    def recurrence_plot(data, key):
        d = data[-60:]; D = np.abs(d[:,None]-d[None,:]); eps=0.1*np.std(d)
        fig = px.imshow((D<eps).astype(int), color_continuous_scale='gray', title="Recurrence Plot")
        fig.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20), xaxis_title="Time i", yaxis_title="Time j")
        return fig
    
    @staticmethod
    def phase_velocity(df, key):
        m = df['MAP'].iloc[-60:]; dm = np.gradient(m); c = df['CI'].iloc[-60:]
        fig = go.Figure(data=[go.Scatter3d(x=m, y=dm, z=c, mode='lines+markers', marker=dict(size=3, color=np.arange(60)), line=dict(color='gray'))])
        fig.update_layout(scene=dict(xaxis_title="MAP", yaxis_title="dMAP/dt", zaxis_title="CI"), title="Velocity Phase Space", height=250, margin=dict(l=0,r=0,b=0,t=30))
        return fig

# ==========================================
# 8. MAIN EXECUTION & LAYOUT
# ==========================================

if 'events' not in st.session_state: st.session_state['events'] = []
if 'fluids' not in st.session_state: st.session_state['fluids'] = 0

def main():
    # Sidebar Controls
    with st.sidebar:
        st.title("TITAN | L9 IRIDIUM")
        res_mins = st.select_slider("Resolution", [60, 180, 360, 720], value=360)
        case_id = st.selectbox("Profile", ["65M Post-CABG", "24F Septic Shock", "82M HFpEF Sepsis", "50M Trauma"])
        
        c1, c2 = st.columns(2)
        with c1: h = st.number_input("Ht", 150, 200, 175); norepi = st.number_input("Norepi", 0.0, 2.0, 0.0)
        with c2: w = st.number_input("Wt", 50, 150, 80); vaso = st.number_input("Vaso", 0.0, 0.1, 0.0)
        bsa = np.sqrt((h*w)/3600)
        
        dobu = st.number_input("Dobutamine", 0.0, 10.0, 0.0); bb = st.number_input("Esmolol", 0.0, 1.0, 0.0)
        fio2 = st.slider("FiO2", 0.21, 1.0, 0.4); peep = st.slider("PEEP", 0, 20, 5)
        is_paced = st.checkbox("Pacemaker"); vent_mode = st.selectbox("Vent", ["Spontaneous", "Control (AC)"])
        
        if st.button("Bolus 500mL"): st.session_state['fluids'] += 500
        live = st.checkbox("LIVE")

    # Run L9 Simulator
    sim = PatientSimulatorL9(res_mins)
    drugs = {'norepi':norepi, 'vaso':vaso, 'dobu':dobu, 'bb':bb, 'fio2':fio2}
    df = sim.run_deep(case_id, drugs, st.session_state['fluids'], bsa, peep, is_paced, vent_mode)
    
    # Counterfactual Sim
    sim_b = PatientSimulator(60)
    base_d = {'norepi':0, 'vaso':0, 'dobu':0, 'bb':0, 'fio2':0.21}
    df_b = sim_b.run(case_id, base_d, 0, bsa, peep, False, 'Spontaneous')

    # Calc Advanced Metrics
    granger_sig, granger_p = DeepAnalytics.granger(df)
    probs = Analytics.bayes_shock(df.iloc[-1])
    sugg, conf = Analytics.rl_advisor(df.iloc[-1], drugs)
    p10, p50, p90 = Analytics.monte_carlo_forecast(df)
    t2, spe = Analytics.spc_multivariate(df)
    src, _, reason = Analytics.signal_forensics(df['HR'].iloc[-120:], is_paced)
    centroids = Analytics.inverse_centroids(df)
    curr = df.iloc[-1]

    # Render Function
    def render_layout(df, i):
        st.markdown(f"""
        <div class="status-banner" style="border-left-color: {CONFIG.COLORS['ai']};">
            <div><div style="font-size:0.8rem; font-weight:800; color:{CONFIG.COLORS['ai']}">BAYESIAN STATE</div>
            <div style="font-size:1.5rem; font-weight:800;">{max(probs, key=probs.get).upper()}</div></div>
            <div style="text-align:right">
                <div style="font-size:0.8rem;">GFR: {curr['GFR']:.0f} | UO: {curr['UrineOutput']:.1f}</div>
                <div style="font-size:1.2rem; font-weight:800;">{centroids[0]}</div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        tabs = st.tabs(["🫀 Clinical", "🫁 Resp", "🤖 AI/Forensics", "📊 SPC", "🔬 Deep Physics"])
        
        with tabs[0]: 
            c1, c2 = st.columns(2)
            c1.plotly_chart(Viz.hemodynamic_profile(df, i), use_container_width=True)
            c1.markdown("<div class='clinical-hint'><b>Forrester Plot:</b> Classifies shock state based on Flow (CI) vs Resistance (SVR).</div>", unsafe_allow_html=True)
            c2.plotly_chart(Viz.phase_space(df, i), use_container_width=True)
            c2.markdown("<div class='clinical-hint'><b>Coupling:</b> Detects if the heart (Power) can meet metabolic demand (Lactate).</div>", unsafe_allow_html=True)
            
        with tabs[4]: 
            st.markdown('<div class="zone-header">ZONE Z: L9 ADVANCED PHYSICS</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("Baroreflex Tone", f"{curr['Symp_Tone']:.2f}")
            c2.metric("Granger (MAP->Lac)", f"p={granger_p:.4f}")
            
            d1, d2 = st.columns(2)
            fig_mewma = px.line(df, x='Time', y='MEWMA', title="MEWMA Drift Detection")
            d1.plotly_chart(fig_mewma, use_container_width=True)
            d1.markdown("<div class='clinical-hint'><b>MEWMA:</b> Multivariate EWMA detects subtle, simultaneous shifts in correlated vitals.</div>", unsafe_allow_html=True)
            
            fig_ae = px.area(df, x='Time', y='Recon_Error', title="Autoencoder Anomaly Score")
            d2.plotly_chart(fig_ae, use_container_width=True)
            d2.markdown("<div class='clinical-hint'><b>Autoencoder:</b> Measures how 'strange' the current physiology is compared to the model's learned baseline.</div>", unsafe_allow_html=True)
            
            e1, e2 = st.columns(2)
            e1.plotly_chart(Viz.recurrence_plot(df['MAP'].values, i), use_container_width=True)
            e1.markdown("<div class='clinical-hint'><b>Recurrence:</b> Visualizes non-linear stability. Periodic=Checkered, Chaotic=No pattern.</div>", unsafe_allow_html=True)
            e2.plotly_chart(Viz.phase_velocity(df, i), use_container_width=True)
            e2.markdown("<div class='clinical-hint'><b>Velocity Phase:</b> Plots Value vs Speed of Change. Spirals indicate loss of autoregulation.</div>", unsafe_allow_html=True)
            
            f1, f2 = st.columns(2)
            fig_pk = go.Figure()
            fig_pk.add_trace(go.Scatter(x=df['Time'], y=df['Norepi_C1'], name="Central (C1)"))
            fig_pk.update_layout(title="2-Compartment PK (Norepi)", height=250)
            f1.plotly_chart(fig_pk, use_container_width=True)
            f2.metric("Granger (MAP->Lac)", f"p={granger_p:.4f}")
            f2.markdown("<div class='clinical-hint'><b>Causality:</b> Does MAP drive Lactate changes? Significant if p < 0.05.</div>", unsafe_allow_html=True)

        with tabs[2]:
            st.plotly_chart(Viz.chaos(df, src, i), use_container_width=True)
            st.markdown("<div class='clinical-hint'><b>Poincaré Plot:</b> Visualizes HRV complexity. Cigar=Healthy, Dot=Stressed/Paced.</div>", unsafe_allow_html=True)
            st.plotly_chart(Viz.spectral(df, i), use_container_width=True)
            st.markdown("<div class='clinical-hint'><b>Spectral:</b> Frequency domain analysis of autonomic tone. LF=Sympathetic, HF=Vagal.</div>", unsafe_allow_html=True)
            st.plotly_chart(Viz.wasserstein(df, i), use_container_width=True)
            st.markdown("<div class='clinical-hint'><b>Wasserstein:</b> Quantifies the total 'distance' the patient state has drifted from admission.</div>", unsafe_allow_html=True)

        with tabs[3]:
             st.plotly_chart(Viz.spc_charts(df, i), use_container_width=True)
             st.markdown("<div class='clinical-hint'><b>SPC:</b> Distinguishes signal from noise. Points outside red lines are significant.</div>", unsafe_allow_html=True)
             st.plotly_chart(Viz.cpk_tol(df, i), use_container_width=True)
             st.markdown("<div class='clinical-hint'><b>Cpk:</b> Process Capability. Is the patient capable of staying within safe MAP limits?</div>", unsafe_allow_html=True)
             st.plotly_chart(Viz.mspc(t2, spe, i), use_container_width=True)
             st.markdown("<div class='clinical-hint'><b>Multivariate:</b> T2 detects system shifts. SPE detects model errors.</div>", unsafe_allow_html=True)
             st.plotly_chart(Viz.adv_control(df, i), use_container_width=True)
             st.markdown("<div class='clinical-hint'><b>Control:</b> CUSUM/EWMA track small sustained shifts.</div>", unsafe_allow_html=True)
             st.plotly_chart(Viz.method_comp(df, i), use_container_width=True)
             st.markdown("<div class='clinical-hint'><b>Validation:</b> Sensor drift analysis.</div>", unsafe_allow_html=True)

        with tabs[1]:
            st.plotly_chart(Viz.vq_scatter(df, i), use_container_width=True)
            st.markdown("<div class='clinical-hint'><b>V/Q Scatter:</b> Maps Oxygenation (Shunt) vs Ventilation (Dead Space).</div>", unsafe_allow_html=True)
            
    # Render
    if live:
        holder = st.empty()
        for i in range(max(10, res_mins-60), res_mins):
            with holder.container(): render_layout(df.iloc[:i], i)
            time.sleep(0.1)
    else:
        render_layout(df, len(df))

if __name__ == "__main__":
    main()
