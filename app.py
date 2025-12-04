import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging
import io
import base64
from typing import Dict, List, Any, Optional

# ==========================================
# 0. METADATA & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="TITAN-Med | Clinical Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_VERSION = "2.0.0"
SCORING_MODEL_VER = "CM-24-Q4"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TITAN_Core")

class ClinicalConfig:
    """
    Centralized configuration for clinical domains, scoring thresholds, 
    and risk logic.
    """
    DOMAINS = {
        "physical": {
            "label": "Physical Symptom Burden",
            "weight": 1.0,
            "description": "Assessment of somatic sensation and physical discomfort.",
            "items": ["pain", "fatigue", "nausea", "breathlessness", "insomnia"],
            "max_item_score": 10
        },
        "mental": {
            "label": "Psychological Distress",
            "weight": 1.5,  # Higher weight for risk stratification
            "description": "Assessment of mood, anxiety, and existential distress.",
            "items": ["anxiety", "depression", "isolation", "stress", "fear_of_future"],
            "max_item_score": 10
        },
        "functional": {
            "label": "Functional Impairment",
            "weight": 1.0,
            "description": "Impact on activities of daily living (ADLs).",
            "items": ["mobility", "self_care", "work_life", "social_interaction"],
            "max_item_score": 10
        }
    }

    # Severity Thresholds (Normalized 0-100)
    THRESHOLDS = {
        "normal": (0, 25),
        "mild": (26, 50),
        "moderate": (51, 75),
        "severe": (76, 100)
    }

    # Critical Thresholds for specific items (Red Flags)
    RED_FLAGS = {
        "pain": 8,
        "breathlessness": 8,
        "depression": 9,
        "isolation": 9
    }

# ==========================================
# 1. DATA MANAGER
# ==========================================
class DataManager:
    """Handles session state initialization, input validation, and data integrity."""

    @staticmethod
    def init_session():
        """Initialize all session state variables if they don't exist."""
        defaults = {
            "step": 1,
            "demographics": {},
            "responses": {},
            "scores": {},
            "flags": [],
            "clinician_mode": False,
            "privacy_accepted": False,
            "session_id": datetime.now().strftime("%Y%m%d-%H%M%S")
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    @staticmethod
    def validate_input(key: str, value: float, min_val=0, max_val=10) -> bool:
        """Ensure inputs are within valid clinical ranges."""
        if not (min_val <= value <= max_val):
            logger.warning(f"Input validation failed for {key}: {value}")
            return False
        return True

    @staticmethod
    def impute_missing(data: Dict[str, float]) -> Dict[str, float]:
        """
        Imputes missing data points using the mean of provided values 
        to prevent scoring failures.
        """
        clean_data = data.copy()
        valid_values = [v for v in data.values() if v is not None]
        
        if not valid_values:
            return {k: 0.0 for k in data}  # Safe fallback
        
        avg_val = sum(valid_values) / len(valid_values)
        
        for k, v in clean_data.items():
            if v is None:
                clean_data[k] = avg_val
                logger.info(f"Imputed missing value for item '{k}' with average: {avg_val:.2f}")
                
        return clean_data

# ==========================================
# 2. SCORING ENGINE
# ==========================================
class ScoringEngine:
    """
    Logic for transforming raw inputs into normalized clinical scores.
    """
    
    @staticmethod
    def normalize(raw: float, max_possible: float) -> float:
        """Normalizes a raw score to a 0-100 scale."""
        if max_possible == 0: return 0.0
        return (raw / max_possible) * 100.0

    @staticmethod
    def classify_severity(score: float) -> str:
        """Maps a normalized score to a clinical severity label."""
        for label, (low, high) in ClinicalConfig.THRESHOLDS.items():
            if low <= score <= high:
                return label.title()
        return "Unknown"

    @staticmethod
    def calculate_scores(responses: Dict[str, float]) -> Dict[str, Any]:
        """
        Main scoring pipeline:
        1. Impute missing data
        2. Calculate domain scores
        3. Apply weights
        4. Detect cross-domain patterns
        """
        responses = DataManager.impute_missing(responses)
        results = {}
        
        global_weighted_sum = 0.0
        total_weight = 0.0
        
        # Process each domain
        for domain_key, config in ClinicalConfig.DOMAINS.items():
            items = config["items"]
            # Get values, defaulting to 0 if somehow missing after imputation
            item_values = [responses.get(i, 0) for i in items]
            raw_score = sum(item_values)
            max_score = len(items) * config["max_item_score"]
            
            norm_score = ScoringEngine.normalize(raw_score, max_score)
            weighted_part = norm_score * config["weight"]
            
            global_weighted_sum += weighted_part
            total_weight += config["weight"]
            
            results[domain_key] = {
                "raw": raw_score,
                "normalized": norm_score,
                "severity": ScoringEngine.classify_severity(norm_score),
                "weight": config["weight"],
                "contribution": weighted_part
            }

        # Global Risk Index calculation
        if total_weight > 0:
            global_risk = global_weighted_sum / total_weight
        else:
            global_risk = 0.0

        # Pattern Detection (Rule-based)
        patterns = []
        # Example: High Distress + Low Function = "High-Risk Disconnect"
        if results['mental']['normalized'] > 70 and results['functional']['normalized'] < 30:
            patterns.append("Psychological distress disproportionate to functional loss.")
        # Example: High Physical + Low Mental = "Resilient Coping"
        if results['physical']['normalized'] > 70 and results['mental']['normalized'] < 30:
            patterns.append("High physical burden with strong psychological resilience.")

        results['global'] = {
            "risk_index": global_risk,
            "severity": ScoringEngine.classify_severity(global_risk),
            "patterns": patterns
        }
        
        return results

    @staticmethod
    def detect_flags(responses: Dict[str, float]) -> List[str]:
        """Identify specific red flags based on single-item thresholds."""
        flags = []
        for item, threshold in ClinicalConfig.RED_FLAGS.items():
            val = responses.get(item, 0)
            if val >= threshold:
                flags.append(f"CRITICAL: {item.upper()} score ({val}/10) exceeds safety threshold.")
        return flags

# ==========================================
# 3. VISUALIZATION ENGINE
# ==========================================
class Visualizer:
    """Generates Plotly charts for the dashboard."""

    @staticmethod
    def draw_gauge(score: float):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Global Risk Index"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "midnightblue"},
                'steps': [
                    {'range': [0, 25], 'color': "#d4f7dc"},   # Green
                    {'range': [25, 50], 'color': "#fff3cd"},  # Yellow
                    {'range': [50, 75], 'color': "#ffe5d0"},  # Orange
                    {'range': [75, 100], 'color': "#f8d7da"}  # Red
                ]
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    @staticmethod
    def draw_radar(scores: Dict[str, Any]):
        # Filter out 'global' key
        domains = [k for k in scores.keys() if k != 'global']
        labels = [ClinicalConfig.DOMAINS[k]['label'] for k in domains]
        values = [scores[k]['normalized'] for k in domains]
        
        # Close the loop
        labels.append(labels[0])
        values.append(values[0])

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values, theta=labels,
            fill='toself', name='Patient Profile',
            line_color='#008080'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            margin=dict(l=40, r=40, t=20, b=20),
            height=350
        )
        return fig

    @staticmethod
    def draw_waterfall(scores: Dict[str, Any]):
        domains = [k for k in scores.keys() if k != 'global']
        labels = [ClinicalConfig.DOMAINS[k]['label'] for k in domains]
        # We display the weighted contribution relative to the total sum
        contributions = [scores[k]['contribution'] for k in domains]
        
        fig = go.Figure(go.Waterfall(
            name="Risk Factors", orientation="v",
            measure=["relative"] * len(domains),
            x=labels,
            y=contributions,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig.update_layout(
            title="Weighted Risk Contribution",
            showlegend=False, 
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    @staticmethod
    def draw_heatmap(responses: Dict[str, float]):
        items = list(responses.keys())
        values = list(responses.values())
        
        # Calculate grid dimensions for a roughly square heatmap
        n = len(items)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        
        # Pad with NaNs to fill grid
        padded_values = values + [np.nan] * (rows * cols - n)
        padded_items = items + [""] * (rows * cols - n)
        
        z_data = np.array(padded_values).reshape(rows, cols)
        text_data = np.array(padded_items).reshape(rows, cols)
        
        # Clean up text for display
        text_display = [[t.replace("_", " ").title() for t in row] for row in text_data]

        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            text=text_display,
            texttemplate="%{text}<br>(%{z})",
            colorscale="RdBu_r",
            zmin=0, zmax=10,
            showscale=True
        ))
        fig.update_layout(
            title="Symptom Severity Heatmap",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, autorange="reversed"),
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

# ==========================================
# 4. CLINICAL FEEDBACK ENGINE
# ==========================================
class ClinicalFeedback:
    """Generates text-based insights and recommendations."""
    
    @staticmethod
    def get_recommendation(domain: str, score: float) -> str:
        if score < 25:
            return "Routine monitoring. Encourage healthy maintenance behaviors."
        elif score < 50:
            return "Mild elevation. Consider lifestyle modifications and symptom journaling."
        elif score < 75:
            return "Moderate severity. Pharmacologic intervention or therapy referral indicated."
        else:
            return "Severe impairment. **Urgent** specialist review and comprehensive care plan required."

    @staticmethod
    def generate_report(demographics, scores, flags, responses):
        """Generates a structured text report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Header
        report = f"CLINICAL ASSESSMENT REPORT\n"
        report += f"Date: {timestamp}\n"
        report += f"Patient ID: {demographics.get('id', 'Unknown')}\n"
        report += f"Age: {demographics.get('age', 'Unknown')}\n"
        report += "-" * 40 + "\n\n"
        
        # Summary
        report += f"GLOBAL RISK INDEX: {scores['global']['risk_index']:.1f} ({scores['global']['severity']})\n"
        
        if flags:
            report += "\n*** RED FLAGS DETECTED ***\n"
            for f in flags:
                report += f"- {f}\n"
        
        report += "\n--- DOMAIN BREAKDOWN ---\n"
        for key, val in scores.items():
            if key == 'global': continue
            label = ClinicalConfig.DOMAINS[key]['label']
            report += f"\n{label.upper()}: {val['normalized']:.1f}% ({val['severity']})\n"
            report += f"Recommendation: {ClinicalFeedback.get_recommendation(key, val['normalized'])}\n"

        if scores['global']['patterns']:
            report += "\n--- DETECTED PATTERNS ---\n"
            for p in scores['global']['patterns']:
                report += f"- {p}\n"

        return report

# ==========================================
# 5. MAIN APPLICATION LOGIC
# ==========================================
def main():
    # Initialize
    DataManager.init_session()

    # --- Sidebar UI ---
    with st.sidebar:
        st.title("🏥 TITAN-Med")
        st.caption(f"System v{APP_VERSION} | Model {SCORING_MODEL_VER}")
        
        # Navigation Progress
        steps = ["Intake", "Physical", "Mental", "Functional", "Analysis"]
        curr_step = st.session_state['step']
        st.progress(min(curr_step / len(steps), 1.0))
        st.markdown(f"**Current Phase:** {steps[min(curr_step-1, len(steps)-1)]}")
        
        st.markdown("---")
        
        # Live Sticky Summary
        if curr_step > 1:
            st.markdown("#### 📊 Live Summary")
            st.metric("Data Points", len(st.session_state['responses']))
            if 'scores' in st.session_state and 'global' in st.session_state['scores']:
                 risk = st.session_state['scores']['global']['risk_index']
                 delta = None # Could track previous assessment if available
                 st.metric("Est. Risk", f"{risk:.0f}", delta, delta_color="inverse")
        
        st.markdown("---")
        st.session_state['clinician_mode'] = st.toggle("Clinician Expert Mode", value=False)
        
        with st.expander("ℹ️ About"):
            st.info("This tool implements a multi-domain weighted scoring algorithm (TITAN-CM) for holistic health assessment.")

    # --- STEP 1: INTAKE ---
    if st.session_state['step'] == 1:
        st.header("Patient Intake & Privacy")
        
        st.info("🔒 **Privacy Notice:** All data processing occurs locally within this session. No data is transmitted to external servers.")
        
        with st.form("intake_form"):
            c1, c2 = st.columns(2)
            pid = c1.text_input("Patient ID / MRN")
            age = c2.number_input("Patient Age", 18, 110, 45)
            gender = st.selectbox("Gender Identity", ["Male", "Female", "Non-binary", "Prefer not to say"])
            
            consent = st.checkbox("I confirm patient consent for digital assessment.")
            
            submitted = st.form_submit_button("Begin Assessment")
            if submitted:
                if not consent:
                    st.error("Consent is required to proceed.")
                elif not pid:
                    st.error("Patient ID is required.")
                else:
                    st.session_state['demographics'] = {"id": pid, "age": age, "gender": gender}
                    st.session_state['step'] = 2
                    st.rerun()

    # --- STEP 2, 3, 4: DYNAMIC DOMAIN INPUT ---
    elif st.session_state['step'] in [2, 3, 4]:
        # Map steps 2,3,4 to domain keys physical, mental, functional
        domain_keys = list(ClinicalConfig.DOMAINS.keys())
        # step 2 -> index 0, step 3 -> index 1, etc.
        d_index = st.session_state['step'] - 2
        d_key = domain_keys[d_index]
        d_config = ClinicalConfig.DOMAINS[d_key]

        st.subheader(f"Assessment: {d_config['label']}")
        st.markdown(f"*{d_config['description']}*")
        
        with st.form(f"domain_form_{d_key}"):
            st.markdown("Please rate the severity of the following over the **past 7 days** (0 = None, 10 = Extreme):")
            
            # Create a grid for sliders
            cols = st.columns(2)
            for i, item in enumerate(d_config['items']):
                col = cols[i % 2]
                current_val = st.session_state['responses'].get(item, 0)
                val = col.slider(
                    item.replace("_", " ").title(), 
                    min_value=0, max_value=10, value=int(current_val),
                    key=f"slider_{item}"
                )
                st.session_state['responses'][item] = val
            
            st.markdown("---")
            
            # Navigation buttons in form
            c_prev, c_next = st.columns([1, 5])
            # Note: Streamlit forms allow one submit button usually, but we can handle logic via session state 
            # outside or use a single submit that advances. Simpler to have just "Next".
            submitted = st.form_submit_button("Save & Continue")
            
            if submitted:
                st.session_state['step'] += 1
                st.rerun()

    # --- STEP 5: ANALYSIS & REPORT ---
    elif st.session_state['step'] == 5:
        st.header("Clinical Assessment Results")
        
        # Perform Scoring
        with st.spinner("Processing Clinical Logic..."):
            scores = ScoringEngine.calculate_scores(st.session_state['responses'])
            flags = ScoringEngine.detect_flags(st.session_state['responses'])
            st.session_state['scores'] = scores # Cache result
        
        # 1. Alerts Section
        if flags:
            st.error(f"⚠️ {len(flags)} CRITICAL ALERTS DETECTED")
            for f in flags:
                st.warning(f, icon="🚩")
        else:
            st.success("No critical red flags detected.", icon="✅")

        # 2. Dashboard Grid
        r1_c1, r1_c2 = st.columns([1, 2])
        
        with r1_c1:
            st.markdown("### Global Risk")
            st.plotly_chart(
                Visualizer.draw_gauge(scores['global']['risk_index']), 
                use_container_width=True
            )
            st.caption(f"Severity Class: **{scores['global']['severity']}**")

        with r1_c2:
            st.markdown("### Domain Profile")
            st.plotly_chart(
                Visualizer.draw_radar(scores), 
                use_container_width=True
            )

        # 3. Detailed Clinical Reasoning
        st.markdown("---")
        st.subheader("Clinical Reasoning & Recommendations")
        
        report_text = ClinicalFeedback.generate_report(
            st.session_state['demographics'], scores, flags, st.session_state['responses']
        )

        # Display logic per domain
        for d_key in ClinicalConfig.DOMAINS.keys():
            d_data = scores[d_key]
            with st.expander(f"{ClinicalConfig.DOMAINS[d_key]['label']} - {d_data['severity'].upper()}", expanded=True):
                c_exp1, c_exp2 = st.columns([3, 1])
                with c_exp1:
                    st.markdown(f"**Score:** {d_data['normalized']:.1f}%")
                    st.markdown(f"**Impression:** {ClinicalFeedback.get_recommendation(d_key, d_data['normalized'])}")
                with c_exp2:
                    st.progress(d_data['normalized'] / 100)

        # 4. Expert Mode (Conditional)
        if st.session_state['clinician_mode']:
            st.markdown("---")
            st.subheader("🔬 Expert Analysis Tools")
            t1, t2, t3 = st.tabs(["Risk Attribution", "Symptom Matrix", "Raw Data"])
            
            with t1:
                st.plotly_chart(Visualizer.draw_waterfall(scores), use_container_width=True)
                st.caption("Visualizes which domains contribute most to the global risk index.")
            
            with t2:
                st.plotly_chart(Visualizer.draw_heatmap(st.session_state['responses']), use_container_width=True)
                st.caption("Heatmap of raw symptom intensity.")
                
            with t3:
                st.json(scores)

        # 5. Export Utilities
        st.markdown("---")
        st.subheader("📂 Actions")
        
        c_down1, c_down2, c_reset = st.columns(3)
        
        # CSV Export
        flat_data = {"PatientID": st.session_state['demographics']['id']}
        flat_data.update(st.session_state['responses'])
        for k in scores:
            if k != 'global': flat_data[f"Score_{k}"] = scores[k]['normalized']
        df_export = pd.DataFrame([flat_data])
        csv = df_export.to_csv(index=False).encode('utf-8')
        
        c_down1.download_button(
            "📥 Download Data (CSV)",
            data=csv,
            file_name=f"patient_{st.session_state['demographics']['id']}_data.csv",
            mime="text/csv"
        )
        
        c_down2.download_button(
            "📄 Download Report (TXT)",
            data=report_text,
            file_name=f"patient_{st.session_state['demographics']['id']}_report.txt",
            mime="text/plain"
        )
        
        if c_reset.button("🔄 Start New Patient"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ==========================================
# 8. SELF-TESTING & EXECUTION
# ==========================================
def _run_unit_tests():
    """Internal consistency checks for scoring logic."""
    try:
        # Test 1: Normalization
        assert ScoringEngine.normalize(5, 10) == 50.0
        # Test 2: Severity
        assert ScoringEngine.classify_severity(80) == "Severe"
        # Test 3: Config Integrity
        assert "physical" in ClinicalConfig.DOMAINS
        # logger.info("Self-tests passed.")
    except AssertionError as e:
        st.error(f"Critical System Error: Self-test failed. {e}")
        st.stop()

if __name__ == "__main__":
    _run_unit_tests()
    main()
