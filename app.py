import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time
from disaster_tweet_predictor import DisasterTweetPredictor

# =========================================================================
# PAGE CONFIGURATION
# =========================================================================
st.set_page_config(
    page_title="Crisis Sentinel AI • Govind Mishra",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================
# PREMIUM CUSTOM CSS & TYPOGRAPHY
# =========================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1350px;
    }

    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #312E81 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 18px;
        padding: 26px 30px;
        margin-bottom: 22px;
        box-shadow: 0 20px 35px -10px rgba(15, 23, 42, 0.45);
        color: white;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(239, 68, 68, 0.2) 0%, rgba(99, 102, 241, 0.05) 70%, transparent 100%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.25rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 60%, #93C5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #CBD5E1;
        max-width: 850px;
        line-height: 1.5;
        margin-bottom: 12px;
    }
    .hero-tagline {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
    }

    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-author {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: #111827;
        font-weight: 800;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
    }
    .badge-live {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid #10B981;
        color: #34D399;
    }
    .badge-disaster {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
    }
    .badge-safe {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
    }

    /* Alert Result Cards */
    .alert-card-disaster {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-left: 8px solid #DC2626;
        border-radius: 16px;
        padding: 24px;
        margin-top: 18px;
        margin-bottom: 18px;
        box-shadow: 0 12px 24px -6px rgba(239, 68, 68, 0.25);
        animation: pulseBorder 2s infinite alternate;
    }
    @keyframes pulseBorder {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        100% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0.0); }
    }
    .alert-card-safe {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 2px solid #10B981;
        border-left: 8px solid #059669;
        border-radius: 16px;
        padding: 24px;
        margin-top: 18px;
        margin-bottom: 18px;
        box-shadow: 0 12px 24px -6px rgba(16, 185, 129, 0.2);
    }

    /* Developer Card in Sidebar */
    .dev-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 16px;
        color: white;
        margin-bottom: 16px;
        box-shadow: 0 10px 20px -5px rgba(0,0,0,0.3);
    }
    .dev-name {
        font-family: 'Outfit', sans-serif;
        font-size: 1.15rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 2px;
    }
    .dev-role {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-bottom: 8px;
    }

    /* Metric Cards */
    .metric-card-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.06);
    }

    /* Footer */
    .footer-section {
        margin-top: 45px;
        padding: 22px;
        border-top: 1px solid #E2E8F0;
        text-align: center;
        color: #64748B;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# MODEL ENGINE LOADER WITH CACHING (SILENT OPTIMAL ENGINE)
# =========================================================================
@st.cache_resource(show_spinner="Initializing Intelligence Engine...")
def load_predictor():
    return DisasterTweetPredictor(model_dir='processed_data', default_model='distilbert')

try:
    predictor = load_predictor()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ Error loading intelligence engine: {e}")

# =========================================================================
# SIDEBAR CONFIGURATION & DEVELOPER PROFILE
# =========================================================================
with st.sidebar:
    # Developer Branding
    st.markdown("""
    <div class='dev-card'>
        <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 10px;'>
            <div style='background: linear-gradient(135deg, #6366F1, #8B5CF6); width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.35rem; color: white; font-weight: bold;'>
                GM
            </div>
            <div>
                <div class='dev-name'>GOVIND MISHRA</div>
                <div class='dev-role'>Lead Machine Learning Engineer</div>
            </div>
        </div>
        <div style='display: flex; gap: 6px; flex-wrap: wrap;'>
            <span class='badge badge-author'>Engineered by Govind Mishra</span>
            <span class='badge badge-live'>Active Engine</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎚️ Detection Sensitivity Control")
    threshold = st.slider(
        "Alert Sensitivity Cutoff:",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.01,
        help="Controls the threshold for triggering crisis alerts. Default 0.40 is the optimal balanced cutoff."
    )
    if threshold == 0.40:
        st.success("⭐ **Calibrated Balanced Mode** (Recommended)")
    elif threshold < 0.40:
        st.warning("⚠️ **High Sensitivity Mode** (Higher Alert Volume)")
    else:
        st.info("🛡️ **High Precision Mode** (Anti-Fatigue / Strict Verification)")

    st.divider()
    st.markdown("### 📡 System Status")
    st.markdown("""
    - **Engine State**: Operational & Ready 🟢
    - **Host**: `localhost:8505`
    - **Mode**: Real-time Crisis Identification
    - **Disambiguation**: Active Lexical & Semantic Parser
    """)

    st.divider()
    st.caption("Crisis Sentinel AI Platform • Developed by **Govind Mishra**")

# =========================================================================
# HERO HEADER BANNER
# =========================================================================
st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>🚨 Crisis Sentinel AI: Emergency Tweet Intelligence</div>
    <div class='hero-subtitle'>
        Real-time Natural Language Processing system engineered to identify emergency crisis events and disambiguate literal disasters from informal slang and conversational chatter.
    </div>
    <div class='hero-tagline'>
        <span class='badge badge-author'>👨‍💻 Developed by GOVIND MISHRA</span>
        <span class='badge badge-live'>Real-Time Stream Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================================
# APPLICATION TABS (CLEAN PRODUCTION WORKFLOW)
# =========================================================================
tab1, tab2, tab3 = st.tabs([
    "🎯 Real-Time Incident Classifier",
    "📁 Batch CSV Processing Engine",
    "📖 System Overview & Operator Guide"
])

# =========================================================================
# TAB 1: REAL-TIME INCIDENT CLASSIFIER
# =========================================================================
with tab1:
    st.subheader("Analyze Live Message or Tweet")
    st.markdown("Input any message below to determine if it indicates a genuine emergency or informal conversational chatter:")

    # Query parameter handling for direct demonstration & screenshot capture
    query_params = st.query_params
    if "demo" in query_params:
        demo_mode = query_params["demo"]
        if demo_mode == "wildfire" and "tweet_input" not in st.session_state:
            st.session_state["tweet_input"] = "13,000 people receive #wildfires evacuation orders in California as blaze spreads rapidly!"
            st.session_state["kw_input"] = "wildfires"
        elif demo_mode == "metaphor" and "tweet_input" not in st.session_state:
            st.session_state["tweet_input"] = "My math test was a total disaster today, I got body bagged by question 4 loool"
            st.session_state["kw_input"] = "disaster"

    # Preset test cases
    c_p1, c_p2, c_p3, c_p4, c_p5, c_p6 = st.columns(6)

    if c_p1.button("🔥 Wildfire Alert", width='stretch', help="Real Emergency"):
        st.session_state["tweet_input"] = "13,000 people receive #wildfires evacuation orders in California as blaze spreads rapidly!"
        st.session_state["kw_input"] = "wildfires"
    if c_p2.button("🌊 Flash Flooding", width='stretch', help="Real Emergency"):
        st.session_state["tweet_input"] = "Heavy rain causes flash flooding of streets in Manitou, Colorado Springs areas! Cars submerged!"
        st.session_state["kw_input"] = "flooding"
    if c_p3.button("🏚️ Structural Collapse", width='stretch', help="Real Emergency"):
        st.session_state["tweet_input"] = "Overpass bridge collapsed near downtown highway, emergency sirens sounding and first responders on scene!"
        st.session_state["kw_input"] = "collapse"
    if c_p4.button("🎭 Slang: Exam Disaster", width='stretch', help="Metaphor / Informal Non-Disaster"):
        st.session_state["tweet_input"] = "My math test was a total disaster today, I got body bagged by question 4 loool"
        st.session_state["kw_input"] = "disaster"
    if c_p5.button("🎵 Slang: Music is Fire", width='stretch', help="Metaphor / Informal Non-Disaster"):
        st.session_state["tweet_input"] = "The new Drake album track is absolute fire! I am screaming this beat is blowing up my speakers haha"
        st.session_state["kw_input"] = "fire"
    if c_p6.button("💬 Weekend Banter", width='stretch', help="Casual Non-Disaster"):
        st.session_state["tweet_input"] = "Having iced cold coffee and watching Netflix with my best friends, great relaxing weekend!"
        st.session_state["kw_input"] = "none"

    tweet_text = st.text_area(
        "Enter Message / Tweet Text:",
        value=st.session_state.get("tweet_input", ""),
        height=110,
        placeholder="Type or paste text here... (e.g. 'Emergency evacuation order issued following flash floods on Highway 101')",
        key="main_tweet_input"
    )

    col_kw, col_btn = st.columns([2, 1])
    with col_kw:
        keyword_input = st.text_input(
            "Optional Keyword / Hashtag:",
            value=st.session_state.get("kw_input", ""),
            placeholder="e.g. wildfires, flood, earthquake (leave blank if none)",
            key="main_kw_input"
        )
    with col_btn:
        st.write("")
        st.write("")
        classify_btn = st.button("🚀 Analyze Emergency Status", type="primary", width='stretch', key="run_predict_btn")

    if (classify_btn or tweet_text.strip()) and model_loaded:
        if not tweet_text.strip():
            st.warning("Please enter text to analyze.")
        else:
            with st.spinner("Executing linguistic & threat assessment..."):
                start_time = time.time()
                kw = keyword_input.strip() if keyword_input.strip() and keyword_input.strip().lower() != "none" else None
                res = predictor.predict(
                    tweet_text.strip(),
                    keywords=[kw] if kw else None,
                    threshold=threshold
                )[0]
                elapsed_ms = (time.time() - start_time) * 1000

            is_disaster = res['is_disaster']
            prob = res['disaster_probability']
            prob_percent = prob * 100

            # Result Announcement Card
            if is_disaster:
                st.markdown(f"""
                <div class='alert-card-disaster'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                        <span class='badge badge-disaster'>🚨 CRITICAL EMERGENCY DETECTED</span>
                        <span style='font-size: 0.85rem; font-weight: 700; color: #DC2626;'>ASSESSMENT BY GOVIND MISHRA AI</span>
                    </div>
                    <h2 style='color: #DC2626; margin: 4px 0 10px 0;'>REAL CRISIS / HAZARD EVENT IDENTIFIED</h2>
                    <p style='font-size: 1.15rem; color: #1E293B; line-height: 1.4; margin-bottom: 10px;'>
                        <b>Message Content:</b> &ldquo;{res['tweet']}&rdquo;
                    </p>
                    <p style='color: #475569; margin: 0;'>
                        <b>Operational Assessment:</b> High-urgency indicators detected. This message indicates a genuine ongoing civil hazard, natural disaster, or incident requiring emergency attention.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='alert-card-safe'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                        <span class='badge badge-safe'>🟢 ALL CLEAR / NON-DISASTER</span>
                        <span style='font-size: 0.85rem; font-weight: 700; color: #059669;'>ASSESSMENT BY GOVIND MISHRA AI</span>
                    </div>
                    <h2 style='color: #059669; margin: 4px 0 10px 0;'>NO EMERGENCY HAZARD IDENTIFIED</h2>
                    <p style='font-size: 1.15rem; color: #1E293B; line-height: 1.4; margin-bottom: 10px;'>
                        <b>Message Content:</b> &ldquo;{res['tweet']}&rdquo;
                    </p>
                    <p style='color: #475569; margin: 0;'>
                        <b>Operational Assessment:</b> Evaluated as safe social commentary, colloquial slang, or metaphorical expression. No emergency dispatch needed.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Key Incident Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Classification Status", "🚨 REAL CRISIS" if is_disaster else "🟢 NON-DISASTER")
            m2.metric("Crisis Confidence Level", f"{prob_percent:.1f}%", delta=f"{prob - threshold:+.2f} vs cutoff")
            m3.metric("Active Sensitivity Cutoff", f"{threshold:.2f}")
            m4.metric("Processing Time", f"{elapsed_ms:.1f} ms")

            # Risk Confidence Meter
            st.markdown(f"**Emergency Risk Probability Rating ({prob_percent:.1f}%):**")
            st.progress(float(min(max(prob, 0.0), 1.0)))

            # Deep NLP Diagnostic Expander
            with st.expander("🔍 Deep Linguistic Diagnostics & Feature Signals", expanded=True):
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("#### 📝 Message Structural Attributes")
                    st.markdown(f"- **Character Length**: `{res['char_length']}` characters")
                    st.markdown(f"- **Word Count**: `{res['word_count']}` words")
                    st.markdown(f"- **Hyperlinks Detected**: `{res['url_count']}` links")
                    st.markdown(f"- **Hashtags Detected**: `{res['hashtag_count']}` tags")
                    st.markdown(f"- **User Mentions**: `{res['mention_count']}` mentions")

                with d2:
                    st.markdown("#### ⚡ Lexical & Sentiment Indicators")
                    st.markdown(f"- **Sentiment Polarity**: Compound `{res['sentiment_compound']}` | Negative `{res['sentiment_neg']}`")
                    urg_text = ", ".join([f"`{w}`" for w in res['urgency_words']]) if res['urgency_words'] else "*None detected*"
                    st.markdown(f"- **Urgency & Hazard Keywords**: {urg_text}")
                    slang_text = ", ".join([f"`{w}`" for w in res['slang_words']]) if res['slang_words'] else "*None detected*"
                    st.markdown(f"- **Informal / Slang Counter-Signals**: {slang_text}")

                    if not is_disaster and res['slang_words']:
                        st.info("💡 **Metaphor Disambiguation**: Informal slang tokens successfully neutralized dramatic words, preventing false alarm escalation.")

# =========================================================================
# TAB 2: BATCH CSV PROCESSING ENGINE
# =========================================================================
with tab2:
    st.subheader("Batch CSV Ingestion & Scalable Stream Processing")
    st.markdown("Upload any CSV dataset containing text to run bulk classifications across records simultaneously:")

    # Built-in sample dataset
    sample_df = pd.DataFrame({
        'record_id': [101, 102, 103, 104, 105, 106],
        'text': [
            "Evacuation underway in northern communities following severe wildfire alert",
            "OMG this math assignment is a complete disaster I am going to cry lol",
            "Magnitude 6.2 earthquake felt off the coast of Japan, tsunami advisory in effect",
            "Having iced coffee and watching Netflix, great weekend so far",
            "Overpass collapsed due to flash flood, emergency responders on scene!",
            "That basketball game was absolute fire! The final dunk blew up the arena"
        ]
    })
    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)

    col_btn_down, col_btn_demo = st.columns([1, 1])
    with col_btn_down:
        st.download_button(
            "📥 Download Sample CSV Template",
            data=csv_buffer.getvalue(),
            file_name="sample_messages_template.csv",
            mime="text/csv",
            key="dl_sample_template"
        )
    with col_btn_demo:
        use_demo = st.button("🧪 Load Built-in Demo Test Batch (6 Records)", key="load_demo_batch")

    uploaded_file = st.file_uploader("Or Upload Your CSV File (must contain a text column):", type=["csv"], key="csv_file_uploader")

    df_to_process = None
    if use_demo:
        df_to_process = sample_df.copy()
        st.info("Loaded demo test batch.")
    elif uploaded_file is not None:
        try:
            df_to_process = pd.read_csv(uploaded_file)
            st.success(f"Uploaded CSV loaded successfully! Found {len(df_to_process):,} records.")
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")

    if df_to_process is not None:
        candidate_cols = [c for c in df_to_process.columns if c.lower() in ['text', 'tweet', 'clean_text', 'message', 'tweet_text']]
        default_col_idx = 0
        if candidate_cols:
            default_col_idx = list(df_to_process.columns).index(candidate_cols[0])

        text_column = st.selectbox("Select Text Column to Process:", options=df_to_process.columns, index=default_col_idx, key="batch_text_col")

        if st.button("🚀 Process Batch Records", type="primary", key="btn_run_batch"):
            with st.spinner(f"Classifying {len(df_to_process):,} records..."):
                t_start = time.time()
                tweets_list = df_to_process[text_column].fillna('').astype(str).tolist()
                batch_res = predictor.predict(tweets_list, threshold=threshold)
                t_elapsed = time.time() - t_start

                df_to_process['is_disaster'] = [r['is_disaster'] for r in batch_res]
                df_to_process['disaster_probability'] = [r['disaster_probability'] for r in batch_res]
                df_to_process['status_label'] = ['EMERGENCY ALERT [DISASTER]' if r['is_disaster'] else 'CLEAR [NON-DISASTER]' for r in batch_res]
                df_to_process['sentiment_score'] = [r['sentiment_compound'] for r in batch_res]

            tps = len(df_to_process) / max(t_elapsed, 0.001)
            st.success(f"✅ Completed processing {len(df_to_process):,} records in {t_elapsed:.2f} seconds ({tps:.1f} records/sec)!")

            # Batch Summary Metrics
            disaster_count = int((df_to_process['is_disaster'] == True).sum())
            safe_count = len(df_to_process) - disaster_count
            disaster_pct = (disaster_count / len(df_to_process)) * 100

            bm1, bm2, bm3, bm4 = st.columns(4)
            bm1.metric("Total Records Processed", f"{len(df_to_process):,}")
            bm2.metric("Crises Identified", f"{disaster_count:,} ({disaster_pct:.1f}%)")
            bm3.metric("Non-Emergency Records", f"{safe_count:,} ({100 - disaster_pct:.1f}%)")
            bm4.metric("Throughput Rate", f"{tps:.1f} /sec")

            # Visual Summaries
            ch1, ch2 = st.columns([1, 1])
            with ch1:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.pie(
                    [disaster_count, safe_count],
                    labels=['Disasters', 'Non-Disaster'],
                    autopct='%1.1f%%',
                    colors=['#EF4444', '#10B981'],
                    startangle=140,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
                )
                ax.axis('equal')
                st.pyplot(fig)
            with ch2:
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                sns.histplot(df_to_process['disaster_probability'], bins=10, kde=True, color='#6366F1', ax=ax2)
                ax2.set_title("Probability Distribution")
                ax2.set_xlabel("Disaster Probability")
                st.pyplot(fig2)

            # Data Table Preview & Filter
            st.markdown("#### 📋 Processed Batch Results")
            filter_choice = st.radio("Filter Results:", ["All Records", "Crises Only (🚨)", "Non-Emergency Only (🟢)"], horizontal=True)
            if filter_choice == "Crises Only (🚨)":
                filtered_df = df_to_process[df_to_process['is_disaster'] == True]
            elif filter_choice == "Non-Emergency Only (🟢)":
                filtered_df = df_to_process[df_to_process['is_disaster'] == False]
            else:
                filtered_df = df_to_process

            st.dataframe(filtered_df, width='stretch')

            # Download Annotated Results
            res_buffer = io.StringIO()
            df_to_process.to_csv(res_buffer, index=False)
            st.download_button(
                "📥 Download Annotated CSV Results",
                data=res_buffer.getvalue(),
                file_name="emergency_predictions_govind_mishra.csv",
                mime="text/csv",
                key="dl_annotated_results"
            )

# =========================================================================
# TAB 3: SYSTEM OVERVIEW & OPERATOR GUIDE
# =========================================================================
with tab3:
    st.subheader("📖 System Overview & Operator Guide")

    st.markdown("""
    ### 👨‍💻 Engineering Leadership & Attribution
    * **Lead Machine Learning Engineer**: **GOVIND MISHRA**
    * **System Name**: Crisis Sentinel AI Platform
    * **Domain**: Social Media Emergency Intelligence & NLP Disambiguation
    * **Deployment Host**: `http://localhost:8505`

    ---
    ### 🎯 Operational Purpose
    During civil emergencies, earthquakes, wildfires, and floods, public social feeds contain vital real-time intelligence. However, automatic monitors often drown in false positives triggered by informal slang (*"I'm dying"*, *"total disaster"*, *"blew up"*).
    
    This platform automates:
    1. **Real-Time Signal Filtering**: Separating genuine crisis reports from benign colloquialisms.
    2. **Urgency Assessment**: Flagging high-probability incidents for emergency response coordination.
    3. **Scalable Stream Ingestion**: Evaluating streaming batches of messages with sub-second response times.

    ---
    ### ⚙️ Operator Sensitivity Guidelines
    * **Balanced Mode ($0.40$)**: The calibrated default setting, offering the optimal balance between catching emergencies and suppressing false alarms.
    * **High Sensitivity Mode ($0.20 - 0.35$)**: Recommended for life-safety monitoring during active severe storm or wildfire watches, where missing a report cannot be tolerated.
    * **High Precision Mode ($0.50 - 0.70$)**: Recommended when routing alerts to public dispatch channels to prevent alarm fatigue.
    """)

# =========================================================================
# FOOTER SECTION
# =========================================================================
st.markdown("""
<div class='footer-section'>
    <b>Crisis Sentinel AI Platform</b> • Real-Time Emergency Intelligence System<br>
    Engineered & Developed by <b>GOVIND MISHRA</b> • Running on Localhost: 8505
</div>
""", unsafe_allow_html=True)
