"""Shared Streamlit styling for Handwritten Character Recognition."""

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #4f46e5;
    --primary-light: #6366f1;
    --primary-dark: #3730a3;
    --secondary: #0ea5e9;
    --accent: #8b5cf6;
    --success: #10b981;
    --warning: #f59e0b;
    --surface: #ffffff;
    --surface-alt: #f8fafc;
    --surface-border: #e2e8f0;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #64748b;
}

html, body, [class*="css"], .stMarkdown, .stText {
    font-family: 'Plus Jakarta Sans', 'Outfit', 'Noto Sans Devanagari', -apple-system, sans-serif !important;
}

/* Hide Streamlit default chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stSidebarNav"] {display: none;}
[data-testid="stSidebar"] {display: none;}

/* Main App Container padding */
.main .block-container {
    max-width: 1280px;
    padding-top: 1.5rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

/* Premium Hero Header */
.hero-wrapper {
    position: relative;
    background: linear-gradient(135deg, #090d16 0%, #111827 45%, #1e1b4b 100%);
    border-radius: 24px;
    padding: 2.5rem 2.8rem;
    color: #ffffff;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px -15px rgba(15, 23, 42, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.08);
    overflow: hidden;
}

.hero-wrapper::before {
    content: "अ क ज्ञ";
    position: absolute;
    right: 2rem;
    top: -1rem;
    font-size: 8rem;
    font-family: 'Noto Sans Devanagari', sans-serif;
    font-weight: 800;
    color: rgba(255, 255, 255, 0.03);
    pointer-events: none;
    letter-spacing: 0.5rem;
}

.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(99, 102, 241, 0.16);
    border: 1px solid rgba(165, 180, 252, 0.3);
    padding: 0.35rem 0.9rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #c7d2fe;
    margin-bottom: 0.8rem;
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.025em;
    line-height: 1.15;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #93c5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0.7rem;
    margin-bottom: 1.2rem;
    max-width: 820px;
    line-height: 1.6;
}

.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 1rem;
}

.hero-pill-item {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.35rem 0.8rem;
    border-radius: 10px;
}

/* Styled Streamlit Tabs (Floating Navbar Style) */
[data-testid="stTabs"] {
    margin-top: 0.5rem;
    margin-bottom: 2rem;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: #f1f5f9;
    padding: 0.4rem;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    height: 44px;
    border-radius: 12px !important;
    padding: 0 1.25rem !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    color: #475569 !important;
    background-color: transparent !important;
    border: none !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #0f172a !important;
    background: rgba(255, 255, 255, 0.6) !important;
}

[data-testid="stTabs"] [aria-selected="true"] {
    background: #ffffff !important;
    color: #4f46e5 !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.12), 0 1px 3px rgba(0, 0, 0, 0.06) !important;
}

[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none !important;
}

/* Glassmorphism Cards */
.card-glass {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 1.8rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.06), 0 4px 10px -4px rgba(0, 0, 0, 0.02);
    margin-bottom: 1.5rem;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.card-glass:hover {
    box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.09), 0 6px 14px -4px rgba(0, 0, 0, 0.03);
}

/* Primary AI Recognition Card */
.result-card {
    background: linear-gradient(180deg, #ffffff 0%, #fcfdfe 100%);
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    padding: 2rem;
    box-shadow: 0 14px 35px -8px rgba(15, 23, 42, 0.08), 0 4px 12px -2px rgba(15, 23, 42, 0.03);
    margin-bottom: 1.5rem;
}

.result-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 1.1rem;
    margin-bottom: 1.4rem;
}

/* Glyph Showcase Box */
.glyph-showcase {
    background: linear-gradient(135deg, #eff6ff 0%, #e0e7ff 100%);
    border: 2px solid #6366f1;
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 135px;
    height: 135px;
    font-size: 4.8rem;
    font-weight: 800;
    color: #3730a3;
    box-shadow: 0 12px 24px -6px rgba(79, 70, 229, 0.25);
    font-family: 'Noto Sans Devanagari', sans-serif;
    transition: transform 0.2s ease;
}

.glyph-showcase:hover {
    transform: scale(1.03);
}

/* Badges */
.badge-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.35rem 0.9rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 700;
    margin-right: 0.4rem;
}

.badge-blue {
    background: #eef2ff;
    color: #4338ca;
    border: 1px solid #c7d2fe;
}

.badge-green {
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
}

.badge-amber {
    background: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
}

.confidence-pill {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    padding: 0.45rem 1.15rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.92rem;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3);
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}

/* Audio Pronunciation Pill */
.audio-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
    color: #ffffff !important;
    padding: 0.45rem 1rem;
    border-radius: 12px;
    font-size: 0.88rem;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
    text-decoration: none;
}

.audio-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35);
}

/* Info Labels and Descriptions */
.info-section-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1e293b;
    margin-top: 1.1rem;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.info-section-desc {
    font-size: 0.95rem;
    color: #475569;
    line-height: 1.6;
    background: #f8fafc;
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    border: 1px solid #f1f5f9;
}

/* Vocabulary Chips */
.vocab-chip {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
    font-size: 0.95rem;
    color: #1e293b;
    border-left: 4px solid #4f46e5;
    transition: all 0.2s ease;
}

.vocab-chip:hover {
    transform: translateX(4px);
    border-color: #6366f1;
}

/* Upload Dropzone styling */
[data-testid="stFileUploader"] {
    background: #f8fafc;
    border: 2px dashed #cbd5e1;
    border-radius: 20px;
    padding: 1.5rem;
    transition: all 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #6366f1;
    background: #f5f3ff;
}

/* Model Metric Cards */
.model-metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    transition: transform 0.2s ease;
}

.model-metric-card:hover {
    transform: translateY(-2px);
    border-color: #c7d2fe;
}

.model-metric-name {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 0.35rem;
}

.model-metric-glyph {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1e293b;
    font-family: 'Noto Sans Devanagari', sans-serif;
    line-height: 1.2;
}

.model-metric-conf {
    font-size: 0.95rem;
    font-weight: 700;
    color: #4f46e5;
    margin-top: 0.2rem;
}

/* Streamlit Button Styling */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.4rem !important;
    font-size: 0.95rem !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border: 1px solid #e2e8f0 !important;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4) !important;
}

/* Streamlit Download Button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    color: #ffffff !important;
    border-radius: 14px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.25) !important;
    transition: all 0.2s ease !important;
}

.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.35) !important;
}

/* Expander styling */
.streamlit-expanderHeader {
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 14px !important;
    background: #f8fafc !important;
}

/* Metric Widget styling */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.1rem 1.4rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
}

[data-testid="stMetricLabel"] {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
}
</style>
"""
