import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------- Page setup & styling ---------------------------
st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1400px;}
    .hero {
        padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
        background: linear-gradient(120deg, rgba(49, 95, 155, .16), rgba(74, 168, 147, .13));
        border: 1px solid rgba(128,128,128,.22);
    }
    .hero h1 {margin: 0 0 .45rem 0; font-size: 2.1rem;}
    .hero p {margin: 0; font-size: 1.02rem; opacity: .85;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: 14px 16px;
        background: rgba(128,128,128,.035);
    }
    .section-note {opacity: .75; font-size: .92rem;}
    .result-card {
        border-radius: 16px; padding: 1.1rem 1.25rem;
        border: 1px solid rgba(128,128,128,.25); margin-top: .4rem;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def project_file(filename):
    return os.path.join(BASE_DIR, filename)

@st.cache_resource
def load_assets():
    model = joblib.load(project_file("best_model.joblib"))
    scaler = joblib.load(project_file("scaler.joblib"))
    features = joblib.load(project_file("feature_columns.joblib"))
    data = pd.read_csv(project_file("data.csv"), sep=";", encoding="utf-8-sig")
    return model, scaler, list(features), data

try:
    model, scaler, features, data = load_assets()
except Exception as exc:
    st.error(f"Could not load the project files: {exc}")
    st.markdown("Keep `app.py`, `data.csv`, `best_model.joblib`, `scaler.joblib`, `feature_columns.joblib` in the same folder.")
    st.stop()

# UCI data dictionary: numeric code -> human-readable label.
# The saved model was trained on the numeric codes, so the UI shows the readable
# text to the user and silently passes the matching code to the model.
YES_NO = {1: "Yes", 0: "No"}

EDUCATION_LEVELS = {  # ordered roughly from lowest to highest level of education
    35: "Cannot read or write",
    36: "Can read, but did not finish 4th year",
    37: "Primary school (1st cycle, 4th/5th year)",
    38: "Lower secondary (2nd cycle, 6th–8th year)",
    11: "7th year (old system)",
    26: "7th year of schooling",
    30: "8th year of schooling",
    29: "9th year not completed",
    19: "Basic education, 3rd cycle (9th–11th year)",
    14: "10th year of schooling",
    10: "11th year not completed",
    12: "Other — 11th year of schooling",
    9: "12th year not completed",
    27: "General high school (2nd cycle)",
    13: "Complementary high school — 2nd year",
    25: "Complementary high school — not completed",
    20: "Complementary high school course",
    1: "Secondary education (12th year)",
    18: "General commerce course",
    31: "General course of administration and commerce",
    33: "Supplementary accounting and administration",
    22: "Technical-professional course",
    39: "Technological specialization course",
    42: "Professional higher technical course",
    6: "Attended higher education (no degree)",
    41: "Specialized higher studies course",
    2: "Bachelor's degree",
    3: "Higher education — degree",
    40: "Higher education — degree (1st cycle)",
    4: "Master's degree",
    43: "Higher education — master's (2nd cycle)",
    5: "Doctorate",
    44: "Higher education — doctorate (3rd cycle)",
    34: "Unknown",
}

OCCUPATIONS = {
    0: "Student",
    1: "Legislators, directors and executive managers",
    2: "Specialists in intellectual and scientific activities",
    3: "Intermediate-level technicians and professions",
    4: "Administrative staff",
    5: "Personal services, security and sales workers",
    6: "Farmers and skilled agriculture/fishery/forestry workers",
    7: "Skilled industry, construction and craft workers",
    8: "Machine operators and assembly workers",
    9: "Unskilled workers",
    10: "Armed forces professions",
    90: "Other situation",
    99: "Not specified",
    101: "Armed forces officers",
    102: "Armed forces sergeants",
    103: "Other armed forces personnel",
    112: "Directors of administrative and commercial services",
    114: "Directors of hotel, catering, trade and other services",
    121: "Specialists in physical sciences, mathematics and engineering",
    122: "Health professionals",
    123: "Teachers",
    124: "Specialists in finance, accounting and public/commercial relations",
    125: "ICT specialists",
    131: "Intermediate-level science and engineering technicians",
    132: "Intermediate-level health technicians",
    134: "Intermediate-level legal, social, sports and cultural technicians",
    135: "ICT technicians",
    141: "Office workers, secretaries and data-entry operators",
    143: "Accounting, statistical and financial services operators",
    144: "Other administrative support staff",
    151: "Personal service workers",
    152: "Sellers",
    153: "Personal care workers",
    154: "Protection and security personnel",
    161: "Market-oriented farmers and skilled agricultural workers",
    163: "Subsistence farmers, fishermen, hunters and gatherers",
    171: "Skilled construction workers (except electricians)",
    172: "Skilled metallurgy and metalworking workers",
    173: "Skilled workers in printing, precision instruments and crafts",
    174: "Skilled electricity and electronics workers",
    175: "Food processing, woodworking, clothing and craft workers",
    181: "Fixed plant and machine operators",
    182: "Assembly workers",
    183: "Vehicle drivers and mobile equipment operators",
    191: "Cleaning workers",
    192: "Unskilled workers in agriculture, fisheries and forestry",
    193: "Unskilled workers in mining, construction, manufacturing and transport",
    194: "Meal preparation assistants",
    195: "Street vendors and street service providers",
}

CATEGORY_LABELS = {
    "Marital status": {
        1: "Single", 2: "Married", 3: "Widowed", 4: "Divorced",
        5: "Living together (de facto union)", 6: "Legally separated",
    },
    "Daytime/evening attendance\t": {1: "Daytime", 0: "Evening"},
    "Displaced": YES_NO,
    "Educational special needs": YES_NO,
    "Debtor": YES_NO,
    "Tuition fees up to date": YES_NO,
    "Gender": {1: "Male", 0: "Female"},
    "Scholarship holder": YES_NO,
    "International": YES_NO,
    "Application order": {
        0: "1st choice (first preference)", 1: "2nd choice", 2: "3rd choice",
        3: "4th choice", 4: "5th choice", 5: "6th choice", 6: "7th choice",
        7: "8th choice", 8: "9th choice", 9: "10th choice (last preference)",
    },
    "Course": {
        33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
        8014: "Social Service (evening)", 9003: "Agronomy",
        9070: "Communication Design", 9085: "Veterinary Nursing",
        9119: "Informatics Engineering", 9130: "Equinculture",
        9147: "Management", 9238: "Social Service", 9254: "Tourism",
        9500: "Nursing", 9556: "Oral Hygiene",
        9670: "Advertising and Marketing Management",
        9773: "Journalism and Communication", 9853: "Basic Education",
        9991: "Management (evening)",
    },
    "Previous qualification": {
        9: "12th year not completed", 10: "11th year not completed",
        12: "Other — 11th year of schooling", 14: "10th year of schooling",
        15: "10th year not completed", 38: "Basic education, 2nd cycle (6th–8th year)",
        19: "Basic education, 3rd cycle (9th–11th year)",
        1: "Secondary education (12th year)",
        39: "Technological specialization course",
        42: "Professional higher technical course",
        6: "Attended higher education (no degree)",
        2: "Bachelor's degree", 3: "Higher education — degree",
        40: "Higher education — degree (1st cycle)",
        4: "Master's degree", 43: "Higher education — master's (2nd cycle)",
        5: "Doctorate",
    },
    "Nacionality": {
        1: "Portuguese", 2: "German", 6: "Spanish", 11: "Italian",
        13: "Dutch", 14: "English", 17: "Lithuanian", 21: "Angolan",
        22: "Cape Verdean", 24: "Guinean", 25: "Mozambican",
        26: "Santomean", 32: "Turkish", 41: "Brazilian", 62: "Romanian",
        100: "Moldovan", 101: "Mexican", 103: "Ukrainian",
        105: "Russian", 108: "Cuban", 109: "Colombian",
    },
    "Application mode": {
        1: "1st phase — general contingent",
        17: "2nd phase — general contingent",
        18: "3rd phase — general contingent",
        5: "1st phase — special contingent (Azores Island)",
        16: "1st phase — special contingent (Madeira Island)",
        39: "Over 23 years old",
        7: "Holder of another higher course",
        53: "Short-cycle diploma holder",
        44: "Technological specialization diploma holder",
        42: "Transfer from another institution",
        43: "Change of course",
        51: "Change of institution/course",
        15: "International student (bachelor)",
        57: "Change of institution/course (international)",
        2: "Ordinance No. 612/93",
        10: "Ordinance No. 854-B/99",
        26: "Ordinance No. 533-A/99, item b2 (different plan)",
        27: "Ordinance No. 533-A/99, item b3 (other institution)",
    },
    "Mother's qualification": EDUCATION_LEVELS,
    "Father's qualification": EDUCATION_LEVELS,
    "Mother's occupation": OCCUPATIONS,
    "Father's occupation": OCCUPATIONS,
}

BINARY_FEATURES = {
    "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date",
    "Scholarship holder", "International",
}
CATEGORICAL_FEATURES = set(CATEGORY_LABELS)
DISPLAY_NAMES = {
    "Nacionality": "Nationality",
    "Daytime/evening attendance\t": "Attendance schedule",
    "Previous qualification (grade)": "Previous qualification grade",
    "Mother's qualification": "Mother's education",
    "Father's qualification": "Father's education",
    "Mother's occupation": "Mother's occupation",
    "Father's occupation": "Father's occupation",
    "Curricular units 1st sem (credited)": "1st semester · Units credited",
    "Curricular units 1st sem (enrolled)": "1st semester · Units enrolled",
    "Curricular units 1st sem (evaluations)": "1st semester · Evaluations",
    "Curricular units 1st sem (approved)": "1st semester · Units approved",
    "Curricular units 1st sem (grade)": "1st semester · Average grade",
    "Curricular units 1st sem (without evaluations)": "1st semester · Units without evaluation",
    "Curricular units 2nd sem (credited)": "2nd semester · Units credited",
    "Curricular units 2nd sem (enrolled)": "2nd semester · Units enrolled",
    "Curricular units 2nd sem (evaluations)": "2nd semester · Evaluations",
    "Curricular units 2nd sem (approved)": "2nd semester · Units approved",
    "Curricular units 2nd sem (grade)": "2nd semester · Average grade",
    "Curricular units 2nd sem (without evaluations)": "2nd semester · Units without evaluation",
    "Unemployment rate": "Unemployment rate (%)",
    "Inflation rate": "Inflation rate (%)",
    "GDP": "GDP indicator"
}

def nice_name(feature):
    return DISPLAY_NAMES.get(feature, feature.replace("\t", "").strip())

def get_code_options(feature, series):
    """Return [(code, readable label)] in a sensible order. No raw codes are shown to the user."""
    known = CATEGORY_LABELS.get(feature, {})
    observed = {int(v) for v in series.dropna().unique()}
    options = [(code, label) for code, label in known.items()]
    # Safety net: any code present in the data but missing from the dictionary.
    for code in sorted(observed - set(known)):
        options.append((code, "Other / not listed"))
    return options

FRIENDLY_HELP = {
    "Age at enrollment": "Student's age (in years) when they first enrolled.",
    "Previous qualification (grade)": "Grade of the previous qualification, on a 0–200 scale.",
    "Admission grade": "Admission grade, on a 0–200 scale.",
    "Unemployment rate": "National unemployment rate (%) at the time of enrollment.",
    "Inflation rate": "National inflation rate (%) at the time of enrollment.",
    "GDP": "National GDP growth indicator at the time of enrollment.",
    "Application order": "Which preference this course was in the student's application.",
}

# --------------------------- Header ---------------------------
st.markdown("""
<div class="hero">
  <h1>🎓 Student Dropout Predictor</h1>
  <p>Fill in the student's details below and click <b>Predict</b> to find out whether the student is likely to drop out.</p>
</div>
""", unsafe_allow_html=True)

with st.container():
    st.subheader("Student details")
    st.markdown('<p class="section-note">Pick an option from each dropdown and type in the numbers. Every field already has a typical starting value, so just change what applies to the student.</p>', unsafe_allow_html=True)

    groups = {
        "👤 Personal details": [
            "Marital status", "Gender", "Age at enrollment", "Nacionality",
            "Displaced", "Educational special needs", "International"
        ],
        "📝 Admission & previous education": [
            "Application mode", "Application order", "Course",
            "Daytime/evening attendance\t", "Previous qualification",
            "Previous qualification (grade)", "Admission grade"
        ],
        "👪 Family background": [
            "Mother's qualification", "Father's qualification",
            "Mother's occupation", "Father's occupation"
        ],
        "💳 Financial information": [
            "Debtor", "Tuition fees up to date", "Scholarship holder"
        ],
        "📚 First semester performance": [f for f in features if "1st sem" in f],
        "📚 Second semester performance": [f for f in features if "2nd sem" in f],
        "🌍 Economic indicators": ["Unemployment rate", "Inflation rate", "GDP"]
    }

    numeric_data = data[features].copy()
    for f in features:
        numeric_data[f] = pd.to_numeric(numeric_data[f], errors="coerce")

    inputs = {}
    with st.form("prediction_form"):
        for group, group_features in groups.items():
            available = [f for f in group_features if f in features]
            if not available:
                continue
            with st.container(border=True):
                st.markdown(f"#### {group}")
                cols = st.columns(2)
                for i, feature in enumerate(available):
                    s = numeric_data[feature].dropna()
                    if s.empty:
                        lo, hi, default = 0.0, 100.0, 0.0
                    else:
                        lo, hi, default = float(s.min()), float(s.max()), float(s.median())
                    label = nice_name(feature)
                    with cols[i % 2]:
                        if feature in CATEGORICAL_FEATURES:
                            options = get_code_options(feature, s)
                            codes = [c for c, _ in options]
                            labels = dict(options)
                            mode_series = s.mode()
                            default_code = int(mode_series.iloc[0]) if not mode_series.empty else codes[0]
                            if default_code not in codes:
                                default_code = codes[0]
                            help_text = FRIENDLY_HELP.get(feature, "Click and type to search the list.")
                            if feature in BINARY_FEATURES or len(codes) == 2:
                                inputs[feature] = st.radio(
                                    label, options=codes, index=codes.index(default_code),
                                    format_func=lambda c, lb=labels: lb[c],
                                    horizontal=True, key=f"radio_{feature}", help=help_text,
                                )
                            else:
                                inputs[feature] = st.selectbox(
                                    label, options=codes, index=codes.index(default_code),
                                    format_func=lambda c, lb=labels: lb[c],
                                    key=f"select_{feature}", help=help_text,
                                )
                        else:
                            help_text = FRIENDLY_HELP.get(feature) or ("Average grade, usually between 0 and 20." if "grade" in feature.lower() else None)
                            if pd.api.types.is_integer_dtype(data[feature].dtype):
                                lo_i, hi_i = int(lo), int(hi)
                                default_i = max(lo_i, min(hi_i, int(round(default))))
                                inputs[feature] = st.number_input(
                                    label, min_value=lo_i, max_value=hi_i, value=default_i,
                                    step=1, key=f"num_{feature}", help=help_text
                                )
                            else:
                                inputs[feature] = st.number_input(
                                    label, min_value=float(lo), max_value=float(hi),
                                    value=float(round(default, 2)), step=0.1, format="%.2f",
                                    key=f"num_{feature}", help=help_text
                                )
        submitted = st.form_submit_button("🔮 Predict", type="primary")

    if submitted:
        row = pd.DataFrame([[inputs[f] for f in features]], columns=features)
        try:
            scaled = scaler.transform(row)
            prediction = model.predict(scaled)[0]
            st.divider()
            st.subheader("Result")
            if prediction == "Dropout":
                st.error("## ⚠️ Dropout")
            else:
                st.success("## ✅ Not Dropout")
            with st.expander("Review the details you entered"):
                readable = []
                for f in features:
                    value = inputs[f]
                    if f in CATEGORICAL_FEATURES:
                        value = CATEGORY_LABELS.get(f, {}).get(int(value), "Other / not listed")
                    readable.append({"Feature": nice_name(f), "Selected value": str(value)})
                st.dataframe(pd.DataFrame(readable), hide_index=True)
        except Exception as exc:
            st.error(f"Prediction could not be generated: {exc}")
