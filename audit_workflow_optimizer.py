import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

# --- CONFIGURATION ---
ROOT = Path(__file__).resolve().parent
ENGAGEMENTS_FILE = ROOT / "engagements_sample.csv"
STAFF_FILE = ROOT / "staff_sample.csv"
REQUIRED_SKILLS = ["Audit"]
LEVEL_SCORES = {"Junior": 1, "Associate": 2, "Senior": 3, "Manager": 4}
HOURS_PER_PERSON_PER_WEEK = 30
BASELINE_STAFF = 4

# --- UTILITY FUNCTIONS ---
@st.cache_data
def load_data():
    try:
        eng = pd.read_csv(ENGAGEMENTS_FILE)
        staff = pd.read_csv(STAFF_FILE)
        return eng, staff
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def encode_features(eng):
    size_map = {"Small": 0, "Medium": 1, "Large": 2}
    comp_map = {"Low": 0, "Medium": 1, "High": 2}
    ind_map = {v: i for i, v in enumerate(sorted(eng.client_industry.unique()))}
    return size_map, comp_map, ind_map

def fit_regression(eng, size_map, comp_map, ind_map):
    X = np.array([
        [
            ind_map[row.client_industry],
            size_map[row.client_size],
            comp_map[row.complexity],
            row.prev_issues
        ]
        for _, row in eng.iterrows()
    ])
    y = eng.hours_spent.values
    model = LinearRegression()
    model.fit(X, y)
    return model

def estimate_hours(model, industry, size, complexity, prev_issues, size_map, comp_map, ind_map):
    xnew = np.array([[ind_map[industry], size_map[size], comp_map[complexity], prev_issues]])
    est_hours = float(model.predict(xnew)[0])
    return est_hours

def score_staff_row(r, req_skills=REQUIRED_SKILLS):
    skills = [s.strip() for s in r.skills.split(",")]
    skill_score = sum(1 for s in req_skills if s in skills)
    level_score = LEVEL_SCORES.get(r.level, 1)
    avail_score = r.available_hours_per_week / 40
    return skill_score * 2 + level_score + avail_score

def match_staff(staff):
    staff_df = staff.copy()
    staff_df["score"] = staff_df.apply(score_staff_row, axis=1)
    staff_df = staff_df.sort_values("score", ascending=False)
    return staff_df

# --- MAIN APP ---
st.set_page_config(page_title="Audit Workflow Optimizer (Improved)", layout="wide")
st.title("Audit Workflow Optimizer — Improved Demo")

eng, staff = load_data()
if eng is None or staff is None:
    st.stop()

st.sidebar.header("New Engagement - Input")
size_map, comp_map, ind_map = encode_features(eng)
industry = st.sidebar.selectbox("Client Industry", sorted(eng.client_industry.unique()), help="Select the industry of the client.")
size = st.sidebar.selectbox("Client Size", ["Small", "Medium", "Large"], help="Select the size of the client.")
complexity = st.sidebar.selectbox("Complexity", ["Low", "Medium", "High"], help="Select the engagement complexity.")
prev_issues = st.sidebar.number_input("Number of previous issues", min_value=0, max_value=10, value=1, help="How many issues were found in previous audits?")
submit = st.sidebar.button("Estimate & Match Staff")

st.header("Historical Engagements (sample)")
with st.expander("Show/Hide Historical Engagements"):
    st.dataframe(eng)

if submit:
    try:
        model = fit_regression(eng, size_map, comp_map, ind_map)
        est_hours = estimate_hours(model, industry, size, complexity, prev_issues, size_map, comp_map, ind_map)
        st.success(f"Estimated total hours for this engagement: {est_hours:.0f} hours")

        st.subheader("Suggested Timeline (weeks)")
        recommended_weeks = max(1, int(np.round(est_hours / (BASELINE_STAFF * HOURS_PER_PERSON_PER_WEEK))))
        st.write(f"Estimated duration: **{recommended_weeks} week(s)** (baseline staffing: {BASELINE_STAFF} people at {HOURS_PER_PERSON_PER_WEEK} hrs/week)")

        st.subheader("Staff Matching (heuristic)")
        staff_df = match_staff(staff)
        st.dataframe(staff_df[["staff_id", "name", "level", "skills", "available_hours_per_week", "score"]])

        st.subheader("Suggested Core Team (top 4)")
        suggested = staff_df.head(4)
        st.table(suggested[["staff_id", "name", "level", "skills"]])

        st.subheader("Scenario Simulation")
        add_delay = st.number_input("If engagement extends by (weeks)", min_value=0, max_value=8, value=0, help="Simulate an extension to the engagement timeline.")
        if add_delay > 0:
            new_weeks = recommended_weeks + int(add_delay)
            st.write(f"New estimated duration: **{new_weeks} week(s)**")
            st.info("AI Suggestion: re-check staff availability; consider moving a Senior from a low-risk engagement or hire temporary Associate.")

        st.download_button(
            label="Download Suggested Team as CSV",
            data=suggested.to_csv(index=False),
            file_name="suggested_team.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"An error occurred during estimation: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Tip: This is a demo with limited capabilities. More features can be added")
