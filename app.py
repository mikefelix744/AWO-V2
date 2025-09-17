
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
eng = pd.read_csv(ROOT/"engagements_sample.csv")
staff = pd.read_csv(ROOT/"staff_sample.csv")

st.set_page_config(page_title="Audit Workflow Optimizer (Prototype)", layout="wide")

st.title("Audit Workflow Optimizer — Prototype Demo")

st.sidebar.header("New Engagement - Input")
industry = st.sidebar.selectbox("Client Industry", sorted(eng.client_industry.unique()))
size = st.sidebar.selectbox("Client Size", ["Small","Medium","Large"])
complexity = st.sidebar.selectbox("Complexity", ["Low","Medium","High"])
prev_issues = st.sidebar.number_input("Number of previous issues", min_value=0, max_value=10, value=1)
submit = st.sidebar.button("Estimate & Match Staff")

st.header("Historical Engagements (sample)")
st.dataframe(eng)

if submit:
    # Simple feature encoding
    size_map = {"Small":0,"Medium":1,"Large":2}
    comp_map = {"Low":0,"Medium":1,"High":2}
    ind_map = {v:i for i,v in enumerate(sorted(eng.client_industry.unique()))}
    # Build X and y from historicals
    X = []
    y = eng.hours_spent.values
    for _, row in eng.iterrows():
        X.append([ind_map[row.client_industry], size_map[row.client_size], comp_map[row.complexity], row.prev_issues])
    X = np.array(X, dtype=float)
    # Fit a simple linear regression (normal equation)
    X_design = np.hstack([np.ones((X.shape[0],1)), X])
    theta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y
    # Prepare input
    xnew = np.array([1, ind_map[industry], size_map[size], comp_map[complexity], prev_issues])
    est_hours = float(xnew @ theta)
    st.success(f"Estimated total hours for this engagement: {est_hours:.0f} hours")

    st.subheader("Suggested Timeline (weeks)")
    # simple timeline: assume 30 hours/week per fulltime staff
    recommended_weeks = max(1, int(np.round(est_hours / 120)))  # baseline 4 people * 30 hrs = 120 hrs/week
    st.write(f"Estimated duration: **{recommended_weeks} week(s)** (baseline staffing: 4 people at 30 hrs/week)")

    st.subheader("Staff Matching (heuristic)")
    # simplistic scoring: skill match + level weight + availability
    req_skills = ["Audit"]  # assume all engagements require Audit
    staff_df = staff.copy()
    def score_row(r):
        skills = [s.strip() for s in r.skills.split(",")]
        skill_score = sum(1 for s in req_skills if s in skills)
        level_score = {"Junior":1,"Associate":2,"Senior":3,"Manager":4}.get(r.level,1)
        avail_score = r.available_hours_per_week / 40
        return skill_score*2 + level_score + avail_score
    staff_df["score"] = staff_df.apply(score_row, axis=1)
    staff_df = staff_df.sort_values("score", ascending=False)
    st.dataframe(staff_df[["staff_id","name","level","skills","available_hours_per_week","score"]])

    st.subheader("Suggested Core Team (top 4)")
    suggested = staff_df.head(4)
    st.table(suggested[["staff_id","name","level","skills"]])

    st.subheader("Scenario Simulation")
    add_delay = st.number_input("If engagement extends by (weeks)", min_value=0, max_value=8, value=0)
    if add_delay>0:
        new_weeks = recommended_weeks + int(add_delay)
        st.write(f"New estimated duration: **{new_weeks} week(s)**")
        st.write("AI Suggestion: re-check staff availability; consider moving a Senior from a low-risk engagement or hire temporary Associate.")
