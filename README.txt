
Audit Workflow Optimizer — Prototype Demo
=======================================

What this prototype includes (saved in this folder):
- engagements_sample.csv : sample historical engagement dataset
- staff_sample.csv : sample staff dataset
- app.py : simple Streamlit demo that:
    * estimates engagement hours using a simple linear regression (from historicals)
    * suggests a baseline timeline
    * matches staff using a heuristic score
    * allows a simple scenario simulation for delays

How to run the demo locally:
1. Ensure you have Python 3.8+ and pip.
2. Install dependencies:
    pip install streamlit pandas numpy
3. Run the app:
    streamlit run app.py
4. The app will open in your browser. Use the sidebar to create a new engagement and click "Estimate & Match Staff".

Notes:
- This is a lightweight prototype built for the Hack-a-thon to showcase core concepts.
- In production, replace the heuristic model with a trained ML model (scikit-learn or similar),
  add authentication, integrate with the firm's HR/time systems, and implement constraints (budgets, skills, client preferences).

Files saved to: /mnt/data/audit_workflow_optimizer_demo
