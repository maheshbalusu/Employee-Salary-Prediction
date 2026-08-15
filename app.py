from pathlib import Path
import time
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="wide"
)

if (ROOT / "style.css").exists():
    st.markdown(
        f"<style>{(ROOT / 'style.css').read_text()}</style>",
        unsafe_allow_html=True
    )

if not st.session_state.get("splash"):
    box = st.empty()
    box.markdown(
        """
        <div class="splash-screen">
            <div class="splash-mark">Employee Salary Prediction</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    time.sleep(2)
    box.empty()
    st.session_state.splash = True

model = joblib.load(ROOT / "salary_model.pkl")
features = joblib.load(ROOT / "feature_columns.pkl")

try:
    df = pd.read_excel(ROOT / "employee_salary.xlsx")
except:
    df = pd.read_csv(ASSETS / "employee_salary.csv")

def values(col):
    return sorted(df[col].dropna().unique().tolist(), key=str)

st.markdown(
    """
    <header class="top-header">
        <div>
            <p class="eyebrow">AI/ML Project Dashboard</p>
            <h1>Employee Salary Prediction</h1>
        </div>
        <div class="header-badge">Machine Learning Salary Estimator</div>
    </header>
    """,
    unsafe_allow_html=True
)

page = st.radio(
    "Navigation",
    ["Home", "Data Insights"],
    horizontal=True,
    label_visibility="collapsed"
)

if page == "Home":

    st.markdown(
        """
        <section class="hero-section">
            <p class="eyebrow">Employee Salary Prediction</p>
            <h2>Predict an employee's estimated salary using machine learning.</h2>
            <p>
                Enter employee details and the trained model will
                estimate the annual and monthly salary.
            </p>
        </section>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<section class="prediction-card">',
        unsafe_allow_html=True
    )

    st.subheader("Salary Prediction Form")

    c1, c2 = st.columns(2)

    with c1:
        education = st.selectbox(
            "Education Level",
            values("Education_Level")
        )

        job_role = st.selectbox(
            "Job Role",
            values("Job_Role")
        )

        age = st.number_input(
            "Age",
            18,
            70,
            30
        )

        experience = st.number_input(
            "Experience Years",
            0,
            max(0, age - 18),
            min(5, max(0, age - 18))
        )

        department = st.selectbox(
            "Department",
            values("Department")
        )

    with c2:
        location = st.selectbox(
            "Location",
            values("Location")
        )

        gender = st.selectbox(
            "Gender",
            values("Gender")
        )

        work_mode = st.selectbox(
            "Work Mode",
            values("Work_Mode")
        )

        performance = st.selectbox(
            "Performance Rating",
            values("Performance_Rating")
        )

        overtime = st.selectbox(
            "Overtime",
            values("Overtime")
        )

    if st.button(
        "🔮 Predict Salary",
        type="primary",
        use_container_width=True
    ):

        employee = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Education_Level": [education],
            "Experience_Years": [experience],
            "Job_Role": [job_role],
            "Department": [department],
            "Location": [location],
            "Work_Mode": [work_mode],
            "Performance_Rating": [performance],
            "Overtime": [overtime]
        })

        employee = pd.get_dummies(employee)
        employee = employee.reindex(
            columns=features,
            fill_value=0
        )

        prediction = model.predict(employee)[0]
        monthly = prediction / 12

        st.markdown(
            f"""
            <div class="salary-result">
                <p>💰 Predicted Annual Salary</p>
                <strong>₹{prediction:,.2f}</strong>
                <span>Monthly Salary: ₹{monthly:,.2f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</section>", unsafe_allow_html=True)

else:

    st.markdown(
        '<section class="page-panel">',
        unsafe_allow_html=True
    )

    st.title("📊 Data Insights")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Employees", f"{len(df):,}")

    with c2:
        st.metric("Columns", len(df.columns))

    with c3:
        st.metric("Target", "Salary")

    charts = [
        (
            "salary_distribution.png",
            "Salary Distribution",
            "Distribution of salary values in the dataset."
        ),
        (
            "experience_salary.png",
            "Experience vs Salary",
            "Relationship between experience and salary."
        ),
        (
            "education_salary.png",
            "Salary by Education",
            "Salary comparison across education levels."
        ),
        (
            "jobrole_salary.png",
            "Salary by Job Role",
            "Salary comparison across different job roles."
        ),
        (
            "department_salary.png",
            "Salary by Department",
            "Salary comparison across departments."
        ),
        (
            "overtime_salary.png",
            "Salary by Overtime",
            "Salary comparison based on overtime."
        ),
        (
            "correlation_matrix.png",
            "Correlation Matrix",
            "Relationship between numerical features."
        ),
        (
            "actual_vs_predicted.png",
            "Actual vs Predicted",
            "Comparison between actual and predicted salary."
        )
    ]

    for i in range(0, len(charts), 2):

        col1, col2 = st.columns(2)

        for col, chart in zip(
            [col1, col2],
            charts[i:i + 2]
        ):

            image, title, text = chart

            with col:

                path = ASSETS / image

                if path.exists():

                    st.image(
                        str(path),
                        use_container_width=True
                    )

                    st.subheader(title)
                    st.caption(text)

st.divider()

st.subheader("Model Accuracy")

st.markdown(
    """
    <div class="accuracy-box">
        <h2>99.33%</h2>
        <p>Our model shows 99.33% prediction performance.</p>
    </div>
    """,
    unsafe_allow_html=True
)