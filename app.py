from pathlib import Path
import time
import joblib
import pandas as pd
import streamlit as st

# Setup Paths
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"

# Page Configuration
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="wide"
)

# Load CSS
css_path = ROOT / "style.css"
if css_path.exists():
    st.markdown(
        f"<style>{css_path.read_text()}</style>",
        unsafe_allow_html=True
    )

# Splash Screen Logic
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

# Load Models and Data
@st.cache_resource
def load_assets():
    model = joblib.load(ROOT / "salary_model.pkl")
    features = joblib.load(ROOT / "feature_columns.pkl")
    return model, features

@st.cache_data
def load_data():
    try:
        df = pd.read_excel(ROOT / "employee_salary.xlsx")
    except Exception:  # Catch all errors (like missing openpyxl) and safely fallback to CSV
        df = pd.read_csv(ASSETS / "employee_salary.csv")
    return df

model, features = load_assets()
df = load_data()

# Helper function for dropdowns
def get_unique_values(col):
    return sorted(df[col].dropna().unique().tolist(), key=str)

# Header Section
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

# Navigation
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

    st.markdown('<section class="prediction-card">', unsafe_allow_html=True)
    st.subheader("Salary Prediction Form")

    # Form Layout
    c1, c2 = st.columns(2)

    with c1:
        education = st.selectbox("Education Level", get_unique_values("Education_Level"))
        job_role = st.selectbox("Job Role", get_unique_values("Job_Role"))
        age = st.number_input("Age", min_value=18, max_value=70, value=30)
        
        # Calculate experience bounds based on age
        max_exp = max(0, age - 18)
        default_exp = min(5, max_exp)
        experience = st.number_input("Experience Years", min_value=0, max_value=max_exp, value=default_exp)
        
        department = st.selectbox("Department", get_unique_values("Department"))

    with c2:
        location = st.selectbox("Location", get_unique_values("Location"))
        gender = st.selectbox("Gender", get_unique_values("Gender"))
        work_mode = st.selectbox("Work Mode", get_unique_values("Work_Mode"))
        performance = st.selectbox("Performance Rating", get_unique_values("Performance_Rating"))
        overtime = st.selectbox("Overtime", get_unique_values("Overtime"))

    # Prediction Logic
    if st.button("🚀 Predict Salary", type="primary", use_container_width=True):
        employee_data = pd.DataFrame({
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

        employee_data = pd.get_dummies(employee_data)
        employee_data = employee_data.reindex(columns=features, fill_value=0)

        prediction = model.predict(employee_data)[0]
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
    # Data Insights Page
    st.markdown('<section class="page-panel">', unsafe_allow_html=True)
    st.title("📊 Data Insights")

    # Metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Employees", f"{len(df):,}")
    with c2:
        st.metric("Columns", len(df.columns))
    with c3:
        st.metric("Target", "Salary")

    # Charts configuration
    charts = [
        ("salary_distribution.png", "Salary Distribution", "Distribution of salary values in the dataset."),
        ("experience_salary.png", "Experience vs Salary", "Relationship between experience and salary."),
        ("education_salary.png", "Salary by Education", "Salary comparison across education levels."),
        ("jobrole_salary.png", "Salary by Job Role", "Salary comparison across different job roles."),
        ("department_salary.png", "Salary by Department", "Salary comparison across departments."),
        ("overtime_salary.png", "Salary by Overtime", "Salary comparison based on overtime."),
        ("correlation_matrix.png", "Correlation Matrix", "Relationship between numerical features."),
        ("actual_vs_predicted.png", "Actual vs Predicted", "Comparison between actual and predicted salary.")
    ]

    # Render Charts in rows of 2
    for i in range(0, len(charts), 2):
        col1, col2 = st.columns(2)
        for col, chart in zip([col1, col2], charts[i:i + 2]):
            image_name, title, description = chart
            with col:
                path = ASSETS / image_name
                if path.exists():
                    st.image(str(path), use_container_width=True)
                    st.subheader(title)
                    st.caption(description)
                else:
                    st.info(f"Image not found: {image_name}")

st.divider()

# Footer / Accuracy Section
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