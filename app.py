import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# ============================================================
# 1. PAGE CONFIGURATION AND PATHS
# ============================================================

st.set_page_config(
    page_title="Patient Health Profile Segmentation",
    page_icon="🩺",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "diabetes_012_health_indicators_BRFSS2015.csv"

# Keep this order exactly aligned with the model's training features.
FEATURES = [
    "HighBP",
    "HighChol",
    "BMI",
    "Stroke",
    "HeartDiseaseorAttack",
    "GenHlth",
    "DiffWalk",
    "Smoker",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "MentHlth",
    "PhysHlth",
    "Age",
]

BINARY_FEATURE_LABELS = {
    "HighBP": "High blood pressure",
    "HighChol": "High cholesterol",
    "Stroke": "History of stroke",
    "HeartDiseaseorAttack": "Heart disease or heart attack",
    "DiffWalk": "Difficulty walking or climbing stairs",
    "Smoker": "Smoking indicator",
    "PhysActivity": "Physical activity indicator",
    "Fruits": "Fruit consumption indicator",
    "Veggies": "Vegetable consumption indicator",
    "HvyAlcoholConsump": "Heavy alcohol consumption indicator",
}

BINARY_FEATURES = list(BINARY_FEATURE_LABELS.keys())

AGE_LABELS = {
    1: "18–24 years",
    2: "25–29 years",
    3: "30–34 years",
    4: "35–39 years",
    5: "40–44 years",
    6: "45–49 years",
    7: "50–54 years",
    8: "55–59 years",
    9: "60–64 years",
    10: "65–69 years",
    11: "70–74 years",
    12: "75–79 years",
    13: "80 years or older",
}

DIABETES_LABELS = {
    0: "No diabetes reported",
    1: "Prediabetes reported",
    2: "Diabetes reported",
}


# ============================================================
# 2. LOAD THE SAVED MODEL, SCALER, AND DATASET
# ============================================================

@st.cache_resource
def load_models():
    scaler_path = MODEL_DIR / "scaler.pkl"
    model_path = MODEL_DIR / "kmeans_model.pkl"

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. "
            "Make sure models/scaler.pkl exists."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"K-Means model not found at {model_path}. "
            "Make sure models/kmeans_model.pkl exists."
        )

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return scaler, model


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place diabetes_012_health_indicators_BRFSS2015.csv "
            "in the project root, beside app.py."
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + ["Diabetes_012"]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"The dataset is missing required columns: {missing_columns}"
        )

    # K-Means expects the same feature columns and order used in training.
    if data[FEATURES].isna().any().any():
        raise ValueError(
            "Missing values were found in the model features. "
            "Review the dataset preprocessing used during training."
        )

    return data


try:
    scaler, kmeans_model = load_models()
    cluster_data = load_dataset()

    # Diabetes_012 is deliberately NOT included in FEATURES.
    scaled_dataset = scaler.transform(cluster_data[FEATURES])
    cluster_data["Cluster"] = kmeans_model.predict(scaled_dataset)

except Exception as error:
    st.error("The app could not load its model or dataset.")
    st.exception(error)
    st.stop()


# ============================================================
# 3. APP INTRODUCTION
# ============================================================

st.title("🩺 Patient Health Profile Segmentation")

st.write(
    "This project uses unsupervised machine learning to discover "
    "groups of survey respondents with similar health and lifestyle "
    "characteristics. After creating those groups, it explores how "
    "reported diabetes status varies between them."
)

st.info(
    "Educational and exploratory project only. This app does not "
    "diagnose diabetes, estimate personal disease risk, or provide "
    "medical advice. Results describe patterns in a historical survey dataset."
)

with st.expander("What does the model use?"):
    st.markdown(
        """
        - **Clustering inputs:** 15 health and lifestyle features, including
          BMI, age category, general health, blood pressure, cholesterol,
          physical activity, and other survey indicators.
        - **Algorithm:** the saved K-Means model, with the saved MinMaxScaler.
        - **Diabetes status:** `Diabetes_012` is excluded from clustering and
          examined only after clusters have been assigned.
        - **Purpose:** explore patterns in the survey, not classify a person
          as diabetic or non-diabetic.
        """
    )

st.divider()


# ============================================================
# 4. HEALTH AND LIFESTYLE QUESTIONNAIRE
# ============================================================

st.header("Explore a health profile")
st.write(
    "Enter the responses you want to explore. The app will assign the "
    "submitted feature values to one of the segments learned from the dataset."
)

with st.form("health_questionnaire"):
    st.subheader("Health history")

    high_bp = st.selectbox(
        "Have you been told you have high blood pressure?",
        ["No", "Yes"],
    )
    high_chol = st.selectbox(
        "Have you been told you have high cholesterol?",
        ["No", "Yes"],
    )
    stroke = st.selectbox(
        "Have you ever had a stroke?",
        ["No", "Yes"],
    )
    heart_disease = st.selectbox(
        "Have you had coronary heart disease or a heart attack?",
        ["No", "Yes"],
    )
    diff_walk = st.selectbox(
        "Do you have serious difficulty walking or climbing stairs?",
        ["No", "Yes"],
    )

    st.subheader("Lifestyle")
    smoker = st.selectbox(
        "Have you smoked at least 100 cigarettes in your life?",
        ["No", "Yes"],
    )
    phys_activity = st.selectbox(
        "Have you done physical activity or exercise outside your regular "
        "job in the past 30 days?",
        ["No", "Yes"],
    )
    fruits = st.selectbox(
        "Does your fruit consumption meet the survey indicator?",
        ["No", "Yes"],
    )
    veggies = st.selectbox(
        "Does your vegetable consumption meet the survey indicator?",
        ["No", "Yes"],
    )
    heavy_alcohol = st.selectbox(
        "Does your alcohol consumption meet the survey's heavy-drinking definition?",
        ["No", "Yes"],
    )

    st.caption(
        "Some questions refer to specific BRFSS survey definitions. "
        "These simplified questions do not reproduce every detail of the "
        "original survey instrument."
    )

    st.subheader("Health measurements")
    bmi = st.number_input(
        "Body Mass Index (BMI)",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=0.1,
    )

    general_health = st.selectbox(
        "How would you rate your general health?",
        options=[1, 2, 3, 4, 5],
        format_func=lambda value: {
            1: "Excellent",
            2: "Very good",
            3: "Good",
            4: "Fair",
            5: "Poor",
        }[value],
    )

    mental_health_days = st.number_input(
        "Days mental health was not good in the past 30 days",
        min_value=0,
        max_value=30,
        value=0,
        step=1,
    )
    physical_health_days = st.number_input(
        "Days physical health was not good in the past 30 days",
        min_value=0,
        max_value=30,
        value=0,
        step=1,
    )

    selected_age_label = st.selectbox(
        "What is your age range?",
        options=list(AGE_LABELS.values()),
    )
    age_category = next(
        code for code, label in AGE_LABELS.items()
        if label == selected_age_label
    )

    submitted = st.form_submit_button(
        "Explore health profile",
        type="primary",
        width='stretch',
    )


# ============================================================
# 5. ENCODE INPUTS AND ASSIGN A CLUSTER
# ============================================================

if submitted:
    input_values = [
        1 if high_bp == "Yes" else 0,
        1 if high_chol == "Yes" else 0,
        float(bmi),
        1 if stroke == "Yes" else 0,
        1 if heart_disease == "Yes" else 0,
        int(general_health),
        1 if diff_walk == "Yes" else 0,
        1 if smoker == "Yes" else 0,
        1 if phys_activity == "Yes" else 0,
        1 if fruits == "Yes" else 0,
        1 if veggies == "Yes" else 0,
        1 if heavy_alcohol == "Yes" else 0,
        int(mental_health_days),
        int(physical_health_days),
        int(age_category),
    ]

    input_df = pd.DataFrame([input_values], columns=FEATURES)
    scaled_input = scaler.transform(input_df)
    predicted_cluster = int(kmeans_model.predict(scaled_input)[0])

    st.session_state["profile_input"] = input_values
    st.session_state["profile_cluster"] = predicted_cluster


# ============================================================
# 6. DISPLAY THE RESULTS AFTER SUBMISSION
# ============================================================

if "profile_input" in st.session_state:
    input_df = pd.DataFrame(
        [st.session_state["profile_input"]],
        columns=FEATURES,
    )
    predicted_cluster = int(st.session_state["profile_cluster"])

    group = cluster_data[
        cluster_data["Cluster"] == predicted_cluster
    ].copy()

    if group.empty:
        st.error("No survey records were found for this segment.")
        st.stop()

    segment_size = len(group)
    dataset_size = len(cluster_data)
    segment_share = segment_size / dataset_size * 100

    segment_rates = group[BINARY_FEATURES].mean() * 100
    overall_rates = cluster_data[BINARY_FEATURES].mean() * 100
    segment_means = group[FEATURES].mean()

    # --------------------------------------------------------
    # REVIEWER VIEW: PATIENT INTERVIEW SUMMARY
    # --------------------------------------------------------

    st.divider()
    with st.expander("Reviewer view: patient interview summary", expanded=True):
        st.write(
            "A structured summary of the responses entered above, with neutral "
            "topics a reviewer may wish to clarify. This is not a clinical assessment."
        )

        st.markdown("#### Reported health history")
        history_items = {
            "High blood pressure": int(input_df.iloc[0]["HighBP"]),
            "High cholesterol": int(input_df.iloc[0]["HighChol"]),
            "History of stroke": int(input_df.iloc[0]["Stroke"]),
            "Coronary heart disease or heart attack": int(input_df.iloc[0]["HeartDiseaseorAttack"]),
            "Difficulty walking or climbing stairs": int(input_df.iloc[0]["DiffWalk"]),
        }
        history_df = pd.DataFrame(
            [{"Reported item": label, "Submitted response": "Yes" if value == 1 else "No"}
             for label, value in history_items.items()]
        )
        st.dataframe(history_df, hide_index=True, width='stretch')

        st.markdown("#### Reported lifestyle indicators")
        lifestyle_items = {
            "Smoking indicator": int(input_df.iloc[0]["Smoker"]),
            "Physical activity indicator": int(input_df.iloc[0]["PhysActivity"]),
            "Fruit consumption indicator": int(input_df.iloc[0]["Fruits"]),
            "Vegetable consumption indicator": int(input_df.iloc[0]["Veggies"]),
            "Heavy alcohol consumption indicator": int(input_df.iloc[0]["HvyAlcoholConsump"]),
        }
        lifestyle_df = pd.DataFrame(
            [{"Reported item": label, "Submitted response": "Yes" if value == 1 else "No"}
             for label, value in lifestyle_items.items()]
        )
        st.dataframe(lifestyle_df, hide_index=True, width='stretch')

        st.markdown("#### Self-reported measurements")
        submitted_age_label = AGE_LABELS.get(int(input_df.iloc[0]["Age"]), "Unknown age category")
        submitted_health_rating = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}.get(
            int(input_df.iloc[0]["GenHlth"]), "Unknown"
        )
        measurement_cols = st.columns(3)
        measurement_cols[0].metric("BMI entered", f"{float(input_df.iloc[0]['BMI']):.1f}")
        measurement_cols[1].metric("General health", submitted_health_rating)
        measurement_cols[2].metric("Age range", submitted_age_label)
        day_cols = st.columns(2)
        day_cols[0].metric("Mental health-not-good days", f"{int(input_df.iloc[0]['MentHlth'])}/30")
        day_cols[1].metric("Physical health-not-good days", f"{int(input_df.iloc[0]['PhysHlth'])}/30")

        st.markdown("#### Neutral topics to clarify during an interview")
        follow_up_topics = []
        if int(input_df.iloc[0]["HighBP"]) == 1:
            follow_up_topics.append("Clarify whether the reported high blood pressure is current and whether the person is receiving follow-up care.")
        if int(input_df.iloc[0]["HighChol"]) == 1:
            follow_up_topics.append("Clarify the context and recency of the reported high cholesterol history.")
        if int(input_df.iloc[0]["Stroke"]) == 1:
            follow_up_topics.append("Ask whether the reported stroke history has any ongoing effects the person wants to describe.")
        if int(input_df.iloc[0]["HeartDiseaseorAttack"]) == 1:
            follow_up_topics.append("Clarify the reported heart disease or heart attack history and any current follow-up, if relevant to the conversation.")
        if int(input_df.iloc[0]["DiffWalk"]) == 1:
            follow_up_topics.append("Ask what activities are affected by the reported difficulty walking or climbing stairs.")
        if int(input_df.iloc[0]["PhysHlth"]) > 0:
            follow_up_topics.append("Ask what contributed to the reported physical-health-not-good days and whether the person wants to add context.")
        if int(input_df.iloc[0]["MentHlth"]) > 0:
            follow_up_topics.append("Offer the person an opportunity to describe the context of the reported mental-health-not-good days, if they are comfortable.")
        if not follow_up_topics:
            follow_up_topics.append("No follow-up topic was triggered by these simple rules. A reviewer can still ask open-ended questions and clarify any response as needed.")

        for topic in follow_up_topics:
            st.markdown(f"- {topic}")

        st.caption(
            "These prompts are simple, rule-based interview aids, not medical recommendations. "
            "They do not recommend tests, diagnose conditions, or determine urgency."
        )

    # --------------------------------------------------------
    # A. SEGMENT SUMMARY
    # --------------------------------------------------------

    bp_rate = float(segment_rates["HighBP"])
    chol_rate = float(segment_rates["HighChol"])
    smoking_rate = float(segment_rates["Smoker"])
    walking_rate = float(segment_rates["DiffWalk"])
    activity_rate = float(segment_rates["PhysActivity"])

    if bp_rate >= 80 and chol_rate >= 80:
        segment_title = "High blood pressure and cholesterol indicators"
        segment_intro = (
            "A high proportion of respondents in this segment reported "
            "both high blood pressure and high cholesterol."
        )
    elif walking_rate >= 70:
        segment_title = "Higher reported difficulty with mobility"
        segment_intro = (
            "A prominent characteristic of this segment is the proportion "
            "of respondents reporting serious difficulty walking or climbing stairs."
        )
    elif smoking_rate >= 90:
        segment_title = "Smoking-associated survey profile"
        segment_intro = (
            "The smoking indicator is particularly prominent in this "
            "segment of the survey dataset."
        )
    elif bp_rate < 30 and chol_rate < 30 and activity_rate >= 70:
        segment_title = "Lower reported blood pressure and cholesterol indicators"
        segment_intro = (
            "This segment has lower reported rates of high blood pressure "
            "and high cholesterol, alongside a relatively high physical-activity indicator."
        )
    else:
        segment_title = "Distinct health and lifestyle profile"
        segment_intro = (
            "This segment represents a combination of health and lifestyle "
            "responses identified by the K-Means clustering algorithm."
        )

    st.divider()
    st.header("1. Discovered health profile")
    st.subheader(segment_title)
    st.write(segment_intro)
    st.write(
        "The submitted feature values were assigned to this segment by the "
        "trained K-Means model after applying the saved preprocessing scaler. "
        "This is a similarity-based grouping, not a medical assessment."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Survey respondents in segment", f"{segment_size:,}")
    col2.metric("Share of analyzed dataset", f"{segment_share:.1f}%")
    col3.metric("Number of learned segments", str(kmeans_model.n_clusters))

    st.caption(
        "Counts and shares describe the analyzed historical dataset; they "
        "are not estimates of the segment's prevalence in the general population."
    )

    # --------------------------------------------------------
    # B. FEATURES THAT DISTINGUISH THIS SEGMENT
    # --------------------------------------------------------

    st.subheader("What distinguishes this segment?")
    st.write(
        "These indicators have the largest absolute differences in their "
        "Yes-response rates compared with the full analyzed dataset."
    )

    differences = segment_rates - overall_rates
    ranked_features = differences.abs().sort_values(ascending=False)

    comparison_rows = []
    for feature in ranked_features.head(5).index:
        segment_rate = float(segment_rates[feature])
        overall_rate = float(overall_rates[feature])
        difference = segment_rate - overall_rate

        if difference > 0:
            direction = "higher"
        elif difference < 0:
            direction = "lower"
        else:
            direction = "equal"

        comparison_rows.append(
            {
                "Indicator": BINARY_FEATURE_LABELS[feature],
                "In this segment (%)": round(segment_rate, 1),
                "Full dataset (%)": round(overall_rate, 1),
                "Difference (percentage points)": round(difference, 1),
                "Comparison": direction,
            }
        )

    st.dataframe(
        pd.DataFrame(comparison_rows),
        hide_index=True,
        width='stretch',
    )
    st.caption(
        "These are descriptive differences. They do not establish causation "
        "or indicate that an individual will develop a condition."
    )

    # --------------------------------------------------------
    # C. COMPARE A SUBMITTED BINARY ANSWER WITH THE SEGMENT
    # --------------------------------------------------------

    st.subheader("Compare one answer with the segment")
    selected_indicator = st.selectbox(
        "Choose a health or lifestyle indicator",
        options=BINARY_FEATURES,
        format_func=lambda feature: BINARY_FEATURE_LABELS[feature],
        key="segment_indicator_selector",
    )

    user_value = int(input_df.iloc[0][selected_indicator])
    segment_yes = float(segment_rates[selected_indicator])
    segment_no = 100.0 - segment_yes

    left, right = st.columns(2)
    left.metric("Your submitted answer", "Yes" if user_value == 1 else "No")
    right.metric("Segment respondents answering Yes", f"{segment_yes:.1f}%")

    selected_chart = pd.DataFrame(
        {
            "Response": ["Yes", "No"],
            "Percentage (%)": [segment_yes, segment_no],
        }
    )
    selected_fig = px.bar(
        selected_chart,
        x="Percentage (%)",
        y="Response",
        orientation="h",
        text="Percentage (%)",
        range_x=[0, 100],
        title="Response distribution in this segment",
    )
    selected_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    selected_fig.update_layout(
        xaxis_title="Share of segment (%)",
        yaxis_title="",
        xaxis={"range": [0, 100], "ticksuffix": "%"},
        showlegend=False,
        margin={"l": 10, "r": 30, "t": 50, "b": 10},
        height=230,
    )
    st.plotly_chart(selected_fig, width="stretch")

    if user_value == 1:
        st.write(
            f"You answered **Yes** to "
            f"{BINARY_FEATURE_LABELS[selected_indicator].lower()}. "
            f"In this segment, {segment_yes:.1f}% of respondents also answered Yes."
        )
    else:
        st.write(
            f"You answered **No** to "
            f"{BINARY_FEATURE_LABELS[selected_indicator].lower()}. "
            f"In this segment, {segment_no:.1f}% of respondents answered No."
        )

    # --------------------------------------------------------
    # D. NUMERIC FEATURE COMPARISON
    # --------------------------------------------------------

    st.subheader("Your measurements compared with segment averages")

    numeric_labels = {
        "BMI": "Body Mass Index (BMI)",
        "GenHlth": "General health rating (1 = excellent, 5 = poor)",
        "MentHlth": "Mental health-not-good days (past 30 days)",
        "PhysHlth": "Physical health-not-good days (past 30 days)",
    }

    numeric_rows = []
    for feature, label in numeric_labels.items():
        numeric_rows.append(
            {
                "Measure": label,
                "Submitted value": round(float(input_df.iloc[0][feature]), 2),
                "Segment average": round(float(segment_means[feature]), 2),
            }
        )

    st.dataframe(
        pd.DataFrame(numeric_rows),
        hide_index=True,
        width='stretch',
    )
    st.caption(
        "Segment averages are descriptive comparisons, not targets or clinical thresholds."
    )

    st.subheader("Age-range distribution in this segment")
    st.write(
        "Because age is recorded as an ordered category in the source survey, "
        "the app shows the distribution of age ranges rather than treating the "
        "category code average as an age in years."
    )
    age_counts = group["Age"].value_counts().reindex(list(AGE_LABELS.keys()), fill_value=0)
    age_distribution = pd.DataFrame({
        "Age range": [AGE_LABELS[code] for code in AGE_LABELS],
        "Respondents (%)": [
            (float(age_counts[code]) / segment_size * 100) if segment_size else 0.0
            for code in AGE_LABELS
        ],
        "Respondents": [int(age_counts[code]) for code in AGE_LABELS],
    })
    age_fig = px.bar(
        age_distribution,
        x="Age range",
        y="Respondents (%)",
        hover_data={"Respondents": True, "Respondents (%)": ":.2f"},
        title="Share of respondents by age range",
    )
    age_fig.update_layout(
        xaxis_title="Survey age range",
        yaxis_title="Share of segment (%)",
        yaxis={"range": [0, 100], "ticksuffix": "%"},
        xaxis={"categoryorder": "array", "categoryarray": list(AGE_LABELS.values())},
        showlegend=False,
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
        height=350,
    )
    st.plotly_chart(age_fig, width="stretch")
    st.dataframe(
        age_distribution.rename(columns={
            "Age range": "Survey age range",
            "Respondents (%)": "Share of segment (%)",
        }).assign(**{"Share of segment (%)": lambda frame: frame["Share of segment (%)"].round(2)}),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"The submitted age range is {AGE_LABELS.get(int(input_df.iloc[0]['Age']), 'Unknown')}. "
        "The chart describes age ranges among historical survey respondents in this segment; "
        "it is not a personal age comparison or a clinical assessment."
    )

    # --------------------------------------------------------
    # E. DIABETES-STATUS PATTERNS AFTER CLUSTERING
    # --------------------------------------------------------

    st.divider()
    st.header("2. Diabetes-related patterns in the research dataset")
    st.write(
        "Diabetes status was **not used to create the clusters**. After the "
        "model assigned survey respondents to segments using the 15 health "
        "and lifestyle features, we examined the dataset's diabetes-status "
        "categories within each segment."
    )

    st.info(
        "This section describes the respondents in the historical survey "
        "dataset who belong to this segment. It does not identify your "
        "diabetes status or calculate your personal probability of diabetes."
    )

    if "Diabetes_012" not in group.columns:
        st.warning(
            "The diabetes-status column is not available, so this analysis "
            "cannot be displayed."
        )
    else:
        diabetes_counts = (
            group["Diabetes_012"]
            .value_counts()
            .reindex([0, 1, 2], fill_value=0)
        )
        diabetes_percentages = (
            group["Diabetes_012"]
            .value_counts(normalize=True)
            .reindex([0, 1, 2], fill_value=0)
            * 100
        )

        diabetes_cols = st.columns(3)
        for index, category_code in enumerate([0, 1, 2]):
            diabetes_cols[index].metric(
                DIABETES_LABELS[category_code],
                f"{diabetes_percentages[category_code]:.1f}%",
                help=f"{int(diabetes_counts[category_code]):,} respondents in this segment",
            )

        # Keep the same category order as the metrics and table.
        diabetes_chart_df = pd.DataFrame(
            {
                "Diabetes-status category": [
                    DIABETES_LABELS[0],
                    DIABETES_LABELS[1],
                    DIABETES_LABELS[2],
                ],
                "Respondents (%)": [
                    round(float(diabetes_percentages[0]), 2),
                    round(float(diabetes_percentages[1]), 2),
                    round(float(diabetes_percentages[2]), 2),
                ],
            }
        )

        diabetes_fig = px.bar(
            diabetes_chart_df,
            x="Respondents (%)",
            y="Diabetes-status category",
            orientation="h",
            text="Respondents (%)",
            range_x=[0, 100],
            title="Share of respondents in this segment",
        )
        diabetes_fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
            cliponaxis=False,
        )
        diabetes_fig.update_layout(
            xaxis_title="Share of segment (%)",
            yaxis_title="",
            yaxis={"categoryorder": "array", "categoryarray": [
                DIABETES_LABELS[2],
                DIABETES_LABELS[1],
                DIABETES_LABELS[0],
            ]},
            xaxis={"range": [0, 100], "ticksuffix": "%"},
            showlegend=False,
            margin={"l": 10, "r": 30, "t": 55, "b": 10},
            height=300,
        )
        st.plotly_chart(diabetes_fig, width='stretch')

        diabetes_table = pd.DataFrame(
            {
                "Survey diabetes-status category": [
                    DIABETES_LABELS[0],
                    DIABETES_LABELS[1],
                    DIABETES_LABELS[2],
                ],
                "Respondents in this segment": [
                    int(diabetes_counts[0]),
                    int(diabetes_counts[1]),
                    int(diabetes_counts[2]),
                ],
                "Share of this segment (%)": [
                    round(float(diabetes_percentages[0]), 2),
                    round(float(diabetes_percentages[1]), 2),
                    round(float(diabetes_percentages[2]), 2),
                ],
            }
        )

        st.dataframe(
            diabetes_table,
            hide_index=True,
            width='stretch',
        )

        st.caption(
            "The source dataset codes are 0 = no diabetes, 1 = prediabetes, "
            "and 2 = diabetes. These are survey-reported categories, not "
            "independent clinical verification. Percentages may differ across "
            "segments and do not establish that cluster membership causes diabetes."
        )

    # --------------------------------------------------------
    # F. OVERVIEW OF ALL BINARY INDICATORS IN THE SEGMENT
    # --------------------------------------------------------

    with st.expander("View all health and lifestyle indicator rates"):
        group_percentages = group[BINARY_FEATURES].mean() * 100

        all_indicator_df = pd.DataFrame(
            {
                "Indicator": [
                    BINARY_FEATURE_LABELS[feature]
                    for feature in BINARY_FEATURES
                ],
                "Respondents answering Yes (%)": [
                    round(float(group_percentages[feature]), 1)
                    for feature in BINARY_FEATURES
                ],
            }
        )
        all_indicator_fig = px.bar(
            all_indicator_df,
            x="Respondents answering Yes (%)",
            y="Indicator",
            orientation="h",
            text="Respondents answering Yes (%)",
            range_x=[0, 100],
            title="Yes-response rates in this segment",
        )
        all_indicator_fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            cliponaxis=False,
        )
        all_indicator_fig.update_layout(
            xaxis_title="Respondents answering Yes (%)",
            yaxis_title="",
            xaxis={"range": [0, 100], "ticksuffix": "%"},
            yaxis={"categoryorder": "array", "categoryarray": [
                BINARY_FEATURE_LABELS[feature] for feature in reversed(BINARY_FEATURES)
            ]},
            showlegend=False,
            margin={"l": 10, "r": 30, "t": 55, "b": 10},
            height=420,
        )
        st.plotly_chart(all_indicator_fig, width="stretch")

    # --------------------------------------------------------
    # G. METHODOLOGY AND LIMITATIONS
    # --------------------------------------------------------

    with st.expander("How was this result generated?"):
        st.markdown(
            """
            1. The submitted answers were encoded into the 15 model features.
            2. The saved `MinMaxScaler` transformed those features.
            3. The saved K-Means model assigned the transformed observation
               to its nearest learned cluster.
            4. The application compared the segment with the full dataset.
            5. Separately, it summarized `Diabetes_012` for survey respondents
               already assigned to that segment.

            **Important:** `Diabetes_012` is not part of the 15 clustering
            features. It is used only for the post-clustering descriptive analysis.
            """
        )

        st.write(f"Internal cluster ID: {predicted_cluster}")
        st.write(f"Input feature count: {len(FEATURES)}")
        st.write("Feature names and values sent to the model:")
        st.dataframe(input_df, hide_index=True, width='stretch')

    st.warning(
        "Limitations: this is an exploratory portfolio project based on "
        "historical survey data. The clusters are not validated clinical "
        "categories; clustering quality and stability are limited. The "
        "questionnaire uses simplified wording, and this tool must not be "
        "used for diagnosis, treatment, screening decisions, or other medical decisions."
    )

else:
    st.caption(
        "Submit the questionnaire to see the discovered segment, descriptive "
        "comparisons, and diabetes-status patterns for that segment."
    )


# ============================================================
# 7. PROJECT FOOTER
# ============================================================

st.divider()
st.caption(
    "Patient Health Profile Segmentation | Unsupervised learning with K-Means | "
    "Exploratory analysis of BRFSS survey data"
)
