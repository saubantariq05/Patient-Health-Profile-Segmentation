# Patient Health Profile Segmentation using K-Means

An exploratory machine-learning portfolio project that groups survey respondents by health and lifestyle characteristics using **K-Means clustering**, then describes how survey-reported diabetes status varies across the discovered groups.

> **Important:** This is an educational data-science prototype, not a medical device. It does not diagnose diabetes, predict an individual's diabetes risk, or provide treatment advice. Do not use it to make clinical decisions.

## Project objective

1. Use 15 health and lifestyle features to discover groups with similar feature patterns.
2. Describe the resulting segments using cluster-level summaries.
3. After clustering, examine the distribution of the dataset's `Diabetes_012` categories within each segment.
4. Present the results in an interactive Streamlit application, including a patient-input summary and a reviewer-oriented interview summary.

`Diabetes_012` is **not included** in the clustering features. It is used only for post-clustering descriptive analysis.

## App features

- Interactive questionnaire for the 15 clustering features.
- Saved `MinMaxScaler` and K-Means model used to assign a submitted profile to a segment.
- Segment size and dataset share.
- Indicators that differ most between the selected segment and the full dataset.
- Comparison of submitted binary answers with segment-level response rates.
- Comparison of numeric inputs with segment averages.
- Reviewer view that summarizes submitted answers and generates neutral follow-up topics.
- Diabetes-status counts and percentages within the assigned segment, clearly labeled as survey-level research context.
- Methodology and limitations displayed in the app.
- Diabetes chart with a fixed 0–100% scale and consistent category ordering.

## Dataset

The project uses the **Diabetes Health Indicators / BRFSS 2015** dataset file:

`diabetes_012_health_indicators_BRFSS2015.csv`

Dataset source URL used in the project:

https://raw.githubusercontent.com/razibdash/Diabetes-Health-Indicators/main/diabetes_012_health_indicators_BRFSS2015.csv

The dataset contains 253,680 rows and the `Diabetes_012` column, with codes used here as:
- `0`: No diabetes reported
- `1`: Prediabetes reported
- `2`: Diabetes reported

These categories represent survey data and are not independent clinical verification. Review the dataset's original source and terms before redistributing it.

## Model features

The app uses these 15 columns in this exact order:

```text
HighBP
HighChol
BMI
Stroke
HeartDiseaseorAttack
GenHlth
DiffWalk
Smoker
PhysActivity
Fruits
Veggies
HvyAlcoholConsump
MentHlth
PhysHlth
Age
```

The saved scaler and model must have been trained with the same feature order and compatible preprocessing.

## Current reference results

The current saved model reproduced these cluster sizes during validation:

| Cluster ID | Respondents | Dataset share |
|---:|---:|---:|
| 0 | 35,037 | 13.81% |
| 1 | 70,885 | 27.94% |
| 2 | 46,401 | 18.29% |
| 3 | 42,146 | 16.61% |
| 4 | 59,211 | 23.34% |

Post-clustering diabetes-status percentages:

| Cluster ID | No diabetes reported | Prediabetes reported | Diabetes reported |
|---:|---:|---:|---:|
| 0 | 62.01% | 3.23% | 34.76% |
| 1 | 93.10% | 1.13% | 5.77% |
| 2 | 72.78% | 2.97% | 24.25% |
| 3 | 90.98% | 1.30% | 7.71% |
| 4 | 90.98% | 1.30% | 7.72% |

These figures describe the current saved model and dataset. Cluster IDs are arbitrary labels and may change if the model is retrained. The app calculates displayed figures dynamically from the dataset and current saved model.

## Project structure

Place files like this:

```text
Patient_Health_Profile_Segmentation/
├── app.py
├── validate_results.py
├── requirements.txt
├── README.md
├── .gitignore
├── diabetes_012_health_indicators_BRFSS2015.csv
└── models/
    ├── scaler.pkl
    └── kmeans_model.pkl
```

The dataset and model files are not bundled into this starter package. Keep your existing CSV and `models/` directory in the project root as shown.

## Setup on Windows

### 1. Open the project folder

Open the folder in VS Code and open a terminal in the project root.

### 2. Create and activate a virtual environment (recommended)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the app

```bash
python -m streamlit run app.py
```

### 5. Validate the saved model and segment summaries

In a separate terminal:

```bash
python validate_results.py
```

The validation script recreates cluster assignments with the saved scaler and model, then prints cluster sizes, feature profiles, and diabetes-status distributions.

## Methodology

1. Read the dataset.
2. Select the 15 health and lifestyle features in the training order.
3. Transform features using the saved `MinMaxScaler`.
4. Assign rows using the saved K-Means model.
5. Summarize the features within each cluster.
6. Analyze `Diabetes_012` **after** assigning clusters, not as a clustering input.

## Limitations

- This is exploratory clustering, not a validated clinical classification system.
- Earlier evaluation showed relatively low silhouette scores and sensitivity to initialization. Segment separation and stability are limitations and should be reported honestly.
- The questionnaire uses simplified wording and may not capture every detail of the original BRFSS questions or coding.
- Dataset distributions may not generalize to a different country, time period, or population.
- A segment-level diabetes percentage is not an individual's probability of diabetes.
- Associations do not establish causation.
- The app is not connected to patient records, user accounts, or a secure clinical workflow. Do not enter identifiable patient information.

## Deployment

Streamlit Community Cloud is the simplest deployment route for this Streamlit app. The deployed project needs `app.py`, `requirements.txt`, and the model/data assets it reads at runtime. Before publishing, decide how to handle the dataset and model files, check repository limits and dataset terms, and avoid uploading personal or sensitive information.

Vercel is primarily suited to frontend/serverless workflows and is not the straightforward native host for a Streamlit app. If Vercel is a firm requirement, a separate frontend plus a separately hosted API would be a different architecture.

## Suggested portfolio description

**Patient Health Profile Segmentation using K-Means** — Built an interactive Streamlit application that applies a saved MinMaxScaler and K-Means model to group survey respondents using 15 health and lifestyle features. Performed post-clustering analysis of survey-reported diabetes-status distributions by segment, with transparent methodology, reviewer-oriented summaries, and explicit limitations.

## License and attribution

Confirm the dataset's original license and terms before redistributing it. Add a project license only after deciding which license applies to your own code and ensuring it is compatible with the dataset and any included assets.
