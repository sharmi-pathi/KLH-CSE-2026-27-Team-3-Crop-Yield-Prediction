# ============================================================
# Crop Yield Prediction Dashboard
# Step 9.7.8 — Integrated Dashboard with Saved EDA Graphs
# ============================================================

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------------------------------------------
# 2. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "Models"
DATASET_PATH = PROJECT_DIR / "datasets" / "crop-yield.csv"

# Search these folders for existing EDA image files.
EDA_GRAPH_FOLDERS = [
    PROJECT_DIR / "EDA_Graphs",
    PROJECT_DIR / "EDA_Graphs" / "Images",
    PROJECT_DIR / "Graphs",
    PROJECT_DIR / "graphs",
    PROJECT_DIR / "EDA",
    PROJECT_DIR / "EDA_Results",
    PROJECT_DIR / "outputs",
    PROJECT_DIR / "notebooks" / "EDA_Graphs"
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# ------------------------------------------------------------
# 3. LOAD SAVED MODEL ARTIFACTS
# ------------------------------------------------------------

@st.cache_resource
def load_model_artifacts():
    model_path = MODELS_DIR / "stacking_regressor_model.joblib"
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    features_path = MODELS_DIR / "final_feature_names.json"

    required_files = [
        model_path,
        preprocessor_path,
        features_path
    ]

    missing_files = [
        path.name for path in required_files
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing model artifacts in Models/: "
            + ", ".join(missing_files)
        )

    model = joblib.load(model_path)
    fitted_preprocessor = joblib.load(preprocessor_path)

    with open(features_path, "r", encoding="utf-8") as file:
        feature_names = json.load(file)

    if getattr(model, "n_features_in_", len(feature_names)) != len(feature_names):
        raise ValueError(
            "Saved model and feature-name counts do not match."
        )

    return model, fitted_preprocessor, feature_names


# ------------------------------------------------------------
# 4. CROP-YIELD PREDICTION FUNCTION
# ------------------------------------------------------------

def predict_crop_yield_dashboard(input_data):
    model, fitted_preprocessor, feature_names = (
        load_model_artifacts()
    )

    raw_columns = [
        "N", "P", "K", "Soil_pH", "Soil_Moisture",
        "Soil_Type", "Organic_Carbon", "Temperature",
        "Humidity", "Rainfall", "Sunlight_Hours",
        "Wind_Speed", "Region", "Altitude", "Season",
        "Crop_Type", "Irrigation_Type",
        "Fertilizer_Used", "Pesticide_Used"
    ]

    data = pd.DataFrame([input_data]).copy()
    data = data[raw_columns]

    numeric_columns = [
        "N", "P", "K", "Soil_pH", "Soil_Moisture",
        "Organic_Carbon", "Temperature", "Humidity",
        "Rainfall", "Sunlight_Hours", "Wind_Speed",
        "Altitude", "Fertilizer_Used", "Pesticide_Used"
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column], errors="raise"
        )

    numeric_values = data[numeric_columns].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "All numerical inputs must be finite numbers."
        )

    if (data["P"] + data["K"] == 0).any():
        raise ValueError("P + K must not equal zero.")

    if (data["N"] + data["P"] + data["K"] == 0).any():
        raise ValueError("N + P + K must not equal zero.")

    # Recreate the six verified engineered features.
    data["NPK_Total"] = data["N"] + data["P"] + data["K"]

    data["N_to_PK_Ratio"] = (
        data["N"] / (data["P"] + data["K"])
    )

    data["Soil_pH_Deviation"] = (
        data["Soil_pH"] - 7
    ).abs()

    data["Moisture_Humidity_Interaction"] = (
        data["Soil_Moisture"] * data["Humidity"] / 100
    )

    data["Rainfall_Sunlight_Interaction"] = (
        data["Rainfall"] * data["Sunlight_Hours"]
    )

    data["Fertilizer_to_NPK_Ratio"] = (
        data["Fertilizer_Used"] / data["NPK_Total"]
    )

    transformed = fitted_preprocessor.transform(data)

    if hasattr(fitted_preprocessor, "get_feature_names_out"):
        cols = fitted_preprocessor.get_feature_names_out()
    else:
        cols = feature_names

    transformed_df = pd.DataFrame(
        transformed,
        columns=cols
    )

    missing_features = [
        feature for feature in feature_names
        if feature not in transformed_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Required model features are missing: {missing_features}"
        )

    X_prediction = transformed_df[feature_names]

    prediction = model.predict(X_prediction)

    return float(prediction[0])


# ------------------------------------------------------------
# 5. LOAD DATASET FOR INSIGHTS
# ------------------------------------------------------------

@st.cache_data
def load_dataset():
    if not DATASET_PATH.is_file():
        return None

    return pd.read_csv(DATASET_PATH)


# ------------------------------------------------------------
# 6. FIND SAVED EDA GRAPHS
# ------------------------------------------------------------

def find_saved_eda_graphs():
    graph_files = set()

    excluded_folders = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "models",
        "datasets"
    }

    for folder in EDA_GRAPH_FOLDERS:
        if not folder.exists():
            continue

        for file_path in folder.rglob("*"):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            if any(
                part.lower() in excluded_folders
                for part in file_path.parts
            ):
                continue

            graph_files.add(file_path)

    return sorted(
        list(graph_files),
        key=lambda path: str(path).lower()
    )


def select_featured_eda_graphs(graph_files):
    graph_choices = [
        (
            "Crop Yield Distribution",
            [
                "crop_yield_distribution",
                "yield_distribution",
                "yield_histogram",
                "yield_dist",
                "target_distribution",
                "target_histogram",
                "crop_yield_hist",
                "yield_dist_plot"
            ]
        ),
        (
            "Crop Type Distribution",
            [
                "crop_type_distribution",
                "crop_type_count",
                "crop_distribution",
                "crop_counts",
                "crop_type_dist"
            ]
        ),
        (
            "Feature Correlation Heatmap",
            [
                "correlation_heatmap",
                "correlation_matrix",
                "corr_heatmap",
                "corr_matrix",
                "correlation"
            ]
        ),
        (
            "Rainfall vs. Crop Yield",
            [
                "rainfall_vs_yield",
                "rainfall_yield",
                "rainfall_vs_crop_yield",
                "rainfall_yield_scatter"
            ]
        ),
        (
            "Temperature vs. Crop Yield",
            [
                "temperature_vs_yield",
                "temperature_yield",
                "temp_vs_yield",
                "temperature_yield_scatter"
            ]
        ),
        (
            "Fertilizer Usage vs. Crop Yield",
            [
                "fertilizer_vs_yield",
                "fertilizer_yield",
                "fertilizer_vs_crop_yield",
                "fertilizer_yield_scatter"
            ]
        )
    ]

    selected_graphs = []
    used_files = set()

    for title, keywords in graph_choices:
        matches = [
            path for path in graph_files
            if path not in used_files
            and any(
                keyword in path.stem.lower()
                for keyword in keywords
            )
        ]

        if matches:
            chosen_file = matches[0]
            selected_graphs.append((title, chosen_file))
            used_files.add(chosen_file)

    fallback_keywords = [
        "yield", "crop", "correlation", "heatmap", "rainfall",
        "temperature", "fertilizer", "feature", "distribution",
        "outlier", "boxplot", "soil", "humidity", "sunlight"
    ]

    for path in graph_files:
        if len(selected_graphs) >= 6:
            break

        if path in used_files:
            continue

        filename = path.stem.lower()

        if any(keyword in filename for keyword in fallback_keywords):
            title = filename.replace("_", " ").replace("-", " ").title()
            selected_graphs.append((title, path))
            used_files.add(path)

    return selected_graphs[:6]


# ------------------------------------------------------------
# 7. SIDEBAR NAVIGATION
# ------------------------------------------------------------

st.sidebar.title("🌾 Crop Yield Prediction")
st.sidebar.caption("Data Science Project")

page = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "EDA",
        "Crop Yield Prediction",
        "Model Evaluation",
        "Dataset Insights"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("Model Status")

artifact_files = {
    "Trained model": MODELS_DIR / "stacking_regressor_model.joblib",
    "Preprocessor": MODELS_DIR / "preprocessor.joblib",
    "Feature names": MODELS_DIR / "final_feature_names.json"
}

for artifact_name, artifact_path in artifact_files.items():
    if artifact_path.is_file():
        st.sidebar.success(f"{artifact_name}: Found")
    else:
        st.sidebar.error(f"{artifact_name}: Missing")

feature_file = artifact_files["Feature names"]

if feature_file.is_file():
    try:
        with open(feature_file, "r", encoding="utf-8") as file:
            saved_feature_names = json.load(file)

        st.sidebar.info(
            f"Model input features: {len(saved_feature_names)}"
        )
    except (OSError, json.JSONDecodeError):
        st.sidebar.warning("Unable to read the saved feature names.")


# ------------------------------------------------------------
# 8. DASHBOARD HEADER
# ------------------------------------------------------------

st.title("🌾 Crop Yield Prediction Dashboard")

st.markdown(
    """
    An interactive Machine Learning application for predicting
    agricultural crop yield using soil nutrients, environmental
    conditions, and farming practices.
    """
)


# ------------------------------------------------------------
# 9. PROJECT OVERVIEW
# ------------------------------------------------------------

if page == "Project Overview":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Records", "10,000")

    with col2:
        st.metric("Input Predictors", "19")

    with col3:
        st.metric("Trained Models", "10")

    st.subheader("Project Objectives")

    st.markdown(
        """
        - Analyze agricultural and environmental factors.
        - Engineer useful features for crop-yield prediction.
        - Compare multiple Machine Learning models.
        - Predict crop yield using the saved Stacking Regressor.
        - Present predictions and insights through a dashboard.
        """
    )

    st.info(
        "Predictions are estimates based on patterns learned "
        "from the project dataset."
    )


# ------------------------------------------------------------
# 10. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------

elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.write(
        "Explore crop yield patterns, relationships between agricultural "
        "variables, multivariate interactions, and numerical correlations."
    )

    eda_root = PROJECT_DIR / "EDA_Graphs"
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}

    def get_eda_images(folder):
        """Find saved image files inside a folder and its subfolders."""
        if not folder.exists():
            return []
        return sorted(
            [
                path for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in image_extensions
            ]
        )

    def display_eda_images(images, empty_message, limit=None):
        """Display saved EDA images safely supporting paths and tuples."""
        if not images:
            st.info(empty_message)
            return

        if limit is not None:
            images = images[:limit]

        for start in range(0, len(images), 2):
            columns = st.columns(2)

            for offset, item in enumerate(images[start:start + 2]):
                caption = ""
                image_path = None

                if isinstance(item, tuple):
                    caption = item[0]
                    image_path = item[1]
                elif isinstance(item, (str, Path)):
                    image_path = Path(item)
                    caption = image_path.stem.replace("_", " ").title()

                if image_path is None or not Path(image_path).exists():
                    continue

                with columns[offset]:
                    st.image(
                        str(image_path),
                        caption=caption,
                        use_container_width=True
                    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Key EDA",
        "Bivariate Analysis",
        "Multivariate Analysis",
        "Numerical Correlation Heatmap"
    ])

    with tab1:
        st.subheader("Important EDA Visualizations")
        st.caption(
            "A selection of saved graphs showing crop yield distributions "
            "and important agricultural patterns."
        )

        graph_files = find_saved_eda_graphs()
        featured_graphs = select_featured_eda_graphs(graph_files)

        display_eda_images(
            featured_graphs,
            "No featured EDA graphs were found. Check your saved graph folders.",
            limit=6
        )

    with tab2:
        st.subheader("Bivariate Analysis")
        st.write(
            "Bivariate analysis examines the relationship between two "
            "variables, helping identify how individual agricultural "
            "factors relate to crop yield."
        )

        bivariate_images = get_eda_images(
            eda_root / "Bivariate"
        )

        display_eda_images(
            bivariate_images,
            "No saved bivariate graphs found in EDA_Graphs/Bivariate."
        )

    with tab3:
        st.subheader("Multivariate Analysis")
        st.write(
            "Multivariate analysis explores patterns involving multiple "
            "agricultural and environmental variables simultaneously."
        )

        multivariate_images = get_eda_images(
            eda_root / "Multivariate"
        )

        display_eda_images(
            multivariate_images,
            "No saved multivariate graphs found in EDA_Graphs/Multivariate."
        )

    with tab4:
        st.subheader("Numerical Correlation Heatmap")
        st.write(
            "The heatmap displays pairwise correlations among numerical "
            "features, including the target where available. Correlation "
            "indicates association, not necessarily causation."
        )

        heatmap_path = (
            eda_root / "Correlation" /
            "numerical_correlation_heatmap.png"
        )

        if heatmap_path.exists():
            st.image(
                str(heatmap_path),
                caption="Numerical Feature Correlation Heatmap",
                use_container_width=True
            )
        else:
            correlation_images = get_eda_images(
                eda_root / "Correlation"
            )

            display_eda_images(
                correlation_images,
                "The numerical correlation heatmap was not found at "
                "EDA_Graphs/Correlation/numerical_correlation_heatmap.png. "
                "Check the actual filename and folder."
            )


# ------------------------------------------------------------
# 11. INTERACTIVE CROP-YIELD PREDICTION
# ------------------------------------------------------------

elif page == "Crop Yield Prediction":

    st.header("🌱 Interactive Crop Yield Prediction")

    st.write(
        "Enter the soil, environmental, and farming information "
        "to estimate crop yield in tons per hectare."
    )

    with st.form("crop_prediction_form"):

        st.subheader("1. Soil Nutrients")

        col1, col2, col3 = st.columns(3)

        with col1:
            N = st.number_input(
                "Nitrogen (N)", min_value=0.0, value=132.0
            )
            P = st.number_input(
                "Phosphorus (P)", min_value=0.0, value=62.0
            )

        with col2:
            K = st.number_input(
                "Potassium (K)", min_value=0.0, value=22.0
            )
            Soil_pH = st.number_input(
                "Soil pH",
                min_value=0.0,
                max_value=14.0,
                value=6.35
            )

        with col3:
            Soil_Moisture = st.number_input(
                "Soil Moisture", min_value=0.0, value=59.78
            )
            Organic_Carbon = st.number_input(
                "Organic Carbon", min_value=0.0, value=1.2
            )

        st.subheader("2. Environmental Conditions")

        col1, col2, col3 = st.columns(3)

        with col1:
            Temperature = st.number_input(
                "Temperature (°C)", value=27.5
            )
            Humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=53.89
            )

        with col2:
            Rainfall = st.number_input(
                "Rainfall (mm)", min_value=0.0, value=1305.68
            )
            Sunlight_Hours = st.number_input(
                "Sunlight Hours", min_value=0.0, value=7.73
            )

        with col3:
            Wind_Speed = st.number_input(
                "Wind Speed (km/h)", min_value=0.0, value=10.5
            )
            Altitude = st.number_input(
                "Altitude (m)", value=250.0
            )

        st.subheader("3. Farming Practices")

        col1, col2 = st.columns(2)

        with col1:
            Fertilizer_Used = st.number_input(
                "Fertilizer Used (kg/ha)", min_value=0.0, value=223.48
            )
            Pesticide_Used = st.number_input(
                "Pesticide Used (kg/ha)", min_value=0.0, value=2.5
            )

            Soil_Type = st.selectbox(
                "Soil Type",
                ["Clay", "Loamy", "Sandy", "Silt"]
            )

            Region = st.selectbox(
                "Region",
                ["Central", "East", "North", "South", "West"]
            )

        with col2:
            Season = st.selectbox(
                "Season",
                ["Kharif", "Rabi", "Zaid"]
            )

            Crop_Type = st.selectbox(
                "Crop Type",
                [
                    "Cotton", "Maize", "Potato",
                    "Rice", "Sugarcane", "Wheat"
                ]
            )

            Irrigation_Type = st.selectbox(
                "Irrigation Type",
                ["Canal", "Drip", "Rainfed", "Sprinkler"]
            )

        submitted = st.form_submit_button(
            "🌾 Predict Crop Yield",
            use_container_width=True
        )

    if submitted:

        input_data = {
            "N": N,
            "P": P,
            "K": K,
            "Soil_pH": Soil_pH,
            "Soil_Moisture": Soil_Moisture,
            "Soil_Type": Soil_Type,
            "Organic_Carbon": Organic_Carbon,
            "Temperature": Temperature,
            "Humidity": Humidity,
            "Rainfall": Rainfall,
            "Sunlight_Hours": Sunlight_Hours,
            "Wind_Speed": Wind_Speed,
            "Region": Region,
            "Altitude": Altitude,
            "Season": Season,
            "Crop_Type": Crop_Type,
            "Irrigation_Type": Irrigation_Type,
            "Fertilizer_Used": Fertilizer_Used,
            "Pesticide_Used": Pesticide_Used
        }

        try:
            if P + K == 0 or N + P + K == 0:
                st.error(
                    "P + K and N + P + K must be greater than zero."
                )

            else:
                with st.spinner("Predicting crop yield..."):
                    predicted_yield = predict_crop_yield_dashboard(
                        input_data
                    )

                st.success("Prediction completed successfully!")

                st.subheader("🌾 Prediction Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="Estimated Crop Yield",
                        value=f"{predicted_yield:.3f} tons/hectare"
                    )

                with col2:
                    st.metric(
                        label="Selected Crop",
                        value=Crop_Type
                    )

                st.info(
                    "This is a model-based estimate, not a guarantee "
                    "of actual agricultural yield."
                )

                with st.expander("View Submitted Input Details"):
                    input_summary_df = pd.DataFrame(
                        list(input_data.items()),
                        columns=["Input Feature", "Submitted Value"]
                    )

                    st.dataframe(
                        input_summary_df,
                        use_container_width=True,
                        hide_index=True
                    )

                report_data = input_data.copy()

                report_data["Predicted_Crop_Yield_ton_per_hectare"] = (
                    predicted_yield
                )

                report_df = pd.DataFrame([report_data])
                csv_data = report_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="📥 Download Prediction Report (CSV)",
                    data=csv_data,
                    file_name="crop_yield_prediction_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ------------------------------------------------------------
# 12. MODEL EVALUATION
# ------------------------------------------------------------

elif page == "Model Evaluation":

    st.title("📊 Model Evaluation & Comparison")

    st.write(
        "Compare the performance of all 10 trained regression models "
        "using MAE, RMSE, and R² Score."
    )

    evaluation_df = pd.DataFrame([
        {
            "Model": "Linear Regression",
            "MAE": 0.816061,
            "RMSE": 1.020534,
            "R² Score": 0.998213
        },
        {
            "Model": "Decision Tree",
            "MAE": 1.325965,
            "RMSE": 1.657986,
            "R² Score": 0.995284
        },
        {
            "Model": "Random Forest",
            "MAE": 0.898155,
            "RMSE": 1.124261,
            "R² Score": 0.997832
        },
        {
            "Model": "SVR",
            "MAE": 2.460433,
            "RMSE": 4.626130,
            "R² Score": 0.963284
        },
        {
            "Model": "K-Nearest Neighbors",
            "MAE": 15.099617,
            "RMSE": 21.326306,
            "R² Score": 0.219714
        },
        {
            "Model": "Gradient Boosting",
            "MAE": 0.845899,
            "RMSE": 1.056437,
            "R² Score": 0.998085
        },
        {
            "Model": "HistGradientBoosting",
            "MAE": 0.830327,
            "RMSE": 1.044314,
            "R² Score": 0.998129
        },
        {
            "Model": "CatBoost",
            "MAE": 0.833454,
            "RMSE": 1.046120,
            "R² Score": 0.998122
        },
        {
            "Model": "Extra Trees",
            "MAE": 0.900441,
            "RMSE": 1.122438,
            "R² Score": 0.997839
        },
        {
            "Model": "Stacking Regressor",
            "MAE": 0.814537,
            "RMSE": 1.018841,
            "R² Score": 0.998219
        }
    ])

    evaluation_df = evaluation_df.sort_values(
        by="RMSE", ascending=True
    ).reset_index(drop=True)

    st.subheader("1. Overall Model Comparison")

    formatted_eval_df = evaluation_df.copy()
    formatted_eval_df["MAE"] = formatted_eval_df["MAE"].apply(lambda x: f"{x:.6f}")
    formatted_eval_df["RMSE"] = formatted_eval_df["RMSE"].apply(lambda x: f"{x:.6f}")
    formatted_eval_df["R² Score"] = formatted_eval_df["R² Score"].apply(lambda x: f"{x:.6f}")

    st.dataframe(
        formatted_eval_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("2. RMSE Comparison")

    st.caption(
        "Lower RMSE indicates lower prediction error, with larger errors "
        "penalized more strongly."
    )

    st.bar_chart(
        evaluation_df.set_index("Model")[["RMSE"]]
    )

    st.subheader("3. MAE Comparison")

    st.caption(
        "MAE represents the average absolute prediction error."
    )

    st.bar_chart(
        evaluation_df.set_index("Model")[["MAE"]]
    )

    st.subheader("4. R² Score Comparison")

    st.caption(
        "R² closer to 1 indicates a better fit on the evaluated test set."
    )

    st.bar_chart(
        evaluation_df.set_index("Model")[["R² Score"]]
    )

    st.subheader("5. Key Findings")

    best_model = evaluation_df.iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Best Observed Model", best_model["Model"])
    col2.metric("Lowest RMSE", f'{best_model["RMSE"]:.4f}')
    col3.metric("Best Observed R²", f'{best_model["R² Score"]:.4f}')

    st.success(
        "The Stacking Regressor achieved the lowest observed MAE and RMSE "
        "and the highest observed R² among the evaluated models."
    )

    st.info(
        "Linear Regression performs almost as well as the Stacking "
        "Regressor. The observed difference is marginal and does not "
        "establish that stacking is conclusively superior."
    )

    st.subheader("6. Metric Interpretation")

    st.markdown("""
    - **MAE:** Average absolute prediction error in tons/hectare.
    - **RMSE:** Penalizes larger prediction errors more strongly.
    - **R² Score:** Indicates how well predictions explain variation
      in crop yield; values closer to 1 are generally better.
    """)


# ------------------------------------------------------------
# 13. DATASET INSIGHTS
# ------------------------------------------------------------

elif page == "Dataset Insights":

    st.title("🌱 Dataset Insights & Exploration")

    st.write(
        "Explore the original crop-yield dataset using interactive "
        "tables, summary statistics, and visualizations."
    )

    if not DATASET_PATH.is_file():

        st.error(f"Dataset not found: {DATASET_PATH}")

        st.info(
            "Make sure crop-yield.csv is inside the project's datasets folder."
        )

    else:

        insights_df = load_dataset()
        target_column = "Crop_Yield_ton_per_hectare"

        if insights_df is None:

            st.error("Unable to load the crop-yield dataset.")

        elif target_column not in insights_df.columns:

            st.error(
                f"Target column '{target_column}' was not found in the dataset."
            )

        else:

            # 13.1 DATASET OVERVIEW

            st.subheader("1. Dataset Overview")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Records",
                f"{insights_df.shape[0]:,}"
            )

            col2.metric(
                "Total Columns",
                insights_df.shape[1]
            )

            col3.metric(
                "Input Features",
                insights_df.shape[1] - 1
            )

            col4.metric(
                "Missing Values",
                f"{int(insights_df.isnull().sum().sum()):,}"
            )

            # 13.2 DATASET PREVIEW

            st.subheader("2. Preview the Dataset")

            preview_rows = st.slider(
                "Number of rows to display",
                min_value=5,
                max_value=50,
                value=10,
                step=5
            )

            st.dataframe(
                insights_df.head(preview_rows),
                use_container_width=True,
                hide_index=True
            )

            # 13.3 DATASET STRUCTURE

            st.subheader("3. Dataset Structure")

            structure_df = pd.DataFrame({
                "Column": insights_df.columns,
                "Data Type": [
                    str(dtype) for dtype in insights_df.dtypes
                ],
                "Missing Values": [
                    int(value)
                    for value in insights_df.isnull().sum()
                ],
                "Unique Values": [
                    int(insights_df[column].nunique(dropna=True))
                    for column in insights_df.columns
                ]
            })

            st.dataframe(
                structure_df,
                use_container_width=True,
                hide_index=True
            )

            # 13.4 CROP-YIELD DISTRIBUTION

            st.subheader("4. Crop Yield Distribution")

            st.caption(
                "Explore the distribution of the target variable "
                "across the dataset."
            )

            yield_values = pd.to_numeric(
                insights_df[target_column],
                errors="coerce"
            ).dropna()

            if yield_values.empty:

                st.warning(
                    "No valid numeric crop-yield values are available."
                )

            else:

                st.write("Crop yield summary statistics:")

                summary_df = yield_values.describe().to_frame(
                    name="Crop Yield (tons/hectare)"
                ).round(3)

                st.dataframe(
                    summary_df,
                    use_container_width=True
                )

                st.write("Crop yield frequency distribution:")

                histogram_counts, histogram_edges = np.histogram(
                    yield_values.to_numpy(dtype=float),
                    bins=20
                )

                histogram_df = pd.DataFrame({
                    "Yield Range": [
                        (
                            f"{histogram_edges[i]:.2f} – "
                            f"{histogram_edges[i + 1]:.2f}"
                        )
                        for i in range(len(histogram_counts))
                    ],
                    "Number of Records": histogram_counts
                })

                st.bar_chart(
                    histogram_df.set_index("Yield Range")
                )

            # 13.5 CATEGORICAL FEATURE EXPLORATION

            st.subheader("5. Explore Categorical Features")

            categorical_columns = insights_df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()

            if categorical_columns:

                selected_category = st.selectbox(
                    "Choose a categorical feature",
                    categorical_columns
                )

                category_counts = (
                    insights_df[selected_category]
                    .fillna("Missing")
                    .astype(str)
                    .value_counts()
                )

                category_chart_df = category_counts.rename(
                    "Number of Records"
                ).rename_axis("Category").to_frame()

                st.bar_chart(category_chart_df)

                st.write(
                    f"Distribution of **{selected_category}**"
                )

            else:

                st.info("No categorical columns were detected.")

            # 13.6 CORRELATION WITH CROP YIELD

            st.subheader("6. Correlation with Crop Yield")

            numeric_df = insights_df.select_dtypes(
                include=["number"]
            )

            if target_column in numeric_df.columns:

                correlations = (
                    numeric_df.corr()[target_column]
                    .drop(target_column)
                    .dropna()
                    .sort_values()
                )

                if not correlations.empty:

                    st.caption(
                        "Correlation measures linear association. It does "
                        "not prove that a feature causes higher or lower yield."
                    )

                    correlation_df = correlations.rename(
                        "Correlation with Crop Yield"
                    ).rename_axis("Feature").to_frame()

                    st.bar_chart(correlation_df)

                else:

                    st.info(
                        "There are not enough numeric features to calculate "
                        "correlations."
                    )

            # 13.7 MISSING-VALUE ANALYSIS

            st.subheader("7. Missing-Value Analysis")

            missing_df = (
                insights_df.isnull()
                .sum()
                .rename("Missing Values")
                .to_frame()
            )

            missing_df["Missing Percentage (%)"] = (
                missing_df["Missing Values"]
                / len(insights_df)
                * 100
            )

            missing_df = missing_df.sort_values(
                "Missing Values",
                ascending=False
            )

            formatted_missing_df = missing_df.copy()
            formatted_missing_df["Missing Values"] = formatted_missing_df["Missing Values"].apply(lambda x: f"{x:,.0f}")
            formatted_missing_df["Missing Percentage (%)"] = formatted_missing_df["Missing Percentage (%)"].apply(lambda x: f"{x:.2f}%")

            st.dataframe(
                formatted_missing_df,
                use_container_width=True
            )

            st.success(
                "Dataset exploration is ready! Use these insights to "
                "understand the data before interpreting model predictions."
            )


# ------------------------------------------------------------
# 14. DASHBOARD FOOTER
# ------------------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align: center; padding: 12px;">
        <h4>🌾 Crop Yield Prediction</h4>
        <p>
            A Machine Learning Approach for Crop Yield Prediction
            Using Agricultural and Environmental Factors
        </p>
        <p style="font-size: 13px; color: gray;">
            Academic project • Data-driven crop yield estimation
        </p>
        <p style="font-size: 12px; color: gray;">
            Predictions are estimates based on the dataset and trained model.
            Actual crop yields may vary with real-world conditions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)