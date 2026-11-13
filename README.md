# 🌱 A Machine Learning Approach for Crop Yield Prediction Using Agricultural and Environmental Factors

## 📌 Project Overview

Agriculture plays a vital role in food security and economic development. Crop yield is influenced by several factors, including soil properties, nutrient availability, weather conditions, crop type, and agricultural practices.

This project develops a **machine learning-based crop yield prediction system** that uses agricultural and environmental factors to estimate crop yield in tons per hectare.

The project follows an end-to-end machine learning workflow, covering data preprocessing, exploratory data analysis (EDA), feature engineering, feature selection, model development, performance evaluation, and interactive prediction through a Streamlit dashboard.

A **Stacking Regressor** was developed and evaluated alongside other regression algorithms to identify the best-performing model.

## 🎯 Project Objectives

- Analyze agricultural and environmental factors affecting crop yield.
- Perform data preprocessing and exploratory data analysis.
- Apply feature engineering to derive meaningful agricultural features.
- Evaluate feature selection techniques.
- Develop and compare multiple machine learning regression models.
- Identify the best-performing model using appropriate evaluation metrics.
- Build an interactive dashboard for crop yield prediction.
- Generate downloadable prediction reports.

## 📂 Dataset Description

The project uses a crop yield dataset obtained from Kaggle.

| Dataset Property | Description |
|---|---|
| Dataset File | `crop-yield.csv` |
| Total Records | 10,000 |
| Total Columns | 20 |
| Input Features | 19 |
| Target Variable | `Crop_Yield_ton_per_hectare` |
| Problem Type | Supervised Machine Learning |
| Learning Task | Regression |

### Input Features

The dataset contains the following agricultural and environmental attributes:

**Soil and Nutrient Factors**
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Soil pH
- Soil Moisture
- Soil Type
- Organic Carbon

**Environmental Factors**
- Temperature
- Humidity
- Rainfall
- Sunlight Hours
- Wind Speed
- Altitude

**Agricultural and Crop Management Factors**
- Region
- Season
- Crop Type
- Irrigation Type
- Fertilizer Used
- Pesticide Used

**Target Variable:** `Crop_Yield_ton_per_hectare`

The target variable represents crop yield in tons per hectare.

## ⚙️ Methodology

The project follows a structured machine learning pipeline.

### 1. Data Preprocessing

- Checked the dataset structure and data types.
- Handled missing values using appropriate imputation techniques.
- Applied median imputation to numerical features.
- Applied most-frequent imputation to categorical features.
- Used StandardScaler for numerical feature scaling.
- Applied OneHotEncoder for categorical feature encoding.
- Used a ColumnTransformer to combine preprocessing operations.
- Applied preprocessing within the training workflow to prevent data leakage.

### 2. Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset and identify patterns and relationships.

Major analysis activities include:

- Target variable distribution analysis.
- Numerical feature correlation analysis.
- Univariate and bivariate analysis.
- Multivariate analysis.
- Analysis of agricultural and environmental factors.

### 3. Feature Engineering

Six additional features were created to represent relationships between agricultural attributes:

| Engineered Feature | Description |
|---|---|
| `NPK_Total` | Combined nitrogen, phosphorus, and potassium values |
| `N_to_PK_Ratio` | Nitrogen relative to phosphorus and potassium |
| `Soil_pH_Deviation` | Absolute deviation of soil pH from 7 |
| `Moisture_Humidity_Interaction` | Interaction between soil moisture and humidity |
| `Rainfall_Sunlight_Interaction` | Interaction between rainfall and sunlight hours |
| `Fertilizer_to_NPK_Ratio` | Fertilizer usage relative to total NPK |

### 4. Feature Selection

The following feature selection techniques were explored:

- SelectKBest
- Recursive Feature Elimination (RFE)

The final transformed feature representation contained **42 features**, which were used in the final modeling workflow.

### 5. Model Development

Multiple regression algorithms were implemented and evaluated.

The models included:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Support Vector Regression (SVR)
5. K-Nearest Neighbors Regressor
6. Gradient Boosting Regressor
7. HistGradient Boosting Regressor
8. XGBoost Regressor
9. CatBoost Regressor
10. Extra Trees Regressor
11. Stacking Regressor

XGBoost was explored but could not be included in the final evaluation because of an environment-related import issue.

### 6. Stacking Regressor

The Stacking Regressor combines multiple base estimators to generate a final prediction through a meta-learner.

**Base Models:**
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

**Meta-Learner:**
- Ridge Regression

**Configuration:**
- Random Forest: 200 estimators
- Gradient Boosting: 200 estimators
- Learning rate: 0.05
- Gradient Boosting maximum depth: 3
- Ridge regularization parameter: 1.0
- Cross-validation: 5-fold KFold
- Shuffling enabled with random state 42

## 📊 Model Evaluation

The models were evaluated using the following regression metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted values.
- **Mean Squared Error (MSE):** Measures the average squared prediction error.
- **Root Mean Squared Error (RMSE):** Measures prediction error in the original target unit.
- **R² Score:** Measures the proportion of variance in the target variable explained by the model.

### Best Model Results

The Stacking Regressor achieved the following results on the test dataset:

| Evaluation Metric | Result |
|---|---:|
| MAE | 0.814537 |
| MSE | 1.038037 |
| RMSE | 1.018841 |
| R² Score | 0.998219 |

The Stacking Regressor achieved the best observed performance among the evaluated models, with an R² score of approximately **99.82%**.

### Model Comparison

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Stacking Regressor | **0.814537** | **1.018841** | **0.998219** |
| Linear Regression | 0.816061 | 1.020534 | 0.998213 |
| HistGradient Boosting | 0.830327 | 1.044314 | 0.998129 |

*Note: The table presents selected top-performing models from the experimental comparison.*

## 🖥️ Streamlit Dashboard

An interactive web dashboard was developed using **Streamlit** to make the trained machine learning model accessible through a user-friendly interface.

### Dashboard Features

- **Project Overview:** Introduction and overview of the crop yield prediction project.
- **Exploratory Data Analysis:** Interactive visualizations and dataset analysis.
- **Crop Yield Prediction:** Predict crop yield using 19 original input features.
- **Model Evaluation:** View model performance and evaluation metrics.
- **Dataset Insights:** Explore important dataset information.
- **CSV Report Download:** Download prediction results along with the submitted input details.

The dashboard uses the trained model and preprocessing artifacts to generate predictions from user-provided agricultural and environmental inputs.

## 🛠️ Technologies and Tools Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Development Environment | Visual Studio Code |
| Notebook Environment | Jupyter Notebook |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Gradient Boosting | CatBoost |
| Model Serialization | Joblib |
| Web Application | Streamlit |

## 📁 Project Structure

```text
Crop_Yield_Prediction/
│
├── datasets/
│   └── crop-yield.csv
│
├── notebooks/
│   └── Crop_Yield_Prediction.ipynb
│
├── Eda_graphs/
│
├── Feature_Selection/
│
├── Models/
│   ├── stacking_regressor_model.joblib
│   ├── preprocessor.joblib
│   └── final_feature_names.json
│
├── Model_Results/
│
├── app.py
│
└── README.md
```

*Note: The notebook filename and folder contents shown above should be adjusted to match the actual files in your repository.*

## 🚀 Installation and Execution

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/sharmi-pathi/KLH-CSE-2026-27-Team-3-Crop-Yield-Prediction
```

### 2. Navigate to the Project Directory

```bash
cd Crop_Yield_Prediction
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn catboost streamlit joblib
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The dashboard will open in your default web browser.

**Important:** Ensure that the dataset and saved model artifacts are available in their expected project directories before running the application.

## 🔮 Future Enhancements

The project can be extended in several ways:

- Integrate real-time weather and soil information through external APIs.
- Incorporate larger and more diverse agricultural datasets.
- Explore advanced machine learning and deep learning approaches.
- Improve model generalization through additional validation on independent datasets.
- Add explainable AI techniques to understand prediction behavior.
- Develop a mobile-friendly interface for easier accessibility.
- Extend the system to support region-specific crop recommendations.

## 🌾 Applications

The proposed system can support:

- Agricultural planning and decision-making.
- Crop yield estimation.
- Soil and environmental data analysis.
- Precision agriculture research.
- Data-driven agricultural management.
- Academic research and experimentation in machine learning.

## 👩‍💻 Project Team

**Project Title:** A Machine Learning Approach for Crop Yield Prediction Using Agricultural and Environmental Factors

| Student ID | Team Member |
|---|---|
| 2420030042 | Pathi Sharmila |
| 2420030225 | Sanivarapu Sadhana |
| 2420030336 | Maddipatla Pavani Sree |

**Project Guide:** Dr. N. Sunanda, Associate Professor

**Academic Year:** 2026–2027

## 📚 References

1. M. Abdel-Salam, N. Kumar, and S. Mahajan, "A proposed framework for crop yield prediction using hybrid feature selection approach and optimized machine learning," *Neural Computing and Applications*, vol. 36, pp. 20723–20750, 2024.
2. O. S. Nagesh et al., "Boosting enabled efficient machine learning technique for accurate prediction of crop yield towards precision agriculture," *Discover Sustainability*, vol. 5, Art. no. 101, 2024.
3. M. Ashfaq et al., "Accurate wheat yield prediction using machine learning and climate-NDVI data fusion," *IEEE Access*, vol. 12, pp. 40947–40961, 2024.
4. M. A. Jabed and M. A. A. Murad, "Crop yield prediction in agriculture: A comprehensive review of machine learning and deep learning approaches, with insights for future research and sustainability," *Heliyon*, vol. 10, no. 24, e40836, 2024.
5. T. van Klompenburg, A. Kassahun, and C. Catal, "Crop yield prediction using machine learning: A systematic literature review," *Computers and Electronics in Agriculture*, vol. 177, Art. no. 105709, 2020.

---

## ⭐ Acknowledgement

We sincerely thank our project guide, **Dr. N. Sunanda**, for her valuable guidance, encouragement, and support throughout the development of this project.

We also acknowledge the researchers and developers whose work and contributions to machine learning and agricultural data analysis helped us in carrying out this project.

---

**If you find this project interesting, consider giving the repository a ⭐!**
