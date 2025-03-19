# California Housing Price Prediction

This project aims to predict house prices in California using the California Housing dataset. Various regression models and regularization techniques have been explored to achieve the best performance.

## Dataset
- **Source:** Loaded from `sklearn.datasets.fetch_california_housing`.
- **Features:** 8 independent variables (e.g., `MedInc`, `AveRooms`, `HouseAge`) and 1 target variable (`MedHouseVal` - house value in units of $100,000).
- **Number of Samples:** 20,640

## Project Steps
### 1. Exploratory Data Analysis (EDA)
- **Scatter Plots:** Relationships between features and the target were visualized. `MedInc` showed a strong linear correlation, while some features (e.g., `AveRooms`) suggested non-linear patterns.
- **Distribution:** Most features were left-skewed with some outliers observed.
- **Correlation:** The correlation matrix revealed a high correlation (~0.7) between `MedInc` and the target, with multicollinearity present between some features (e.g., `AveRooms` and `AveBedrms`).

### 2. Data Preprocessing
- **Standardization:** Features were standardized using `StandardScaler` (mean=0, std=1).
- **Polynomial Features:** Added degree-2 polynomial terms with `PolynomialFeatures(degree=2)`, increasing the feature count from 8 to 44.
- **Data Split:** 80% for training (`X_train`, `Y_train`) and 20% for testing (`X_test`, `Y_test`) with `random_state=42`.

### 3. Models Used
#### Simple Linear Regression
- **MSE:** 0.56
- **R²:** 0.58
- **Analysis:** A basic model, but insufficient due to non-linear relationships in the data.

#### Polynomial Regression (Degree 2)
- **MSE:** 0.43
- **R²:** 0.67
- **Analysis:** Adding degree-2 terms improved performance by capturing non-linear patterns.

#### Regularization
Regularized models (`Ridge`, `Lasso`, `ElasticNet`) were used to optimize performance:
- **Ridge (L2):**
  - Best \( \alpha = 1.0 \)
  - MSE: 0.43, R²: 0.67
  - Analysis: Controlled coefficients and improved model stability.
- **Lasso (L1):**
  - Best \( \alpha = 0.001 \)
  - MSE: 0.42, R²: 0.68
  - Analysis: Performed feature selection by setting some coefficients to zero.
- **ElasticNet (L1 + L2):**
  - Best \( \alpha = 0.01, l1_ratio = 0.9 \)
  - MSE: 0.42, R²: 0.68
  - Analysis: Balanced Ridge and Lasso, achieving the best performance with proper tuning.

### 4. Hyperparameter Tuning
- **GridSearchCV:**
  - Tested all parameter combinations (e.g., \( \alpha \), \( l1_ratio \)).
  - Time-intensive (several minutes for polynomial data).
- **RandomizedSearchCV:**
  - Tested a random subset of combinations (e.g., 5 for Ridge/Lasso, 10 for ElasticNet).
  - Faster (<1 minute) with results close to GridSearchCV.
  - Parameters: \( n_jobs=-1 \) for parallelization, \( cv=3 \) for 3-fold cross-validation.

### 5. Final Results
- **Best Model:** ElasticNet with \( \alpha = 0.01, l1_ratio = 0.9 \) on polynomial features.
- **MSE:** 0.42
- **R²:** 0.68
- **Analysis:** This model struck the best balance between complexity and accuracy, explaining 68% of the variance.

## Tools and Libraries
- **Python:** 3.x
- **Libraries:**
  - `sklearn` for models, preprocessing, and tuning
  - `pandas` for data handling
  - `matplotlib` and `seaborn` for visualization (to be added in future steps)

## Key Code
- **Preprocessing:** `scaler.fit_transform`, `PolynomialFeatures`
- **Modeling:** `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`
- **Tuning:** `GridSearchCV`, `RandomizedSearchCV`
- **Evaluation:** `mean_squared_error`, `r2_score`

## Next Steps
1. Visualize predictions vs. actual values for better insights.
2. Test non-linear models like Random Forest or XGBoost for comparison.
3. Remove outliers or perform additional feature engineering to boost accuracy.

## Installation and Running
1. Clone the repository:
   ```bash
   git clone <repository-url>
