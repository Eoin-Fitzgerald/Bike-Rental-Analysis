# Bike Rental Data Analysis  
## Leveraging Machine Learning & Statistical Techniques for Operational Insights

### Summary
This project analyzes customer and ride data from a global bike rental company operating in cities like New York, Berlin, and London. The objective is to uncover insights that can inform business decisions around pricing, customer satisfaction, and fleet optimization. The work integrates **data preprocessing**, **exploratory data analysis**, **statistical inference**, and **machine learning** to build predictive models and draw strategic conclusions.

---

### Objectives
- Clean and preprocess real-world rental data for analysis.
- Uncover trends in ride duration, rental costs, and customer satisfaction.
- Perform hypothesis testing and linear regression to validate patterns.
- Use ML models (regression & classification) to predict costs and electric bike usage.

---

### Methodology

#### 1. **Data Preparation**
- 900 raw records reduced to 763 after cleaning and outlier removal.
- Handled missing values via mean/mode imputation.
- Feature engineering: interaction terms, normalization, encoding.

#### 2. **Exploratory Data Analysis (EDA)**
- Key correlation: `ride_duration` highly predicts `bike_rental_cost` (r = 0.90).
- Dublin showed higher costs, likely due to small sample size.
- Rider satisfaction not strongly tied to ride duration or weather.

#### 3. **Statistical Analysis**
- T-tests, ANOVA, and Chi-Square used to test differences in costs/satisfaction.
- Confidence intervals and Shapiro-Wilk tests validated assumptions.
- Simple Linear Regression revealed that:
  - Every extra minute of ride increases rental cost by **$0.25**
  - R² = 0.82 (strong explanatory power)

#### 4. **Machine Learning**
- **Regression models**: Linear Regression, Random Forest, Ridge, Lasso
- **Classification models**: XGBoost, Random Forest, Stacking Classifier
- SMOTE applied for class balance; GridSearchCV for tuning
- Best results:
  - **Regression**: Linear Regression (MAE = 2.02)
  - **Classification**: Stacking Classifier (Accuracy = 0.81, ROC-AUC = 0.89)

---

### Key Insights
- **Ride duration** is the strongest driver of rental cost.
- **Electric bikes** cost significantly more than standard models.
- **Advanced ML models** like XGBoost and stacking improved performance on imbalanced classification tasks.
- **Dimensionality reduction** (PCA, LDA) aided in visualization and model performance.

---

### Tools & Technologies
- **Python**: pandas, numpy, seaborn, matplotlib, scikit-learn, xgboost
- **EDA & Stats**: Boxplots, Histograms, Confidence Intervals, Hypothesis Tests
- **ML Techniques**: GridSearchCV, SMOTE, Feature Engineering, PCA, LDA
