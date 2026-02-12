# Bank Marketing Classification Project

## Overview

This project builds a **binary classification model** to predict whether clients of a Portuguese bank will subscribe to term deposits. By leveraging machine learning, the bank can optimize its marketing campaigns and allocate resources more efficiently.

---

## Table of Contents

1. [Business Problem](#business-problem)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Models](#models)
5. [Results](#results)
6. [Key Findings](#key-findings)
7. [Recommendations](#recommendations)
8. [Installation & Usage](#installation--usage)
9. [Project Structure](#project-structure)

---

## Business Problem

### Stakeholder
A Portuguese bank that conducts direct marketing campaigns via phone calls to potential and existing clients.

### Challenge
Not all clients are interested in term deposit subscriptions. Calling uninterested clients wastes time and money, reducing campaign efficiency.

### Objective
Build a classification model to predict which clients are most likely to subscribe to a term deposit, enabling the bank to:
- Focus marketing efforts on high-probability prospects
- Reduce wasted resources on uninterested clients
- Increase overall subscription rates

### Key Question
*"Which clients are most likely to subscribe to a term deposit so the bank can focus marketing efforts efficiently?"*

---

## Dataset

### Source
UCI Machine Learning Repository – Bank Marketing Dataset

### Size & Scope
- **Records**: ~45,000 clients
- **Features**: 17 client and campaign attributes
- **Target Variable**: `y` (yes/no – whether client subscribed)

### Feature Categories

| Category | Features |
|----------|----------|
| **Client Information** | age, job, marital status, education |
| **Financial Information** | account balance, credit default status, housing loan, personal loan |
| **Campaign Information** | contact type, day, month, contact duration, campaign count, days since previous contact, previous contacts, previous outcome |

### Problem Type
**Binary Classification** – Predict subscription (yes) or no subscription (no)

---

## Methodology

### 1. Data Preparation
- Loaded and explored 45,000+ records with 17 features
- **No missing values** detected
- Separated features (X) and target (y)
- Converted target to binary: 'yes' → 1, 'no' → 0
- Identified numeric and categorical features

### 2. Train-Test Split
- **80% training data** (used for model training)
- **20% test data** (used for evaluation)
- Stratified split to maintain target distribution
- Random state = 42 for reproducibility

### 3. Preprocessing Pipeline
- **Numeric Features**: StandardScaler (scaling for fair comparison)
- **Categorical Features**: OneHotEncoder (convert categories to numeric)
- **Integration**: ColumnTransformer to apply different transformations simultaneously

### 4. Modeling Strategy
Two models were trained and compared:
1. **Logistic Regression** (baseline)
2. **Decision Tree** (with hyperparameter tuning)

---

## Models

### Model 1: Logistic Regression (Baseline)

**Configuration**:
- Pipeline: Preprocessing → Logistic Regression
- Max iterations: 1,000
- Purpose: Simple, interpretable baseline

**Why Logistic Regression?**
- Easy to interpret and explain to stakeholders
- Provides clear feature coefficients
- Fast training and prediction
- Baseline for comparison

---

### Model 2: Decision Tree (Optimized)

**Configuration**:
- Initial max_depth: 5
- **Hyperparameter Tuning** via GridSearchCV:
  - `max_depth`: [3, 5, 7, 9]
  - `min_samples_split`: [2, 5, 10]
- Cross-validation: 5-fold
- Scoring metric: F1 Score

**Why Decision Tree?**
- Better captures non-linear relationships
- Provides feature importance rankings
- More intuitive for business decision-making
- Improved recall for identifying potential subscribers

---

## Results

### Baseline Logistic Regression Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 91.6% |
| **Precision** | 0.71 (71%) |
| **Recall** | 0.44 (44%) |
| **F1 Score** | 0.54 |

### Tuned Decision Tree Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | Improved |
| **Precision** | Maintained |
| **Recall** | **Significantly Improved** |
| **F1 Score** | **Higher** |

### Model Comparison

| Aspect | Logistic Regression | Decision Tree |
|--------|-------------------|---------------|
| **Accuracy** | 91.6% | Comparable |
| **Recall** | 44% | Higher ✓ |
| **Interpretability** | Good | Excellent ✓ |
| **Feature Importance** | Limited | Detailed ✓ |
| **Business Impact** | Misses subscribers | Finds more subscribers ✓ |

---

## Key Findings

### Top Influential Features

The tuned Decision Tree identified the following features as most important for predicting subscription:

1. **Number of Contacts in Current Campaign** – Fewer, quality contacts are more effective
2. **Previous Campaign Outcome** – Past success strongly indicates future subscription
3. **Account Balance** – Higher balances correlate with subscription likelihood
4. **Client Age** – Certain age groups respond better to marketing
5. **Last Contact Duration** – Longer conversations indicate stronger interest
6. Other important features: contact type, day, education, employment status

### Class Imbalance Insight

- The dataset is **imbalanced**: Most clients did not subscribe
- High accuracy (91.6%) can be misleading without considering precision/recall
- **Recall is critical** – We must identify actual subscribers, not just avoid false positives

### Interpretation of Baseline Model

| Metric | Interpretation | Implication |
|--------|---|---|
| **Accuracy (91.6%)** | High but misleading due to class imbalance | Don't rely solely on accuracy |
| **Precision (71%)** | 7 out of 10 predicted "yes" are correct | Good targeting efficiency |
| **Recall (44%)** | Only catches 44% of true subscribers | **Misses half the opportunities** |
| **F1 Score (0.54)** | Moderate balance of precision/recall | Room for improvement |

---

## Recommendations

### For the Marketing Team

1. **Target High-Probability Clients First**
   - Use the model to rank clients by subscription probability
   - Focus campaigns on top-tier prospects
   - Reduce outreach to low-probability clients

2. **Optimize Contact Strategy**
   - Quality over quantity: Fewer but meaningful contacts are more effective
   - Avoid excessive follow-ups (diminishing returns observed)
   - Personalize based on client characteristics

3. **Leverage Key Client Attributes**
   - Prioritize clients with higher account balances
   - Target age groups that show higher engagement
   - Focus on clients with previous positive campaign outcomes
   - Consider client education and employment status

4. **Respect Client Preferences**
   - Avoid contacting clients already subscribed or uninterested
   - Use previous outcomes to refine targeting criteria
   - Implement opt-out preferences

### For the Data Science Team

1. **Update Models Regularly**
   - Client behavior changes over time
   - Retrain quarterly or semi-annually
   - Monitor for model drift and performance degradation

2. **Expand Data Collection**
   - Consider additional features if available (customer lifetime value, social factors, etc.)
   - Track campaign outcomes for model validation

3. **A/B Testing**
   - Test model predictions against traditional targeting
   - Measure actual subscription rates in real campaigns
   - Gather feedback for continuous improvement

### Bottom Line

Using this model, the bank can:
- **Reach more potential subscribers** with fewer resources
- **Understand why clients subscribe** (through feature importance)
- **Make marketing smarter and more cost-effective**
- **Scale campaigns** with confidence in predictions

---

## Installation & Usage

### Requirements

```
pandas
scikit-learn
seaborn
matplotlib
```

### Steps to Run

1. **Clone or download the project files**

2. **Install required packages** (if not already installed):
   ```bash
   pip install pandas scikit-learn seaborn matplotlib
   ```

3. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook BankMarketing_Phase3.ipynb
   ```

4. **Ensure the data file is in the same directory**:
   - `bank-additional-full.csv` (provided)

5. **Run all cells** to:
   - Load and explore data
   - Train baseline model
   - Train and tune decision tree
   - Generate evaluation metrics and visualizations
   - View feature importance rankings

### Expected Output

- Data exploration summaries
- Model performance metrics
- Confusion matrices
- Feature importance plots
- Recommendations for marketing strategy

---

## Project Structure

```
Phase 3 project/
├── BankMarketing_Phase3.ipynb          # Main analysis notebook
├── bank-additional-full.csv            # Dataset (~45,000 records, 17 features)
└── README.md                           # This file
```

---

## Technical Details

### Libraries & Tools

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning models, preprocessing, evaluation
- **matplotlib & seaborn**: Data visualization
- **sklearn.compose.ColumnTransformer**: Multi-type preprocessing
- **sklearn.pipeline.Pipeline**: Model-preprocessing integration
- **sklearn.model_selection.GridSearchCV**: Hyperparameter tuning

### Preprocessing Pipeline

```
Raw Data
    ↓
Train-Test Split (80/20)
    ↓
ColumnTransformer:
  ├─ Numeric → StandardScaler
  └─ Categorical → OneHotEncoder
    ↓
Model Training & Evaluation
```

---

## Conclusion

This project demonstrates how machine learning can transform banking marketing strategy. By predicting client subscription probability, the bank can:

✓ **Increase effectiveness** – Focus on high-probability clients  
✓ **Save resources** – Reduce wasted calls on uninterested clients  
✓ **Gain insights** – Understand what drives subscription decisions  
✓ **Scale operations** – Apply learnings to future campaigns  
✓ **Measure impact** – Track ROI against traditional approaches  

The tuned Decision Tree model provides an excellent balance of accuracy, interpretability, and business value, making it the recommended solution for deployment.

---

## Contact & Questions

This project was completed as Phase 3 of a machine learning initiative. For questions or further analysis, refer to the notebook documentation and comments throughout the analysis.

---

**Last Updated**: February 2026  
**Dataset Source**: UCI Machine Learning Repository – Bank Marketing Dataset
