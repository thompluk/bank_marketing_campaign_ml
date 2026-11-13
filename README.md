# Bank Marketing Campaign: Term Deposit Prediction

## 1. Project Overview
This project analyzes bank telemarketing data to extract insights, improve decision-making, and identify key conversion trends. The primary focus is to **optimize telemarketing operational efficiency and maximize term deposit conversions** by moving away from randomized blanket marketing to a data-driven, targeted approach.

Due to limited operational resources (e.g., daily call quotas), the bank needs an intelligent filtering system so agents can focus their efforts on high-probability prospects.

**Key Objectives:**
- **Objective 1:** Develop a robust Machine Learning model (Gradient Boosting) to predict the probability of a customer subscribing to a term deposit.
- **Objective 2:** Implement a **Top-N Lead Prioritization** strategy to rank unseen prospects and optimize limited daily call quotas (e.g., 500 calls/day).
- **Objective 3:** Prove financial efficiency and incremental Net Profit through a cost-benefit business simulation on unseen data.

## 2. Data Sources
- **Bank Marketing Dataset:** Contains historical data from direct marketing campaigns (phone calls) of a banking institution.
  - **Customer Profile:** Demographic and financial indicators (`age`, `job`, `balance`, `housing`, `loan`).
  - **Marketing Data:** Campaign interaction history (`contact`, `month`, `campaign`, `pdays`, `poutcome`).
  - **Target Variable:** `deposit` (Whether the customer deposits or not: 'yes' or 'no').

## 3. Technologies Used
- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Pipelines, ColumnTransformer, TargetEncoder), Gradient Boosting Classifier
- **Hyperparameter Tuning:** Optuna
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Model Deployment:** Streamlit (Batch Scoring Application), Joblib
- **Version Control:** Git

## 4. Project Structure

```text
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── model              <- Trained and serialized models (e.g., `model_gboost_v1.sav`).
│
├── notebooks          <- Jupyter notebooks for EDA, Feature Engineering, and Modeling.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment.
│
└── src                <- Source code for use in this project (e.g., `app_gboost_batch.py`).
```

## 5. Summary of Findings

### 5.1 Business Insights
- **The "Game Changer" Feature (`poutcome = success`):** Customers who successfully deposited in the previous campaign have a massive conversion rate (>86% - 93%) in the current campaign, regardless of their housing or loan status.
- **High-Risk Communication (`contact = unknown`):** Reaching out via 'cellular' yields high response rates (up to 80% for students). Conversely, using an 'unknown' contact method plummets the conversion rate of highly potential segments down to ~11%.
- **Debt-Free & Liquid Segments:** Customers without housing or personal loans (`housing_no` & `loan_no`), particularly Students, Retirees, and the Unemployed, show significantly higher deposit rates (60% - 79%) due to better financial liquidity.
- **Seasonality Effect:** December, March, October, and September are the "Golden Months" for term deposits, showing consistent spikes in conversion.

### 5.2 Actionable Recommendations
- **Adopt Tiered Lead Prioritization:** Integrate the model's batch scoring results directly into the CRM. Agents must prioritize "Tier 1 (Hot Leads)" and "Tier 2" in the morning to maximize conversion, and skip/hold "Tier 3" to prevent contact fatigue and save costs.
- **Personalize Telemarketing Scripts:** Utilize segment-based insights. For example, offer long-term investment education to the Retiree segment with stable balances, as they have a historically high probability of depositing.
- **Improve CRM Data Hygiene:** Implement strict SOPs for data entry to minimize `unknown` values in critical fields like `job` and `contact`. Better data quality will drastically improve the model's scoring fairness and accuracy.
- **Cost-Sensitive Threshold Tuning:** Always evaluate the model's decision threshold based on actual business costs (Cost Per Call vs. Profit Per Deposit) rather than default statistical metrics (0.5 threshold) to maximize ROI.

## 6. Contact
- **Name:** Thomas Dany H
- **Email:** thomasdany38@gmail.com 
- **LinkedIn:** www.linkedin.com/in/thomas-dany-haryanto
