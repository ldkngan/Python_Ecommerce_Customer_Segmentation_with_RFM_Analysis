# Analyze E-Commerce Customer Segmentation Using RFM Analysis in Python

![customer segment](https://github.com/user-attachments/assets/b113cb2d-67d3-4edd-9b4a-231888640049)

This project focuses on customer segmentation using **RFM analysis** based on e-commerce transactional data. The goal is to transform **raw sales data** into **actionable customer insights** that support marketing, retention, and revenue optimization decisions.
- **Author**: Le Dang Kim Ngan
- **Tool Used**: Python

---

## 📋 Table of Contents
1. Overview
2. Exploratory Data Analysis (EDA)
3. Data Wrangling
4. RFM Score & Segment
5. Visualizations & Key Insights
6. Recommendations

---

## ✨ Overview
### 🎯 Main Objectives
- Segment customers based on **Recency, Frequency, and Monetary value**
- Identify **high-value**, **loyal**, **at-risk**, and **inactive** customer groups
- Understand how customer behavior changes **over time**
- Support business decisions related to **customer retention, re-engagement, and revenue growth**
### 💻 Why RFM Analysis?
- RFM is used because it answers three fundamental business questions using **actual customer behavior**, not assumptions:
    - **Recency** – How recently did the customer purchase?
    - **Frequency** – How often do they purchase?
    - **Monetary** – How much do they spend?
- RFM is particularly suitable for this project because:
    - It is **simple, interpretable, and explainable** to non-technical stakeholders.
    - It works well with **transaction-level data**.
    - It directly links customer behavior to **business value**.
    - It allows easy mapping to **business-friendly segments** (e.g. Champions, Loyal, At Risk, Hibernating).
    - Unlike black-box models, RFM provides **transparent logic**, which is critical for business adoption.
### ❓ Business Questions
- Who are the **most valuable customers** contributing the highest revenue?
- Which customers are **active but low-spending**, and which are **high-spending but infrequent**?
- Which customers are **at risk of churn** due to declining recency or frequency?
- How does **customer value change over time**?
### ⚒️ Tools & Technologies
- **Python** (core language for the entire project)
- pandas, numpy – data manipulation & aggregation
- matplotlib / seaborn – data visualization
- Jupyter / Google Colab – analysis environment
### 📂 Dataset
<details>
  <summary> Table 1: E-Commerce Retail - contains transaction-level data (Click to see detail)</summary>

| Column Name  | Data Type         | Description  |  
|-------------|-----------------|--------------|  
| **InvoiceNo**  | `object`  | Unique invoice number for each transaction (6-digit). If it starts with 'C', it indicates a cancellation. |  
| **StockCode**  | `object`  | Unique product (item) code (5-digit). |  
| **Description**  | `object`  | Product (item) name. |  
| **Quantity**  | `int64`  | The quantities of each product (item) per transaction. |  
| **InvoiceDate**  | `datetime64[ns]`  | Date and time when the transaction was generated. |  
| **UnitPrice**  | `float64`  | Product price per unit in sterling. |  
| **CustomerID**  | `float64`  | Unique 5-digit integral number for each customer. |  
| **Country**  | `object`  | Name of the country where each customer resides. |  

</details>

<details>
  <summary> Table 2: Customer Segmentation & RFM Score - stores customer segments and corresponding RFM scores (Click to see detail)</summary>

| **Segment**              | **RFM Score**  |  
|-------------------------|-----------------------------------------------------------|  
| **Champions**            | 555, 554, 544, 545, 454, 455, 445  |  
| **Loyal**                | 543, 444, 435, 355, 354, 345, 344, 335  |  
| **Potential Loyalist**   | 553, 551, 552, 541, 542, 533, 532, 531, 452, 451, 442, 441, 431, 453, 433, 432, 423, 353, 352, 351, 342, 341, 333, 323  |  
| **New Customers**        | 512, 511, 422, 421, 412, 411, 311  |  
| **Promising**            | 525, 524, 523, 522, 521, 515, 514, 513, 425, 424, 413, 414, 415, 315, 314, 313  |  
| **Need Attention**       | 535, 534, 443, 434, 343, 334, 325, 324  |  
| **About To Sleep**       | 331, 321, 312, 221, 213, 231, 241, 251  |  
| **At Risk**              | 255, 254, 245, 244, 253, 252, 243, 242, 235, 234, 225, 224, 153, 152, 145, 143, 142, 135, 134, 133, 125, 124  |  
| **Cannot Lose Them**     | 155, 154, 144, 214, 215, 115, 114, 113  |  
| **Hibernating Customers** | 332, 322, 233, 232, 223, 222, 132, 123, 122, 212, 211  |  
| **Lost Customers**       | 111, 112, 121, 131, 141, 151   |

</details>

---

## ⚙️ Exploratory Data Analysis (EDA)
This section provides an initial assessment of the retail dataset to understand its structure, data quality, and potential issues that may impact downstream analysis.

In [1]:
```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
In [2]:
```python
# Load data
!pip install gdown
!gdown 14hq8vPUyufScsSSgl7aBzkbw8cSOcphl
df_retail = pd.read_excel('ecommerce retail.xlsx', sheet_name="ecommerce retail")
df_segment = pd.read_excel('ecommerce retail.xlsx', sheet_name="Segmentation")

df_retail.head()
```
Out [2]:

<img width="1026" height="199" alt="image" src="https://github.com/user-attachments/assets/f5254f78-8b81-483d-816c-d01c6a276d55" />


In [3]:
```python
# Check the general info
df_retail.info()
```
Out [3]:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 541909 entries, 0 to 541908
Data columns (total 8 columns):
 #   Column       Non-Null Count   Dtype         
---  ------       --------------   -----         
 0   InvoiceNo    541909 non-null  object        
 1   StockCode    541909 non-null  object        
 2   Description  540455 non-null  object        
 3   Quantity     541909 non-null  int64         
 4   InvoiceDate  541909 non-null  datetime64[ns]
 5   UnitPrice    541909 non-null  float64       
 6   CustomerID   406829 non-null  float64       
 7   Country      541909 non-null  object        
dtypes: datetime64[ns](1), float64(2), int64(1), object(4)
memory usage: 33.1+ MB
```
In [4]:
```python
# Check for missing values
df_retail.isna().sum()
```
Out [4]:

<img width="185" height="324" alt="image" src="https://github.com/user-attachments/assets/79106db9-fd04-428e-a384-5dc3903ef996" />

In [5]:
```python
# Check data summary
df_retail.describe()
```
Out [5]:

<img width="632" height="292" alt="image" src="https://github.com/user-attachments/assets/17a4ec6b-e4cc-4b69-8593-1ed9f0bfed23" />

In [6]:
```python
# Check duplicates
df_retail.duplicated().sum()
```
Out [6]:
```
np.int64(5268)
```

### Summary
- The dataframe (retail) contains 541,909 rows and 8 columns.
- There are 135,080 missing values for CustomerID which is important for RFM analysis.
- There are negative values in Quantity and Unit Price columns, which is not logically valid.
- There are 5268 duplicated values.

---

## 📟 Data Wrangling
This section focuses on **cleaning and transforming** the dataset to ensure **data consistency and reliability** for further analysis. Key steps include converting columns to appropriate data types, removing records with missing CustomerID values, filtering out invalid entries in the Quantity and Unit Price columns, and eliminating duplicated records.

In [7]:
```python
# Adjust data types and remove missing values
column_list = ["InvoiceNo", "StockCode", "Description", "Country"]
for c in column_list:
    df_retail[c] = df_retail[c].astype(str)

df_retail = df_retail.dropna(subset=['CustomerID'])
df_retail["CustomerID"] = df_retail["CustomerID"].astype('Int64').astype(str)

df_retail.info()
```
Out [7]:
```
<class 'pandas.core.frame.DataFrame'>
Index: 406829 entries, 0 to 541908
Data columns (total 8 columns):
 #   Column       Non-Null Count   Dtype         
---  ------       --------------   -----         
 0   InvoiceNo    406829 non-null  object        
 1   StockCode    406829 non-null  object        
 2   Description  406829 non-null  object        
 3   Quantity     406829 non-null  int64         
 4   InvoiceDate  406829 non-null  datetime64[ns]
 5   UnitPrice    406829 non-null  float64       
 6   CustomerID   406829 non-null  object        
 7   Country      406829 non-null  object        
dtypes: datetime64[ns](1), float64(1), int64(1), object(5)
memory usage: 27.9+ MB
```
In [8]:
```python
# Remove invalid data points
df_retail = df_retail [~df_retail ['InvoiceNo'].astype(str).str.startswith("C")]
df_retail = df_retail [df_retail ['Quantity'] > 0]
df_retail = df_retail [df_retail ['UnitPrice'] > 0]

df_retail.describe()
```
Out [8]:

<img width="516" height="291" alt="image" src="https://github.com/user-attachments/assets/87ddd9ae-dc23-4870-98bc-9a12421e1fb5" />

In [9]:
```python
# Remove duplicates
df_retail.drop_duplicates(inplace=True)
df_retail.duplicated().sum()
```
Out [9]:
```
np.int64(0)
```
In [10]:
```python
# Check df after wrangling data
df_retail.info()
```
Out [10]:
```
<class 'pandas.core.frame.DataFrame'>
Index: 392692 entries, 0 to 541908
Data columns (total 8 columns):
 #   Column       Non-Null Count   Dtype         
---  ------       --------------   -----         
 0   InvoiceNo    392692 non-null  object        
 1   StockCode    392692 non-null  object        
 2   Description  392692 non-null  object        
 3   Quantity     392692 non-null  int64         
 4   InvoiceDate  392692 non-null  datetime64[ns]
 5   UnitPrice    392692 non-null  float64       
 6   CustomerID   392692 non-null  object        
 7   Country      392692 non-null  object        
dtypes: datetime64[ns](1), float64(1), int64(1), object(5)
memory usage: 27.0+ MB
```
### Actions
- Convert to correct data types.
- Remove missing values in CustomerID column.
- Remove incorrect data points in Quantity and Unit Price columns.
- Remove duplicated values.
### Result
After wrangling, the dataset has 392,692 rows with no missing values, no duplicates and correct data points in Quantity and Unit Price columns.

---

## 🧮 RFM Score  & Segment
In this section, RFM analysis is applied to evaluate customer value and behavior based on **Recency, Frequency, and Monetary** metrics. Each customer is assigned R, F, and M scores using a **1–5 scale**, where the scoring is derived from **quintile-based distribution** to ensure a balanced and statistically sound segmentation. Based on the combination of these RFM scores and a predefined classification table, customers are then grouped into meaningful segments, enabling clearer insights into customer behavior and supporting targeted business strategies.

In [11]:
```python
# Set current date = 2011-12-31
current_date = pd.to_datetime('2011-12-31')
```
In [12]:
```python
# Calculate Amount = Quantity * UnitPrice
df_retail['Amount'] = df_retail['Quantity'] * df_retail['UnitPrice']
df_retail.head()
```
Out [12]:

<img width="1083" height="198" alt="image" src="https://github.com/user-attachments/assets/ad5b070e-afae-42a1-a709-5779ac35c880" />

In [13]:
```python
# Create df_RFM
df_RFM = df_retail.groupby('CustomerID').agg(Recency=('InvoiceDate', lambda x: (current_date - x.max()).days),  # Calculate the number of days since the last purchase
                                             Frequency=('InvoiceNo', 'nunique'),  # Count the number of transactions per customer
                                             Monetary=('Amount', 'sum')  # Sum the total revenue per customer
                                             ).reset_index()
df_RFM
```
Out [13]:

<img width="381" height="415" alt="image" src="https://github.com/user-attachments/assets/7b65f9c4-438a-45f6-8b51-8392f1bd09ab" />

In [14]:
```python
# R Score (lower Recency → higher score)
df_RFM['R_Score'] = pd.qcut(df_RFM['Recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)

# F Score (higher Frequency → higher score)
df_RFM['F_Score'] = pd.qcut(df_RFM['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)

# M Score (higher Monetary → higher score)
df_RFM['M_Score'] = pd.qcut(df_RFM['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)

# Combined RFM Score
df_RFM['RFM_Score'] = (df_RFM['R_Score'].astype(str) +
                       df_RFM['F_Score'].astype(str) +
                       df_RFM['M_Score'].astype(str)
                      )
df_RFM
```
Out [14]:

<img width="673" height="411" alt="image" src="https://github.com/user-attachments/assets/f29700d3-b095-453b-8403-9001f888c155" />

In [15]:
```python
# Flatten df_segment by splitting the 'RFM Score' column
df_segment['RFM Score'] = df_segment['RFM Score'].astype(str).str.split(',')
df_segment = df_segment.explode('RFM Score').reset_index(drop=True)

# Trim spaces in the 'RFM Score' column to ensure proper merging
df_segment['RFM Score'] = df_segment['RFM Score'].str.strip()

# Merge df_RFM with df_segment on the 'RFM Score'
df_RFM = pd.merge(df_RFM, df_segment, how='left', left_on='RFM_Score', right_on='RFM Score')
df_RFM = df_RFM.drop(columns=['RFM Score'])

df_RFM
```
Out [15]:

<img width="806" height="412" alt="image" src="https://github.com/user-attachments/assets/2eacd8bd-c9df-4f2f-88e8-76a91ea05ef0" />

---

## 📊 Visualizations & Key Insights
### 1. Total Customers by Segmentation
<img width="950" height="505" alt="image" src="https://github.com/user-attachments/assets/48ff1b0e-6eff-4ad8-8760-f5270bcf40a0" />

🔎 **Key Insights:** There is a strong core of Champions (19.2%) and a large block of Hibernating (15.9%), plus meaningful shares of Loyal, Lost, At Risk, Potential Loyalist, etc. The revenue distribution is highly skewed: Champions drive ~63% of total monetary while many segments contribute little individually.

### 2. Recency by Segmentation
<img width="984" height="545" alt="image" src="https://github.com/user-attachments/assets/01035fb8-ec12-402b-aa98-a8d46fffb4d8" />

🔎 **Key Insights:** Several segments show poor recency (Lost ~295 days), At Risk and Hibernating have long gaps, while Champions and some Potential Loyalists show recent activity. Recency separates “ready-to-buy” customers from those needing reactivation.

### 3. Frequency by Segmentation
<img width="979" height="545" alt="image" src="https://github.com/user-attachments/assets/655b3ddd-5666-40fd-871e-1e1041177566" />

🔎 **Key Insights:** Champions have very high frequency (12.1), Loyal are moderate, most other segments show low frequency — indicating either low repeat purchase behavior or one-off buyers in many segments.

### 4. Monetary by Segmentation
<img width="981" height="588" alt="image" src="https://github.com/user-attachments/assets/a13ff728-68c9-4ad6-aa20-263c31113d01" />

🔎 **Key Insights:** Champions dominate monetary share. Some segments with moderate frequency (Loyal, Potential Loyalist) have decent monetary share potential. Hibernating and Lost are low monetary but can be cost-effective to reactivate.

### Two segments with the highest proportion of customers are Champions and Hibernating Customers.
### 5. Champions - Total Orders & Total Sales Over Time
<img width="1186" height="586" alt="image" src="https://github.com/user-attachments/assets/c6df8413-687e-4bfa-8fea-96318f263414" />

🔎 **Key Insights:** Champions produce the bulk of orders and sales but show high month-to-month volatility (huge spikes in Nov, crash in Dec). This suggests Champions respond strongly to seasonal events / major campaigns. Their behavior is high-value but campaign-driven.

### 6. Hibernating Customers - Total Orders & Total Sales Over Time
<img width="1185" height="588" alt="image" src="https://github.com/user-attachments/assets/2eff7fa7-c22a-469a-b617-33d318d54a27" />

🔎 **Key Insights:** Hibernating customers show mostly low activity but spike when strong incentives or campaigns run (e.g., strong activation in Sep). They are responsive but require a clear trigger to purchase.

---

## 🚀 Recommendations
| Customer Segment | Goal | Tactics | Channels | KPIs |
|------------------|------|---------|----------|------|
| **Champions (protect & stabilize)** | Reduce revenue volatility and increase lifetime value further | - Tier-based VIP program (early access, exclusive bundles, personalized concierge)<br>- Predictive replenishment / subscription offers for frequently bought SKUs<br>- Personalization: curated collections via email and on-site homepage<br>- Soft win-back: VIP-only flash events to smooth purchase spikes | Email (highly personalized), push, paid social lookalike campaigns | - Share of revenue from Champions (month-to-month stability)<br>- Repeat rate within 30 days<br>- Churn rate among Champions |
| **Hibernating Customers (cheap reactivation)** | Move them back to Active with minimal CAC | - Micro-conversion campaigns (free shipping + small discount tied to new catalog updates)<br>- 3-touch win-back email/SMS sequence within 14 days with increasing incentives<br>- Time-limited bundles based on previously purchased categories | Email, SMS, owned push, remarketing (dynamic ads) | - Reactivation rate (purchase within 30 days)<br>- CAAV (cost per activated customer)<br>- Post-reactivation AOV |
| **At Risk & About-to-Sleep (prevent churn)** | Stop drift before they leave | - Behavioral triggers based on historical purchase cadence<br>- Exit-intent surveys + friction reduction (returns, price match)<br>- Loyalty point multipliers for next purchase | Email, onsite banners, cart/UX nudges | - Churn rate reduction<br>- Triggered campaign conversion rate<br>- Survey NPS |
| **Potential Loyalists & Promising (scale mid-LTV)** | Convert to Loyal / Champion | - Post-first-purchase onboarding flows (education, usage tips, cross-sell)<br>- Second-purchase incentive (e.g., 15% off)<br>- Social proof and bundles aligned with first-buy category | Email automation, in-app messages, retargeting | - Second-purchase rate<br>- Time between 1st and 2nd purchase<br>- Purchase frequency uplift |
| **Lost Customers (low priority reactivation)** | Low-cost experiments only | - Quarterly re-discovery campaigns highlighting major changes or new categories<br>- Shift budget to lookalike audiences instead of heavy reactivation spend | Email, paid media (lookalikes) | - Activation per campaign spent<br>- Cost per reactivated customer |

### 📢 Which RFM metric should Marketing & Sales prioritize?

- Marketing — Prioritize Recency (R)
    
    Reason: Recency is the strongest near-term predictor of purchase behavior (Champions & Potential Loyalists show high recency; Hibernating respond when re-engaged). Recency-based targeting yields the best conversion efficiency and lowest CAC on reactivation. Use R to drive campaign timing and personalization.
    
- Sales — Prioritize Monetary (M) (with Recency as a secondary filter)
    
    Reason: Sales’ remit is revenue maximization and higher-ticket conversion. Targeting customers with high historical monetary value (and acceptable recency) yields better ROI for one-on-one or B2B-style selling approaches.
    
- Frequency (F) is important as a supporting metric: use it to identify repeat-buy patterns (ideal for subscription offers) and to segment mid-LTV cohorts. But as the single highest-priority metric for operational campaign targeting, R > M > F for Marketing; M > R > F for Sales.
