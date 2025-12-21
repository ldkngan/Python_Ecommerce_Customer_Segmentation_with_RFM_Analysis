# Python_Customer_Segmentation_RFM
This project focuses on customer segmentation using RFM analysis based on e-commerce transactional data. The goal is to transform raw sales data into actionable customer insights that support marketing, retention, and revenue optimization decisions.

---

## Table of Contents
- Overview
- Exploratory Data Analysis (EDA)
- Data Wrangling
- RFM Score & Segment
- Visualizations & Key Insights
- Recommendations

---

## Overview
### Main Objectives
- Segment customers based on **Recency, Frequency, and Monetary value**
- Identify **high-value**, **loyal**, **at-risk**, and **inactive** customer groups
- Understand how customer behavior changes **over time**
- Support business decisions related to **customer retention, re-engagement, and revenue growth**
### Why RFM Analysis?
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
### Business Questions
- Who are the **most valuable customers** contributing the highest revenue?
- Which customers are **active but low-spending**, and which are **high-spending but infrequent**?
- Which customers are **at risk of churn** due to declining recency or frequency?
- How does **customer value change over time**?
### Tools & Technologies
- **Python** (core language for the entire project)
- pandas, numpy – data manipulation & aggregation
- matplotlib / seaborn – data visualization
- Jupyter / Google Colab – analysis environment
### Dataset
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

## Exploratory Data Analysis (EDA)
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
# Import tables
from google.colab import drive
drive.mount('/content/drive')

path = '/content/drive/MyDrive/Python_Final Project/'

df_retail = pd.read_excel(path + 'ecommerce retail.xlsx', sheet_name="ecommerce retail")
df_segment = pd.read_excel(path + 'ecommerce retail.xlsx', sheet_name="Segmentation")

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

## Data Wrangling
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

## RFM Score  & Segment
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

## Visualizations & Insights
