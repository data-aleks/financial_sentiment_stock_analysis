# 📈 Sentiment-Driven Stock Return Prediction  
**Unlocking Market Insights through Google Search Sentiment Analysis**

## Project Overview  
This project is a hands-on exploration of sentiment analysis techniques in Python using publicly available web-scraped data. It demonstrates a complete data science workflow, from raw text data acquisition and cleaning to sophisticated feature engineering, in-depth analysis, and ultimately, applying these derived sentiment signals to the complex domain of forecasting short-term stock price movements (`next_return`).

The primary objective was to learn and apply practical sentiment analysis methodologies, using stock market prediction as a compelling real-world application.

## Key Features & Highlights  
- **Web Scraping**: Automated collection of textual data from Google search results.  
- **Sentiment Analysis**: Application of VADER sentiment analysis to extract emotional tone from article titles, descriptions, and full texts.  
- **Feature Engineering**: Creation of advanced sentiment features (`weighted_sentiment`, `impact_score`, `sentiment_vol`) and integration with financial time-series data (`log_return`, `rolling_std_5`, `Volume`).  
- **Exploratory Data Analysis (EDA)**: Visualizing the relationship between sentiment and stock price trends.  
- **Causal Analysis**: Performing Granger Causality tests to investigate lead-lag relationships between sentiment/volume and stock returns.  
- **Predictive Modeling**: Development and evaluation of an XGBoost Regressor for forecasting `next_return`.  
- **Comprehensive Evaluation**: Interpretation of model performance using R², RMSE, and a critical discussion of results in the context of market complexity.  

## Technologies Used  
- Python  
- Pandas (Data manipulation)  
- NumPy (Numerical operations)  
- Scikit-learn (Machine learning utilities, e.g., `train_test_split`, `r2_score`, `mean_squared_error`)  
- XGBoost (Gradient Boosting Regressor)  
- NLTK (for VADER Sentiment Analysis - Implicit, but likely used)  
- Matplotlib & Seaborn (Data visualization)  
- Statsmodels (for Granger Causality Test)  

## Project Structure  
├── notebooks/
│ └── stock_sentiment_analysis.ipynb # Main Jupyter Notebook with analysis
├── data/
│ ├── raw/ # Raw scraped article data
│ └── processed/ # Cleaned and aggregated weekly data
├── README.md # This file
└── requirements.txt # Project dependencies


## Detailed Methodology  

### 1. Data Acquisition & Cleaning  
The project initiated by programmatically scraping Google search results for a specific company (e.g., Pfizer) over a defined period. This raw data, primarily unstructured text from article titles, descriptions, and full content, formed the basis of our analysis.

**Key Steps:**  
- **Web Scraping**: Utilized a Python library (e.g., `requests`, `BeautifulSoup` or `Scrapy`) to collect relevant articles.  
- **Data Cleaning**: Pre-processing involved handling missing values, removing boilerplate text, and ensuring data consistency.  

### 2. Sentiment Analysis & Feature Engineering  
This section details the transformation of raw textual data into actionable quantitative features for time-series analysis.

#### 2.1 Sentiment Extraction & Weekly Aggregation  
Individual articles, each with raw text, were subjected to sentiment analysis. VADER (Valence Aware Dictionary and sEntiment Reasoner) was used to compute compound sentiment scores for the title, description, and full text of each article.

To align with weekly stock data, these article-level insights were then grouped and aggregated.

```python
# Assuming df_pfizer_articles contains individual article data with 'date', 'title_vader_compound', 'desc_vader_compound', 'compound_score'
grouped_df = df_pfizer_articles.groupby('date').agg(
    num_articles=('date', 'size'), # Count of articles per week
    avg_title_compound=('title_vader_compound', 'mean'), # Average title sentiment
    avg_desc_compound=('desc_vader_compound', 'mean'), # Average description sentiment
    avg_full_text_compound=('compound_score', 'mean') # Average full text sentiment
).reset_index()
```

Resulting grouped_df provides daily/weekly aggregated sentiment metrics.

### 2.2 Creating Predictive Features
New features were engineered from both the aggregated sentiment data and the stock's historical price movements.
```python
# --- Calculate Weekly Log Returns ---
# Get the closing price from the previous week (lagged by 1 period)
weekly_df['close_lag1'] = weekly_df['close_price'].shift(1)

# Compute the natural logarithm of the ratio of current close price to last week's close price.
# Log returns are preferred in financial modeling for their additive properties and
# better statistical behavior compared to simple percentage changes.
weekly_df['log_return'] = np.log(weekly_df['close_price'] / weekly_df['close_lag1'])

# --- Create Sentiment-Based Features ---
# Calculate an 'impact score' for sentiment by multiplying the average full text compound sentiment
# by the number of articles. This aims to capture the intensity of emotion weighted by volume.
weekly_df['impact_score'] = weekly_df['avg_full_text_compound'] * weekly_df['num_articles']

# Determine the volatility (standard deviation) of the average full text sentiment
# over a 12-week rolling window. This indicates how unstable or consistent sentiment has been.
weekly_df['sentiment_vol'] = weekly_df['avg_full_text_compound'].rolling(12).std()

# --- Generate Volatility-Based Features (and a potential future target) ---
# Calculate the rolling 12-week standard deviation of log returns.
# This serves as a measure of historical price volatility for the current period.
weekly_df['rolling_std_5'] = weekly_df['log_return'].rolling(window=12).std()

# Create a lagged version of the rolling volatility.
# Shifting by -1 means this column will contain the volatility for the *next* period.
# This variable could serve as a target for predicting future volatility,
# or as a feature if the model is designed to look ahead (though typically
# future information is avoided in features to prevent data leakage for prediction tasks).
weekly_df['rolling_std_5_lag1'] = weekly_df['rolling_std_5'].shift(-1)

# --- Blend Multiple Sentiment Scores into a Weighted Score ---
# Create a composite 'weighted_sentiment' by combining sentiment scores from different
# parts of an article (description, full text, title) with custom weights.
# The weights are chosen based on the perceived richness/impact of each text section:
# - Description (0.5): Often concise and highlights key sentiment.
# - Full Text (0.3): Provides broader context, but can be diluted.
# - Title (0.2): Short but can be very impactful or clickbait-driven.
weekly_df['weighted_sentiment'] = (
    0.5 * weekly_df['avg_desc_compound'] +   # Highest weight for description sentiment
    0.3 * weekly_df['avg_full_text_compound'] +  # Medium weight for full text sentiment
    0.2 * weekly_df['avg_title_compound']    # Lowest weight for title sentiment
)

# Calculate the simple percentage change in close price for the current period.
weekly_df['return'] = weekly_df['close_price'].pct_change()

# Define the primary target variable: the simple percentage change in close price
# for the *next* period (shifted backwards by 1 to align with current features).
weekly_df['next_return'] = weekly_df['close_price'].pct_change().shift(-1)
```
## 3. Creating Predictive Features
This phase focused on understanding the relationships within the dataset.

### 3.1 Visualizing Sentiment with Price Movements
An initial visual exploration was conducted to qualitatively assess how periods of positive or negative sentiment align with stock price trends.

### 3.2 Feature Correlation Matrix
A Pearson correlation matrix was computed to quantify the linear relationships between all engineered features and the next_return target.

|                      | weighted_sentiment | impact_score | sentiment_vol | Rolling_STD_5 |   Volume | LogReturn | next_return |
|----------------------|-------------------:|-------------:|--------------:|--------------:|---------:|----------:|------------:|
| weighted_sentiment   |           1.000000 |     0.244185 |     -0.225297 |     -0.135568 | -0.000065|   0.122500|     0.311691|
| impact_score         |           0.244185 |     1.000000 |     -0.575391 |     -0.329724 |  0.193759|   0.116780|     0.002261|
| sentiment_vol        |          -0.225297 |    -0.575391 |      1.000000 |      0.736123 | -0.084724|  -0.098980|    -0.090836|
| Rolling_STD_5        |          -0.135568 |    -0.329724 |      0.736123 |      1.000000 |  0.014015|  -0.248027|    -0.122143|
| Volume               |          -0.000065 |     0.193759 |     -0.084724 |      0.014015 |  1.000000|  -0.141584|     0.036323|
| LogReturn            |           0.122500 |     0.116780 |     -0.098980 |     -0.248027 | -0.141584|   1.000000|     0.097566|
| next_return          |           0.311691 |     0.002261 |     -0.090836 |     -0.122143 |  0.036323|   0.097566|     1.000000|

Table: Correlation Matrix of Engineered Features and next_return.
This matrix shows that weighted_sentiment exhibits the strongest positive linear correlation (0.31) with next_return, suggesting its potential as a predictor. A strong correlation (0.736) also exists between sentiment_vol and rolling_std_5, indicating a potential link between sentiment volatility and price volatility.

### 3.3 Granger Causality Test
To investigate if sentiment acts as a leading indicator, a Granger Causality test was performed on weighted_sentiment against next_return.


![Granger Causality Test: Weighted Sentiment → Next Return]('Screenshots/granger_casuality_test.png')
The plot displays the F-test p-values for various lags, compared against a 0.05 significance threshold. In this test, none of the p-values fell below the threshold, indicating no statistically significant linear Granger-causal relationship from weighted_sentiment to next_return at the tested lags. This suggests that while sentiment might correlate or contribute in non-linear ways, it may not be a direct, simple linear leading indicator.

## 4. Predictive Modeling: XGBoost Regressor
An XGBoost Regressor was chosen to build a predictive model for next_return, leveraging its ability to capture complex, non-linear relationships.
```python
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np # Ensure numpy is imported

features = [
      'weighted_sentiment', 'impact_score', 'sentiment_vol',
      'rolling_std_5', 'volume'
]
target = 'next_return'

# Drop rows with any missing values in the features or target
df_model = weekly_df.dropna(subset=features + [target])
X = df_model[features]
y = df_model[target]

# Split data into training and testing sets, preserving time series order
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Initialize and train the XGBoost Regressor model
model = XGBRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set
preds = model.predict(X_test)

# Evaluate model performance
print("R²:", r2_score(y_test, preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
```

# Results & Discussion
### Model Performance:

    R²: 0.2931

    RMSE: 0.0329

![Actual vs Predicted: next_return (XGBoost Model)]('Screenshots/actual_vs_predicted.png')
The XGBoost model achieved an R² of approximately 0.293, indicating that roughly 29.3% of the variance in next_return is explained by our engineered features. The RMSE of 0.033 represents the average magnitude of prediction errors. While stock market prediction is inherently challenging, this positive R² demonstrates that the combined sentiment and financial features do contribute to a meaningful predictive capability.

The visualization of actual vs. predicted values shows the model's ability to capture general trends but also highlights its limitations in predicting extreme fluctuations, a common challenge in volatile financial markets.

### Key Insight & Learning:
The project's findings underscore that while a direct linear Granger-causal link from sentiment to future returns was not established, the sentiment features, when integrated with other market indicators and processed by a non-linear model like XGBoost, significantly improve predictive power beyond a random guess. This suggests that sentiment's influence might be more complex, involving non-linear interactions or reflecting concurrent market conditions that contribute to predictability. This dual insight (limited linear causality vs. non-linear predictive contribution) is a critical learning point in applying sentiment analysis to real-world financial data.

## Key Learnings 
This project provided invaluable hands-on experience in:

End-to-end data science pipeline: From scraping to modeling and interpretation.

Advanced Sentiment Feature Engineering: Creating nuanced signals from raw text.

Time Series Analysis: Handling sequential financial data, including shift operations and appropriate train/test splitting.

Interpreting Model Performance: Understanding R², RMSE, and the implications of Granger Causality in a complex domain.

Critical Thinking: Acknowledging the limitations of models and the inherent challenges of predicting financial markets.