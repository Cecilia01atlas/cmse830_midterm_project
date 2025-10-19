# 🌊 ENSO Interactive Analysis App

An interactive **Streamlit** app analyzing the influence of **ENSO (El Niño / La Niña)** events on ocean-atmosphere variables. The app provides exploratory data analysis, visualization, and missing data imputation for key climate variables.

---

## 📂 Data Sources

This project uses two main data sources:

1. **El Niño dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/122/el+nino).  
2. **ENSO Index CSV (`data_index.csv`)** – local CSV containing monthly ENSO indices (ANOM).

Both datasets are merged for analysis of temporal trends, correlations, and event-driven anomalies.

---

## 🖥 App Features

The app consists of **five interactive tabs**, each providing specific analyses:

- **Overview** 🌊  
  - ✅ Key metrics: total records, features, missing values  
  - 📋 Column information  
  - 🚨 Outlier detection  
  - 🔁 Duplicate detection  

- **Missingness Analysis** 💨  
  - 📊 Missingness summary table  
  - 🌡 Missingness heatmap  
  - 🔧 Humidity imputation using linear regression and stochastic noise  

- **Temporal Coverage** 🌐  
  - 📅 Daily scatter plots colored by ENSO index  
  - 📈 Line plots with ENSO event shading  
  - 📊 Monthly violin plots to show seasonal patterns  

- **Correlation Study** 📊  
  - 🔸 Correlation heatmap between key variables  
  - 🔸 Pairwise scatter matrix (colored by air temperature)  
  - 🔸 Scatter plots with regression line and binned SST vs air temperature plots  

- **Summary & Conclusion** 📖  
  - 🌊 Insights on ENSO influence  
  - 📌 Variable relationships  
  - 📊 Seasonal and temporal patterns  
  - ✅ Data quality and imputation notes
