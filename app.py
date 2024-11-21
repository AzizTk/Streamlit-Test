import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import plotly.graph_objects as go

import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
# import required libraries
import pandas as pd
import numpy as np
import sklearn
import tensorflow
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation, Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential, load_model

import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
import math
import xgboost
import time
from tqdm import tqdm
import math
#function for creating and training models using the "Random forest" and "XGBoost" algorithms
def train_models(data,model = 'FOREST'):
    
    if model != 'LSTM':
        X = data.iloc[:,:14].to_numpy() 
        Y = data.iloc[:,14:].to_numpy()
        Y = np.ravel(Y)

    if model == 'FOREST':
         #  parameters for models are selected in a similar cycle, with the introduction 
         # of an additional param parameter into the function:
         #for i in range(1,11):
         #     xgb = train_models(train_df,param=i,model="XGB",)
         #     y_xgb_i_pred = xgb.predict(X_001_test)
         #     print(f'param = {i}')
         #     score_func(y_true,y_xgb_i_pred)
        model = RandomForestRegressor(n_estimators=70, max_features=7, max_depth=5, n_jobs=-1, random_state=1)
        model.fit(X,Y)
        return model
    
    elif model == 'XGB':
        model = xgboost.XGBRegressor(n_estimators=110, learning_rate=0.018, gamma=0, subsample=0.8,
                           colsample_bytree=0.5, max_depth=3,silent=True)
        model.fit(X,Y)
        return model
    
    elif model == 'LSTM':
        seq_array, label_array, lstm_test_df, sequence_length, sequence_cols = lstm_data_preprocessing(data[0], data[1], data[2])
        model_instance, history = lstm_train(seq_array, label_array, sequence_length)
        return model_instance, history, lstm_test_df, seq_array, label_array, sequence_length, sequence_cols
            
    return

#function for joint display of real and predicted values

def plot_result(y_true,y_pred):
    rcParams['figure.figsize'] = 12,10
    plt.plot(y_pred)
    plt.plot(y_true)
    plt.tick_params(axis='x', which='both', bottom=False, top=False,labelbottom=False)
    plt.ylabel('RUL')
    plt.xlabel('training samples')
    plt.legend(('Predicted', 'True'), loc='upper right')
    plt.title('COMPARISION OF Real and Predicted values')
    plt.show()
    return
# Set page config for layout
st.set_page_config(page_title="NASA C-MAPSS Data Explorer", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Dataset Selection","EDA", "FD001 Preprocessing", "Model Training","Conclusion"])
# Home Page
if page == "Home":
    st.title("Welcome to NASA C-MAPSS Data Explorer")
    st.write("This app allows you to explore the NASA C-MAPSS dataset and visualize turbofan engine data.")
    st.write("Use the navigation panel on the left to explore the different features of the app.")
    st.markdown("""
    ## Description
    Prognostics and health management is an important topic in industry for predicting the state of assets to avoid downtime and failures. This dataset is the Kaggle version of the well-known **NASA's Asset Degradation Modeling** dataset, specifically focusing on turbo fan jet engines. The data set includes **Run-to-Failure simulated data** and was generated using the C-MAPSS simulation tool.

    ### Engine Degradation Simulation
    Four different sets were simulated under different combinations of operational conditions and fault modes. The data includes several sensor channels that characterize fault evolution.

    - The data set was provided by the **Prognostics CoE at NASA Ames**.
    - It records sensor data across various operational scenarios.
    
    ### Prediction Goal
    The goal is to predict the **Remaining Useful Life (RUL)** of each engine in the test dataset. RUL is equivalent to the number of remaining flights an engine can handle before failing.

    ### Experimental Scenario
    - The dataset consists of multiple **multivariate time series**.
    - Each time series corresponds to a different engine, simulating a fleet of engines of the same type.
    - The engines start with varying degrees of **initial wear and manufacturing variation**, which is considered normal and not a fault condition.
    - **Three operational settings** impact engine performance, and these are included in the dataset.
    - Sensor data is contaminated with noise, making it more realistic and challenging for predictions.
    
    #### Key Points:
    - **Training Set**: The fault grows in magnitude until system failure.
    - **Test Set**: The time series ends before system failure, and the task is to predict when failure will happen.
    - True **Remaining Useful Life (RUL)** values are provided for the test data.
    
    ### Data Overview
    The data is provided as a **zip-compressed text file** with 26 columns of sensor data. Each row represents a snapshot taken during an operational cycle, with the following columns:
    - **1)** Unit number
    - **2)** Time (in cycles)
    - **3-5)** Operational settings 1, 2, 3
    - **6-26)** Sensor measurements 1 through 26

    ---
    **Let's dive into the data and start exploring!**
    """)

    st.markdown("""
    ### Explore the dataset in more detail
    You can select the **Dataset Selection** option from the sidebar to upload and explore the data.
    """)
# Dataset Selection Page
elif page == "Dataset Selection":
    st.title("Dataset Selection")
    st.write("""
    ## Welcome to the Dataset Exploration Hub!
    
    Choose a dataset below to dive into the world of **Engine Prognostics** and explore the intricate details of how engine faults evolve over time.

    Each dataset corresponds to a different **engine** and provides unique scenarios with varying **operational conditions** and **fault modes**. Select the one that intrigues you most and start your journey into the heart of **predictive maintenance**.
    """)

    datasets = {
        "FD001": {
            "description": "Train: 100 trajectories, Test: 100 trajectories, Conditions: ONE (Sea Level), Fault Modes: ONE (HPC Degradation)",
            "details": "This dataset simulates engine failures under a single condition (sea level) with only one fault mode: **HPC Degradation**. A perfect starting point for understanding the fundamentals of engine health prediction."
        },
        "FD002": {
            "description": "Train: 260 trajectories, Test: 259 trajectories, Conditions: SIX, Fault Modes: ONE (HPC Degradation)",
            "details": "A more complex dataset with six different operational conditions, all still focusing on **HPC Degradation**. This one is designed to challenge your models with more varied data."
        },
        "FD003": {
            "description": "Train: 100 trajectories, Test: 100 trajectories, Conditions: ONE (Sea Level), Fault Modes: TWO (HPC Degradation, Fan Degradation)",
            "details": "This dataset introduces an additional **Fan Degradation** fault mode alongside the **HPC Degradation**, adding more complexity and making it ideal for testing multi-fault prediction capabilities."
        },
        "FD004": {
            "description": "Train: 248 trajectories, Test: 249 trajectories, Conditions: SIX, Fault Modes: TWO (HPC Degradation, Fan Degradation)",
            "details": "The most complex dataset in terms of both operational conditions (six) and fault modes (HPC and Fan Degradation). It's perfect for building robust models that can handle multiple fault scenarios and varying conditions."
        }
    }

    dataset_choice = st.selectbox("Select a Dataset", list(datasets.keys()))
    
    st.write(f"### You selected: **{dataset_choice}**")
    st.write(f"#### Dataset Overview:")
    st.write(datasets[dataset_choice]["description"])
    
    st.write(f"#### Key Features of {dataset_choice}:")
    st.write(datasets[dataset_choice]["details"])
    
    st.markdown("""
    ---
    **What's Next?**
    - Once you've selected a dataset, you can dive deeper into the data and explore its features, analyze its performance, and begin building predictive models.
    - Click on the tabs above to begin your EDA journey with the dataset you've chosen.
    """)

    # Store selected dataset for further exploration
    st.session_state.selected_dataset = dataset_choice



        

# Data Visualization Page
elif page == "EDA":
    if "selected_dataset" not in st.session_state:
        st.warning("Please select a dataset from the sidebar first.")
    else:
        selected_dataset = st.session_state.selected_dataset
        st.title(f"Exploratory Data Analysis for Dataset {selected_dataset}")
        
        st.write("""
        Welcome to the **Exploratory Data Analysis (EDA)** section! Here, we take a deep dive into the dataset you've selected. By understanding the data at a fundamental level, we can uncover hidden patterns, detect outliers, and explore trends that will guide our future analysis. Let's get started!
        """)

        # Load the dataset
        df_train = pd.read_csv(f'Cmaps/train_{selected_dataset}.txt', sep='\s+', header=None)

        index_names = ['unit_number', 'time']
        setting_names = ['operational setting 1', 'operational setting 2', 'operational setting 3']
        sensor_names = [f'sensor measurement {i}' for i in range(1, 22)]
        col_names = index_names + setting_names + sensor_names
        df_train.columns = col_names
        df_test = pd.read_csv(f'Cmaps/test_{selected_dataset}.txt', sep='\s+', header=None)

        # Recalculate Remaining Useful Life (RUL)
        df_train['RUL'] = df_train.groupby('unit_number')['time'].transform('max') - df_train['time']

        st.subheader(f"Dataset {selected_dataset} - Training Data")
        st.write("""
        The training data consists of time-series sensor readings from various engines in different operational settings. This data provides insights into how each engine deteriorates over time. 
        """)

        # Show the first few rows of the training dataset for a quick look
        st.write(df_train.head())

        # Display Summary Statistics for better understanding
        st.subheader(f"Summary Statistics for {selected_dataset}")
        st.write("""
        The summary statistics give us a snapshot of the dataset's central tendencies and spread. This is crucial to understand the range of values and the potential presence of any extreme values that may require further analysis.
        """)
        st.write(df_train.describe())

        # --- Correlation Heatmap ---
        st.subheader("Correlation Heatmap (Including RUL)")
        st.write("""
        The correlation heatmap provides a visual representation of how strongly different features are related to one another. Understanding these relationships is key to feature selection and model building, especially when trying to predict the **Remaining Useful Life (RUL)** of an engine.
        """)
        corr_matrix = df_train.corr()
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            hoverongaps=False
        ))
        fig_heatmap.update_layout(
            title="Interactive Correlation Heatmap",
            xaxis_title="Features",
            yaxis_title="Features",
            height=600
        )
        st.plotly_chart(fig_heatmap)

        # --- Boxplots for Outliers ---
        st.subheader("Boxplots for Outlier Detection")
        st.write("""
        Boxplots are great tools for detecting outliers in the data. Outliers can skew your model and lead to overfitting, so it's important to identify them early on. 
        In the next section, you can select any feature to visualize and inspect for unusual values.
        """)
        selected_feature = st.selectbox("Select a Feature to Visualize Outliers", df_train.columns)
        fig_boxplot = px.box(df_train, y=selected_feature, points="all")
        fig_boxplot.update_layout(title=f"Boxplot of {selected_feature}", yaxis_title=selected_feature)
        st.plotly_chart(fig_boxplot)

        # --- Sensor Variation Grid Plot ---
        st.subheader("Sensor Variation Grid Plot")
        st.write("""
        Visualizing sensor data over time helps us understand how each sensor behaves throughout the operational cycles. By plotting all the sensors on a grid, we can spot trends and irregularities across different sensors. 
        This can provide important clues about sensor malfunctions or unexpected engine behavior.
        """)
        num_sensors = len(sensor_names)
        cols = 4
        rows = (num_sensors // cols) + (num_sensors % cols > 0)

        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
        fig.tight_layout(pad=5.0)
        axes = axes.flatten()

        for i, sensor in enumerate(sensor_names):
            ax = axes[i]
            ax.plot(df_train['time'], df_train[sensor])
            ax.set_title(sensor)
            ax.set_xlabel("Time")
            ax.set_ylabel("Measurement")

        for j in range(num_sensors, len(axes)):
            axes[j].axis('off')

        st.pyplot(fig)

        # --- Turbofan Engines Lifetime ---
        st.subheader("Turbofan Engines Lifetime")
        st.write("""
        The lifetime of each turbofan engine (in terms of operational cycles) is an important indicator of its health. Visualizing the maximum time cycles of each engine helps us understand the operational lifespan and any potential patterns of failure.
        """)
        max_time_cycles = df_train.groupby('unit_number')['time'].max().reset_index()
        max_time_cycles.columns = ['Unit Number', 'Max Time Cycles']

        # Create the plot using Plotly
        lifetime_fig = px.bar(
            max_time_cycles,
            x='Max Time Cycles',
            y='Unit Number',
            orientation='h',
            title="Turbofan Engines Lifetime",
            labels={'Unit Number': 'Unit Number', 'Max Time Cycles': 'Time Cycles'},
            width=900,
            height=900
        )
        st.plotly_chart(lifetime_fig)
        # Adding insights at the end of the EDA page
        st.subheader("Key Insights from the Data Analysis")
        st.write("""
        After performing an exploratory data analysis on the selected dataset, here are some important takeaways:
        """)

        # Use markdown or st.markdown for formatting
        insights = [
            "**1) Data is Fully Numerical, No Encoding Needed**: The dataset contains only numerical values, so there’s no need for complex encoding techniques. This makes preprocessing simpler and faster.",
            "**2) No Major Outliers & No Missing Data**: Fortunately, the dataset does not contain significant outliers or missing values, meaning the data is clean and ready for analysis without requiring extensive data imputation or outlier treatment.",
            "**3) Scaling Needed Due to Feature Disparity**: Some features exhibit large value ranges. We’ll need to apply scaling to ensure that all features contribute equally to the model, avoiding bias toward features with larger values.",
            "**4) Heatmap & Grid Plot Identify Constant/Null Features**: Both the correlation heatmap and sensor variation grid plot highlight features with constant or null values. These features provide no useful information and should be dropped from the analysis to improve model efficiency."
        ]

        # Displaying the insights in a cool bullet-point format
        for insight in insights:
            st.markdown(f"• {insight}")
            
        st.write("""
        These insights provide a strong foundation for the next steps in data preprocessing and model building. Stay tuned as we move from exploration to more sophisticated analyses and predictions!
        """)




# FD001 Preprocessing Tab
# Set up content for FD001 Preprocessing
elif page == "FD001 Preprocessing":
    st.title("FD001 Dataset Preprocessing")
    
    if "selected_dataset" not in st.session_state or st.session_state.selected_dataset != "FD001":
        st.warning("Please select the FD001 dataset from the sidebar first.")
    else:
        st.write("We will now start preprocessing the FD001 dataset. Each modification will appear under this tab.")

        # Load FD001 dataset
        df_test = pd.read_csv('Cmaps/test_FD001.txt', sep='\s+', header=None)
        df_train = pd.read_csv('Cmaps/train_FD001.txt', sep='\s+', header=None)

        # Assign column names
        col_names = ['unit_number', 'time', 'operational setting 1', 'operational setting 2', 'operational setting 3']
        col_names += [f'sensor measurement {i}' for i in range(1, 22)]
        df_train.columns = col_names
        df_test.columns = col_names
        st.write("Initial Dataset:")
        st.write(df_train.head())

        st.markdown("""We're gonna calculate the RUL value based on the cycles we have: The formula is the Max Value of the time cycle we have minus the cycle of each individual engine.
                    This should give us an indicaton on when each engine died""")
        # Calculate RUL
        max_cycles = df_train.groupby('unit_number')['time'].max()
        df_train = df_train.merge(max_cycles.rename('max_cycle'), on='unit_number')
        df_train['RUL'] = df_train['max_cycle'] - df_train['time']
        df_train = df_train.drop(columns=['max_cycle'])
        st.write("Dataset with RUL:")
        st.write(df_train.head())

        max_cycles = df_test.groupby('unit_number')['time'].max()
        df_test = df_test.merge(max_cycles.rename('max_cycle'), on='unit_number')
        df_test['RUL'] = df_test['max_cycle'] - df_test['time']
        df_test = df_test.drop(columns=['max_cycle'])
        st.subheader("""
        **Columns to drop:** """)
        st.markdown("Unit number , Setting 1-2-3 : Low to none correlation with the RUL value")
       
        st.markdown("Sensor Measurement 1-5-10-16-18-19: Constant values, no real reason to keep these")
        # Drop specified columns from training data
        columns_to_drop = [
            'unit_number', 'operational setting 1', 'operational setting 2', 'operational setting 3', 
            'sensor measurement 1', 'sensor measurement 5',  
            'sensor measurement 10', 'sensor measurement 16', 'sensor measurement 18', 'sensor measurement 19'
        ]
        df_train = df_train.drop(columns=columns_to_drop)
        df_test = df_test.drop(columns=columns_to_drop)
        st.write("Dataset after dropping columns:")
        st.write(df_train.head())

        # Min-Max Scaling
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns)
        df_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)  # Use fit_transform only on train
        st.markdown("Scaling will be through MinMax, As its the best for **Numerical** values that are distant but not too incredibly distant")
        st.write("Scaled Dataset:")
        st.write(df_train_scaled.head())

        # Store the scaled datasets in session state
        st.session_state.df_train_scaled = df_train_scaled
        st.session_state.df_test_scaled = df_test_scaled

        # Feature Selection with RFE
        X = df_train_scaled.drop(columns=['RUL'])
        y = df_train_scaled['RUL']

        # RFE with Random Forest
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestRegressor
        
        rfe = RFE(RandomForestRegressor(n_estimators=10, random_state=42), n_features_to_select=10)
        rfe.fit(X, y)

        selected_features_rfe = X.columns[rfe.support_]
        # Include 'RUL' in the final datasets
        df_train_rfe_selected = df_train_scaled[selected_features_rfe]
        df_train_rfe_selected['RUL'] = y  # Add RUL back to the train dataset

        df_test_rfe_selected = df_test_scaled[selected_features_rfe]
        df_test_rfe_selected['RUL'] = df_test_scaled['RUL']  # Add RUL back to the test dataset
        
        # Save RFE selected dataframe
        st.session_state.df_train_rfe_selected = df_train_rfe_selected
        st.session_state.df_test_rfe_selected = df_test_rfe_selected

        st.write("Dataframe with Selected Features Based on RFE:")
        st.write(df_train_rfe_selected.head())





elif page == "Model Training":
    if "selected_dataset" not in st.session_state:
        st.warning("Please select a dataset from the sidebar first.")
    else:
        selected_dataset = st.session_state.selected_dataset
        st.title(f"Model Training for Dataset {selected_dataset}")

        # Load the feature-selected and scaled datasets from session state
        df_train_selected = st.session_state.df_train_rfe_selected  # Feature-selected and scaled training data

        # Split the training dataset into training and validation sets (80% train, 20% validation)
        df_train, df_val = train_test_split(df_train_selected, test_size=0.2, random_state=42)

        # Split data into features (X) and target (y) for training and validation sets
        X_train = df_train.drop(columns=['RUL'])
        y_train = df_train['RUL']
        X_val = df_val.drop(columns=['RUL'])
        y_val = df_val['RUL']

        # Train Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Train XGBoost Regressor
        xgb_model = XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)

        # Make predictions on the validation set
        rf_predictions = rf_model.predict(X_val)
        xgb_predictions = xgb_model.predict(X_val)

        # Plot predictions vs true RUL
        fig, ax = plt.subplots(2, 2, figsize=(15, 12))

        # Random Forest Predictions
        ax[0, 0].scatter(y_val, rf_predictions, color='blue', alpha=0.7, label='Predicted RUL', edgecolors='black', s=60, marker='o')
        ax[0, 0].scatter(y_val, y_val, color='red', alpha=0.7, label='True RUL', edgecolors='black', s=60, marker='x')
        ax[0, 0].plot([0, max(y_val)], [0, max(y_val)], color='green', linestyle='--', linewidth=2, label='Ideal Fit')
        ax[0, 0].set_title('Random Forest: Predicted vs True RUL')
        ax[0, 0].set_xlabel('True RUL')
        ax[0, 0].set_ylabel('Predicted RUL')
        ax[0, 0].legend()

        # XGBoost Predictions
        ax[0, 1].scatter(y_val, xgb_predictions, color='green', alpha=0.7, label='Predicted RUL', edgecolors='black', s=60, marker='o')
        ax[0, 1].scatter(y_val, y_val, color='red', alpha=0.7, label='True RUL', edgecolors='black', s=60, marker='x')
        ax[0, 1].plot([0, max(y_val)], [0, max(y_val)], color='green', linestyle='--', linewidth=2, label='Ideal Fit')
        ax[0, 1].set_title('XGBoost: Predicted vs True RUL')
        ax[0, 1].set_xlabel('True RUL')
        ax[0, 1].set_ylabel('Predicted RUL')
        ax[0, 1].legend()

        # Random Forest Feature Importance
        rf_importances = rf_model.feature_importances_
        rf_features = X_train.columns
        ax[1, 0].barh(rf_features, rf_importances, color='blue', alpha=0.7)
        ax[1, 0].set_title('Random Forest: Feature Importance')
        ax[1, 0].set_xlabel('Importance')

        # XGBoost Feature Importance
        xgb_importances = xgb_model.feature_importances_
        ax[1, 1].barh(rf_features, xgb_importances, color='green', alpha=0.7)
        ax[1, 1].set_title('XGBoost: Feature Importance')
        ax[1, 1].set_xlabel('Importance')

        plt.tight_layout()
        st.pyplot(fig)

        # Calculate and display performance metrics
        rf_mse = mean_squared_error(y_val, rf_predictions)
        xgb_mse = mean_squared_error(y_val, xgb_predictions)
        rf_r2 = rf_model.score(X_val, y_val)
        xgb_r2 = xgb_model.score(X_val, y_val)

        st.write(f"### Random Forest Metrics:")
        st.write(f"- Mean Squared Error (MSE): {rf_mse}")
        st.write(f"- R-Squared (R²): {rf_r2}")

        st.write(f"### XGBoost Metrics:")
        st.write(f"- Mean Squared Error (MSE): {xgb_mse}")
        st.write(f"- R-Squared (R²): {xgb_r2}")

        # Downloadable Results
        rf_results = pd.DataFrame({'True RUL': y_val, 'Predicted RUL (RF)': rf_predictions})
        xgb_results = pd.DataFrame({'True RUL': y_val, 'Predicted RUL (XGB)': xgb_predictions})

        # Option to download the results as CSV files
        st.download_button(
            label="Download Random Forest Results",
            data=rf_results.to_csv(index=False),
            file_name="rf_results.csv",
            mime="text/csv"
        )
        st.download_button(
            label="Download XGBoost Results",
            data=xgb_results.to_csv(index=False),
            file_name="xgb_results.csv",
            mime="text/csv"
        )
elif page == "Conclusion":
    st.markdown("""# Conclusion

As we’ve explored and preprocessed the **FD001** dataset, we've come a long way in preparing it for predictive modeling. Below are the key takeaways from this project:

---

### 1. Comprehensive Data Understanding
- We’ve gained a solid understanding of the **FD001 dataset**, including its structure and relevant features. The dataset is well-suited for predictive maintenance tasks, with continuous numerical data and clear relationships between sensor measurements and engine health.

---

### 2. Effective Feature Engineering
- By calculating the **Remaining Useful Life (RUL)**, we created a powerful target variable for training predictive models. This step is central to understanding how soon each engine will fail and enabling maintenance teams to take action before failures occur.

---

### 3. Data Cleaning & Feature Selection
- We carefully cleaned the data by removing irrelevant features and outliers, ensuring that the model only focuses on the most important information. Through **Recursive Feature Elimination (RFE)**, we selected the top features, optimizing model performance while reducing complexity.

---

### 4. Data Scaling & Transformation
- To enhance model performance, we applied **Min-Max Scaling**, ensuring that all features were on the same scale. This normalization prevents certain features with larger ranges from dominating the model, improving accuracy.

---

### 5. Key Next Steps
- With the dataset preprocessed and ready, the next steps involve building and training machine learning models to predict **Remaining Useful Life (RUL)**. The features we’ve selected will be used to train models such as Random Forest, Gradient Boosting, or even neural networks to predict engine failures in real-time.

---

## Future Work

Moving forward, we can enhance this analysis by:

- **Model Evaluation**: Testing different machine learning algorithms and tuning them for the best performance using metrics like RMSE (Root Mean Squared Error) or MAE (Mean Absolute Error).
- **Real-Time Application**: Deploying the trained model to predict engine failures in real-time, integrating it with existing maintenance systems.
- **Expanding Dataset**: Including other datasets like **FD002**, **FD003**, and **FD004** to further refine and generalize our model.

---
                

                LSTM Model & Other Regressors: https://www.kaggle.com/code/aziztaktak/damage-propagation-modeling-for-aircraft-engine
""")

