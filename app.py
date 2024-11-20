import streamlit as st
import pandas as pd
import plotly.express as px
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
import math

# Set page config for layout
st.set_page_config(page_title="NASA C-MAPSS Data Explorer", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Dataset Selection", "Exploration", "Visualization", "FD001 Preprocessing", "Model Training"])
# Home Page
if page == "Home":
    st.title("Welcome to NASA C-MAPSS Data Explorer")
    st.write("This app allows you to explore the NASA C-MAPSS dataset and visualize turbofan engine data.")
    st.write("Use the navigation panel on the left to explore the different features of the app.")

# Dataset Selection Page
elif page == "Dataset Selection":
    st.title("Dataset Selection")
    st.write("Choose a dataset to explore:")

    datasets = {
        "FD001": {
            "description": "Train: 100, Test: 100, Conditions: ONE (Sea Level), Fault Modes: ONE (HPC Degradation)"
        },
        "FD002": {
            "description": "Train: 260, Test: 259, Conditions: SIX, Fault Modes: ONE (HPC Degradation)"
        },
        "FD003": {
            "description": "Train: 100, Test: 100, Conditions: ONE (Sea Level), Fault Modes: TWO (HPC, Fan Degradation)"
        },
        "FD004": {
            "description": "Train: 248, Test: 249, Conditions: SIX, Fault Modes: TWO (HPC, Fan Degradation)"
        }
    }

    dataset_choice = st.selectbox("Select a Dataset", list(datasets.keys()))
    st.write(f"You selected {dataset_choice}:")
    st.write(datasets[dataset_choice]["description"])

    # Store selected dataset for further exploration
    st.session_state.selected_dataset = dataset_choice

# Data Exploration Page
elif page == "Exploration":
    if "selected_dataset" not in st.session_state:
        st.warning("Please select a dataset from the sidebar first.")
    else:
        selected_dataset = st.session_state.selected_dataset
        st.title(f"Explore Dataset {selected_dataset}")

        # Load the dataset
        df_train = pd.read_csv(f'Cmaps/train_{selected_dataset}.txt', sep='\s+', header=None)
        index_names = ['unit_number', 'time']
        setting_names = ['operational setting 1', 'operational setting 2', 'operational setting 3']
        sensor_names = [f'sensor measurement {i}' for i in range(1, 22)]
        col_names = index_names + setting_names + sensor_names
        df_train.columns = col_names

        # Recalculate RUL
        df_train['RUL'] = df_train.groupby('unit_number')['time'].transform('max') - df_train['time']

        st.subheader(f"Dataset {selected_dataset} - Training Data")
        st.write(df_train.head())

        # Show data statistics
        st.subheader(f"Summary Statistics for {selected_dataset}")
        st.write(df_train.describe())

# Data Visualization Page
elif page == "Visualization":
    if "selected_dataset" not in st.session_state:
        st.warning("Please select a dataset from the sidebar first.")
    else:
        selected_dataset = st.session_state.selected_dataset
        st.title(f"Visualization for Dataset {selected_dataset}")

        # Load the dataset
        df_train = pd.read_csv(f'Cmaps/train_{selected_dataset}.txt', sep='\s+', header=None)

        index_names = ['unit_number', 'time']
        setting_names = ['operational setting 1', 'operational setting 2', 'operational setting 3']
        sensor_names = [f'sensor measurement {i}' for i in range(1, 22)]
        col_names = index_names + setting_names + sensor_names
        df_train.columns = col_names
        df_test = pd.read_csv(f'Cmaps/test_{selected_dataset}.txt', sep='\s+', header=None)

        # Recalculate RUL
        df_train['RUL'] = df_train.groupby('unit_number')['time'].transform('max') - df_train['time']
        # --- Correlation Heatmap ---
        st.subheader("Correlation Heatmap (Including RUL)")
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
        selected_feature = st.selectbox("Select a Feature to Visualize Outliers", df_train.columns)
        fig_boxplot = px.box(df_train, y=selected_feature, points="all")
        fig_boxplot.update_layout(title=f"Boxplot of {selected_feature}", yaxis_title=selected_feature)
        st.plotly_chart(fig_boxplot)

        # --- Sensor Variation Grid Plot ---
        st.subheader("Sensor Variation Grid Plot")
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

        # Drop specified columns from training data
        columns_to_drop = [
            'unit_number', 'operational setting 1', 'operational setting 2', 'operational setting 3', 
            'sensor measurement 1', 'sensor measurement 5', 'sensor measurement 6', 
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
        
        rfe = RFE(RandomForestRegressor(), n_features_to_select=10)
        rfe.fit(X, y)
        selected_features_rfe = X.columns[rfe.support_]
        df_train_rfe_selected = df_train_scaled[selected_features_rfe]
        
        # Apply RFE-selected features to the test set
        df_test_rfe_selected = df_test_scaled[selected_features_rfe]
        
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
        df_test_selected = st.session_state.df_test_rfe_selected    # Feature-selected and scaled test data
        y_test = df_test_selected['RUL']  # Extract RUL for the test set

        # Split data into features (X) and target (y)
        X_train = df_train_selected.drop(columns=['RUL'])
        y_train = df_train_selected['RUL']
        X_test = df_test_selected.drop(columns=['RUL'])

        # Train Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Train XGBoost Regressor
        xgb_model = XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)

        # Make predictions
        rf_predictions = rf_model.predict(X_test)
        xgb_predictions = xgb_model.predict(X_test)

        # Plot predictions vs true RUL
        plt.figure(figsize=(12, 6))

        # Random Forest Predictions
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, rf_predictions, color='blue', alpha=0.5)
        plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')
        plt.title('Random Forest: Predicted vs True RUL')
        plt.xlabel('True RUL')
        plt.ylabel('Predicted RUL')

        # XGBoost Predictions
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, xgb_predictions, color='green', alpha=0.5)
        plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')
        plt.title('XGBoost: Predicted vs True RUL')
        plt.xlabel('True RUL')
        plt.ylabel('Predicted RUL')

        plt.tight_layout()
        st.pyplot(plt)

        # Calculate and display performance
        rf_mse = mean_squared_error(y_test, rf_predictions)
        xgb_mse = mean_squared_error(y_test, xgb_predictions)

        st.write(f"Random Forest MSE: {rf_mse}")
        st.write(f"XGBoost MSE: {xgb_mse}")

    )

    # Display the heatmap
    st.plotly_chart(heatmap_fig)

