import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
# Importing required libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# This must come immediately after imports and before any Streamlit calls
st.set_page_config(page_title="NASA C-MAPSS Data Explorer", layout="wide")

# Your app's logic starts here
st.title("NASA C-MAPSS Data Explorer")
st.write("Welcome to the NASA C-MAPSS dataset exploration tool!")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', False)

# Title of the app
st.title("NASA Predictive Maintenance - RUL Analysis")

# Load the dataset
df_train = pd.read_csv('Cmaps/train_FD001.txt', sep='\s+', header=None)
df_test = pd.read_csv('Cmaps/test_FD001.txt', sep='\s+', header=None)
df_test_RUL = pd.read_csv('Cmaps/RUL_FD001.txt', sep='\s+', header=None)

# Feature names for the dataset
index_names = ['engine', 'cycle']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = [
    "(Fan inlet temperature) (◦R)", "(LPC outlet temperature) (◦R)", "(HPC outlet temperature) (◦R)", 
    "(LPT outlet temperature) (◦R)", "(Fan inlet Pressure) (psia)", "(bypass-duct pressure) (psia)", 
    "(HPC outlet pressure) (psia)", "(Physical fan speed) (rpm)", "(Physical core speed) (rpm)", 
    "(Engine pressure ratio(P50/P2)", "(HPC outlet Static pressure) (psia)", "(Ratio of fuel flow to Ps30) (pps/psia)", 
    "(Corrected fan speed) (rpm)", "(Corrected core speed) (rpm)", "(Bypass Ratio)", "(Burner fuel-air ratio)",
    "(Bleed Enthalpy)", "(Required fan speed)", "(Required fan conversion speed)", "(High-pressure turbines Cool air flow)", 
    "(Low-pressure turbines Cool air flow)"
]
col_names = index_names + setting_names + sensor_names

df_train.columns = col_names
df_test.columns = col_names
df_test_RUL.columns = ['RUL']

# Show the first few rows of the training dataset
st.subheader("Training Data Preview")
st.write(df_train.head())

# Display Correlation Heatmap
st.subheader("Correlation Heatmap")
corr_matrix = df_train.corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu', linewidths=0.8, linecolor='lightgrey')
st.pyplot(plt)

# Plotting with Plotly
st.subheader("Interactive Plot with Plotly")
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=df_train['cycle'], y=df_train['(Fan inlet temperature) (◦R)'], mode='lines', name='Fan Inlet Temp'))
fig.update_layout(title='Cycle vs Fan Inlet Temperature')
st.plotly_chart(fig)
import streamlit as st

# Set the page title and layout
st.set_page_config(page_title="NASA C-MAPSS Data Explorer", layout="wide")

# Define the datasets and their descriptions
datasets = {
    "FD001": {
        "description": "Train: 100, Test: 100, Conditions: ONE (Sea Level), Fault Modes: ONE (HPC Degradation)",
        "image": "download.png"
    },
    "FD002": {
        "description": "Train: 260, Test: 259, Conditions: SIX, Fault Modes: ONE (HPC Degradation)",
        "image": "download.png"
    },
    "FD003": {
        "description": "Train: 100, Test: 100, Conditions: ONE (Sea Level), Fault Modes: TWO (HPC, Fan Degradation)",
        "image": "download.png"
    },
    "FD004": {
        "description": "Train: 248, Test: 249, Conditions: SIX, Fault Modes: TWO (HPC, Fan Degradation)",
        "image": "download.png"
    }
}

# Create a title
st.title("NASA C-MAPSS Data Explorer")

# Display clickable images for each dataset
selected_dataset = None
col1, col2, col3, col4 = st.columns(4)

for i, (key, value) in enumerate(datasets.items()):
    with [col1, col2, col3, col4][i]:
        if st.button(f"{key}"):
            selected_dataset = key
        st.image(value["image"], caption=key, use_column_width=True)

# Render the selected dataset details
if selected_dataset:
    st.header(f"Dataset: {selected_dataset}")
    st.write(datasets[selected_dataset]["description"])
    st.markdown("---")
    st.subheader(f"Exploration for {selected_dataset}")
    # Add your logic to display details about the dataset (e.g., data, plots, etc.)

