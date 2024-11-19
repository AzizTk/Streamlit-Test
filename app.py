import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


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

