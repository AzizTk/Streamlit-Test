# Save this in a Python file, e.g., app.py
import streamlit as st
import pandas as pd
import numpy as np

st.title("Streamlit Example in Kaggle")
st.write("This is a simple Streamlit page for testing.")

# Create some random data
data = pd.DataFrame({
    'x': np.arange(0, 10),
    'y': np.random.randn(10)
})

# Show a chart
st.line_chart(data)