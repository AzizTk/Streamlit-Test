import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# Set page config
st.set_page_config(page_title="NASA C-MAPSS Data Explorer", layout="wide")

# Your app's logic starts here
st.title("NASA C-MAPSS Data Explorer")
st.write("Welcome to the NASA C-MAPSS dataset exploration tool!")

# Set pandas options for display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', False)

# Define datasets and their descriptions
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

# Create columns for each dataset button
col1, col2, col3, col4 = st.columns(4)

# Display buttons for each dataset
selected_dataset = None
for i, (key, value) in enumerate(datasets.items()):
    with [col1, col2, col3, col4][i]:
        if st.button(f"{key}"):
            selected_dataset = key
        st.image(value["image"], caption=key, use_column_width=True)

# Logic to handle dataset exploration after button click
if selected_dataset:
    # Load the dataset
    df_train = pd.read_csv(f'Cmaps/train_{selected_dataset}.txt', sep='\s+', header=None)
    df_test = pd.read_csv(f'Cmaps/test_{selected_dataset}.txt', sep='\s+', header=None)
    df_test_RUL = pd.read_csv(f'Cmaps/RUL_{selected_dataset}.txt', sep='\s+', header=None)

    # Update column names based on the dataset
    index_names = ['unit_number', 'time']
    setting_names = ['operational setting 1', 'operational setting 2', 'operational setting 3']
    sensor_names = [f'sensor measurement {i}' for i in range(1, 22)]  # Adjust the number based on dataset

    col_names = index_names + setting_names + sensor_names
    df_train.columns = col_names
    df_test.columns = col_names
    df_test_RUL.columns = ['RUL']

    # Display dataset details
    st.header(f"Dataset: {selected_dataset}")
    st.write(datasets[selected_dataset]["description"])

    # Show the first few rows of the training dataset
    st.subheader("Training Data Preview")
    st.write(df_train.head())

    # Use pandas describe function to get statistics for numerical columns
    df_stats = df_train.describe().T  # Transpose for better readability

    # Display column stats
    df_stats.index.name = 'Column'
    st.subheader("Dataset Column Information and Statistics")
    st.write(df_stats)

    # Plot all sensor data in a grid
    st.subheader("Sensor Data Visualization (Grid Layout)")
    st.write("All sensor data plotted together:")

    num_sensors = len(sensor_names)
    cols = 3  # Number of columns in the grid
    rows = (num_sensors + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), constrained_layout=True)

    for idx, sensor in enumerate(sensor_names):
        row, col = divmod(idx, cols)
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.plot(df_train['time'], df_train[sensor])
        ax.set_title(sensor)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

    # Hide any unused subplots
    for idx in range(num_sensors, rows * cols):
        row, col = divmod(idx, cols)
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis("off")

    st.pyplot(fig)

    # Add "Turbofan Engines Lifetime" plot
    st.subheader("Turbofan Engines Lifetime")
    st.write("Visualizing the lifetime (maximum time cycles) of each unit:")

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

    # Correlation heatmap for the entire dataset
    st.subheader("Correlation Heatmap")
    st.write("Hover over the heatmap to see the exact correlation values.")

    # Calculate correlation matrix
    correlation_matrix = df_train.corr()

    # Create the heatmap using Plotly
    heatmap_fig = px.imshow(
        correlation_matrix,
        labels={'x': 'Features', 'y': 'Features', 'color': 'Correlation'},
        color_continuous_scale='Viridis',
        title=f"Correlation Heatmap for {selected_dataset}",
        width=900,  # Set heatmap width
        height=900  # Set heatmap height
    )

    # Display the heatmap
    st.plotly_chart(heatmap_fig)

