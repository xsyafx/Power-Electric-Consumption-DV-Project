import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import kagglehub
import os

st.set_page_config(page_title="Energy Efficiency Project", layout="wide")

# Load dataset from Kaggle function.
def parse_energy_data(path_or_buffer):
    """Parses the semicolon-separated data file."""
    df = pd.read_csv(
        path_or_buffer,
        sep=";",
        na_values="?",
        low_memory=False
    )
    # Basic cleaning & types
    df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    num_cols = [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["DateTime"]).sort_values("DateTime")
    return df

@st.cache_data(show_spinner="Downloading dataset from Kaggle...")
def load_from_kaggle():
    """
    Downloads the dataset from Kaggle using kagglehub, caches it,
    and then loads it into a DataFrame.
    """
    # Download the dataset files. This returns a path to a local directory.
    dataset_path = kagglehub.dataset_download("uciml/electric-power-consumption-data-set")

    # Construct the full path to the specific file within the downloaded directory.
    file_path = os.path.join(dataset_path, "household_power_consumption.txt")

    # Use the parsing function to load the data.
    df = parse_energy_data(file_path)
    return df

def daily_mean(df):
    """Calculates the daily mean for numeric columns."""
    num_cols = [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
    ]
    return df.set_index("DateTime")[num_cols].resample("D").mean().dropna(how="all")

# ---------------------------
# Main App Logic
# ---------------------------

# Attempt to load the data automatically from Kaggle
df = None
try:
    df = load_from_kaggle()
except Exception as e:
    st.error(f"Failed to load data from Kaggle: {e}")
    st.info("""
        **To fix this, please ensure you have authenticated with Kaggle:**
        1. Make sure you have a `kaggle.json` API token. You can create one from your Kaggle account settings page under the "API" section.
        2. Place the `kaggle.json` file in the correct directory (e.g., `~/.kaggle/` on Linux/macOS or `C:\\Users\\<Your-Username>\\.kaggle\\` on Windows).
        
        For more details, see the Kaggle API documentation.
    """)

# ---------------------------
# Sidebar Nav
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Data Sources",
        "Data Preparation & Cleaning",
        "Analysis (Trends)",
        "Dashboard (Interactive)",
        "Recommendations",
        "Outcomes / Results",
    ]
)
st.sidebar.markdown("---")
if df is not None:
    st.sidebar.success(f"Successfully loaded {df.shape[0]:,} rows from Kaggle.")
else:
    st.sidebar.warning("Data could not be loaded. See error message.")


# ---------------------------
# Pages
# ---------------------------
if page == "Overview":
    st.title("🏠 Analyzing Household Energy Usage Patterns for Smarter Consumption")
    st.write("""
**Problem:** Households face high electricity bills due to inefficient usage and lack of visibility into when/how energy is consumed.

**Objectives**
1. Analyze historical household electricity consumption and detect patterns
2. Identify peak usage hours and highlight potential energy-saving opportunities
3. Build an interactive Streamlit dashboard to visualize trends and provide recommendations
    """)

    st.markdown("**What this app contains:**")
    st.markdown("- Data Sources & Documentation\n- Data Preparation & Cleaning\n- Analysis & Trends\n- Interactive Dashboard\n- Recommendations & Outcomes")

    if df is not None:
        st.info(f"Loaded data: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
    else:
        st.warning("Data is not loaded. Please resolve the Kaggle authentication issue to proceed.")

elif page == "Data Sources":
    st.title("📂 Data Sources")
    st.write("""
1) **Household Electric Power Consumption** — UCI/Kaggle
    - **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)
    - Minute-level readings, 2006–2010
    - Columns: Global active/reactive power, Voltage, Intensity, Sub-metering (1–3)

""")
    st.caption("Data is now loaded directly from the source using the `kagglehub` library.")

    if df is not None:
        st.subheader("Raw Sample (first 5 rows)")
        st.dataframe(df.head())

elif page == "Data Preparation & Cleaning":
    st.title("🧹 Data Preparation & Cleaning")
    st.write("""
**Steps performed**
- Download dataset directly from Kaggle
- Read semicolon-separated TXT, treat `?` as missing
- Combine **Date + Time → DateTime**
- Convert numeric fields to float
- Drop rows with invalid timestamps
- Create daily aggregates for simpler plotting
""")

    if df is None:
        st.warning("Load the dataset via the sidebar to see this section.")
    else:
        st.subheader("Preview After Cleaning")
        st.dataframe(df.head())

        daily = daily_mean(df)
        st.subheader("Daily Aggregated Sample")
        st.dataframe(daily.head())

        # Optional download buttons
        st.download_button(
            "Download cleaned minute-level CSV",
            df.to_csv(index=False).encode("utf-8"),
            "cleaned_minute_level.csv",
            "text/csv",
        )
        st.download_button(
            "Download daily mean CSV",
            daily.reset_index().to_csv(index=False).encode("utf-8"),
            "daily_mean_energy.csv",
            "text/csv",
        )

elif page == "Analysis (Trends)":
    st.title("📈 Analysis (Trends)")
    
    if df is None:
        st.warning("Data is not loaded. Please resolve the Kaggle authentication issue to proceed.")
    else:
        st.header("Interactive Hourly Mean Trends")
        st.write("""
            Select a metric from the dropdown menu to visualize its mean (average) value,
            resampled by **hour**. This interactive chart uses Plotly.
        """)
        
        cols_to_plot = [
            "Global_active_power", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
        ]
        
        selected_col = st.selectbox("Select a metric to visualize:", cols_to_plot)
        
        if selected_col:
            try:
                with st.spinner(f"Resampling {selected_col} by hour... This may take a moment."):
                    # Resample the selected column to hourly frequency
                    df_hourly = df.set_index("DateTime")[selected_col].resample('H').mean().reset_index()
                    df_hourly.columns = ['DateTime', 'Mean Value'] # Rename for clarity
                
                st.success(f"Successfully resampled {selected_col}.")

                # Determine the correct Y-axis label based on the selected column
                if selected_col == "Global_active_power":
                    y_label = "Mean Power (kW)"
                elif "Sub_metering" in selected_col:
                    y_label = "Mean Energy (Watt-hour)"
                else:
                    y_label = f"Mean {selected_col}" # Fallback

                # Visualize resampled data using Plotly
                fig = px.line(
                    df_hourly, 
                    x='DateTime', 
                    y='Mean Value', 
                    title=f"Hourly Mean for {selected_col}"
                )
                
                fig.update_layout(
                    xaxis_title="Date / Time",
                    yaxis_title=y_label # Use the new dynamic label
                )
                
                # Display the interactive Plotly chart
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred while generating the trend plot for {selected_col}: {e}")

elif page == "Dashboard (Interactive)":
    st.title("📊 Dashboard (Interactive)")
    st.info("This page is under construction. Come back later for interactive charts!")
    # You could add Plotly charts or other interactive elements here

elif page == "Recommendations":
    st.title("💡 Recommendations")
    st.info("This page is under construction.")

elif page == "Outcomes / Results":
    st.title("🏆 Outcomes / Results")
    st.info("This page is under construction.")




