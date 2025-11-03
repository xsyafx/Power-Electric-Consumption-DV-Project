import streamlit as st
import pandas as pd
import numpy as np
import os
import kagglehub
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from math import sqrt
import plotly.express as px
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Energy Efficiency Project", layout="wide")


# Data Loading Functions from Kaggle
@st.cache_data(show_spinner="Downloading Household dataset...")
def get_kaggle_filepath():
    """
    Downloads the dataset from Kaggle using kagglehub and
    returns the local file path.
    """
    # Download the dataset files. This returns a path to a local directory.
    dataset_path = kagglehub.dataset_download("uciml/electric-power-consumption-data-set")

    # Construct the full path to the specific file
    file_path = os.path.join(dataset_path, "household_power_consumption.txt")
    return file_path



# Loads Raw Data Function from Kaggle
@st.cache_data(show_spinner="Loading Household raw data preview...")
def load_raw_preview(path_or_buffer):
    """Loads the raw Household data with minimal parsing, just for preview."""
    df_raw = pd.read_csv(
        path_or_buffer,
        sep=";",
        low_memory=False,
    )
    return df_raw



# Loads and Cleans the Data from Kaggle
@st.cache_data(show_spinner="Parsing and cleaning Household data...")
def load_and_parse_data(path_or_buffer):
    """
    Loads and parses the Household data, combining date/time, setting index,
    and converting types.
    """
    # Load data, combining Date and Time, and setting ? as NaN
    df = pd.read_csv(
        path_or_buffer,
        sep=';',
        parse_dates={'Datetime': ['Date', 'Time']}, # Combine Date/Time into 'Datetime'
        infer_datetime_format=True,
        low_memory=False,
        na_values='?'  # Treat '?' as a missing value
    )
    
    # Set the new Datetime column as the index
    df.set_index('Datetime', inplace=True)
    
    return df


@st.cache_data(show_spinner="Loading cleaned Household data preview...")
def load_cleaned_data(file_path):
    """
    Loads the parsed Household data and then drops all rows with any
    missing values.
    """
    # Load the parsed dataset (uses cache from load_and_parse_data)
    dataset = load_and_parse_data(file_path)
    
    # Drop all rows that have ANY missing value
    cleaned_dataset = dataset.dropna()
    return cleaned_dataset


# Data Loading Functions from Malaysian Parquet
URL_DATA = 'https://storage.data.gov.my/energy/electricity_consumption.parquet'

@st.cache_data(show_spinner="Loading Local Domestic data raw preview...")
def load_parquet_raw_preview(url):
    """Loads the raw parquet data, just for preview."""
    try:
        df_raw = pd.read_parquet(url)
        return df_raw
    except Exception as e:
        st.error(f"Error loading raw data from Parquet URL: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner="Parsing and cleaning Local Domestic data...")
def load_parquet_and_parse_data(url):
    """
    Loads and parses the Parquet data, sets index,
    and converts types.
    """
    # Load data from Parquet URL
    df = pd.read_parquet(url)
    
    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        st.error("The required 'date' column was not found in the Local Domestic dataset.")
        return pd.DataFrame() # Return empty df
    
    # Set the new 'date' column as the index
    df.set_index('date', inplace=True)
    
    # Convert only 'consumption' column to numeric
    if 'consumption' in df.columns:
        df['consumption'] = pd.to_numeric(df['consumption'], errors='coerce')
    else:
        st.error("The required 'consumption' column was not found in the Malaysian dataset.")
    
    return df


@st.cache_data(show_spinner="Loading Local Domestic data...")
def load_parquet_cleaned_data(url):
    """
    Loads and parses the Parquet data.
    """
    # Load the parsed dataset (uses cache from load_parquet_and_parse_data)
    dataset = load_parquet_and_parse_data(url)
    
    # Per request, do not drop NA rows for the Local Domestic dataset
    cleaned_dataset = dataset
    return cleaned_dataset


# Main App Logic

# Load Household Data 
df_raw = None
df_parsed = None
df_cleaned = None
file_path = None

try:
    file_path = get_kaggle_filepath()
    # Load all Household data versions (will be fast due to caching)
    df_raw = load_raw_preview(file_path)
    df_parsed = load_and_parse_data(file_path) # Parsed, but with NaNs
    df_cleaned = load_cleaned_data(file_path) # Parsed and NaNs dropped
    
except Exception as e:
    st.error(f"Failed to load data from Kaggle: {e}")
    st.info("""
        **To fix this, please ensure you have authenticated with Kaggle:**
        1. Make sure you have a `kaggle.json` API token.
        2. Place the `kaggle.json` file in the correct directory (e.g., `~/.kaggle/`).
        """)

# Load Local Domestic Parquet Data (for previews) 
df_parquet_raw = None
df_parquet_parsed = None
df_parquet_cleaned = None
try:
    df_parquet_raw = load_parquet_raw_preview(URL_DATA)
    df_parquet_parsed = load_parquet_and_parse_data(URL_DATA)
    df_parquet_cleaned = load_parquet_cleaned_data(URL_DATA)
except Exception as e:
    st.warning(f"Could not load Malaysian Parquet data: {e}")


# Sidebar Nav
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Data Sources",
        "Data Preparation & Cleaning",
        "Analysis (Trends)",
        "Data Testing and Evaluation"
        #"Recommendations",
        #"Outcomes / Results"
    ]
)
# st.sidebar.markdown("---")
# # Sidebar status refers to the MAIN (Kaggle) dataset
# if df_cleaned is not None:
#     st.sidebar.success(f"Kaggle data: {df_cleaned.shape[0]:,} cleaned rows.")
#     st.sidebar.info(f"Original Kaggle rows: {df_parsed.shape[0]:,}")
# else:
#     st.sidebar.warning("Kaggle data could not be loaded. See error message.")

# if df_parquet_cleaned is not None:
#     st.sidebar.success(f"Malaysian data: {df_parquet_cleaned.shape[0]:,} cleaned rows.")
# else:
#     st.sidebar.warning("Malaysian data could not be loaded.")


# Pages
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
    st.markdown("- Data Sources & Documentation\n- Data Preparation & Cleaning\n- Analysis & Trends\n- Recommendations & Outcomes")

    # if df_cleaned is not None:
    #     st.info(f"Loaded and cleaned Household data: **{df_cleaned.shape[0]:,} rows × {df_cleaned.shape[1]} columns**")
    # else:
    #     st.warning("Household data is not loaded. Please resolve the Kaggle authentication issue to proceed.")

elif page == "Data Sources":
    st.title("📂 Data Sources")
    
    # --- Kaggle Section ---
    st.header("1) Household Electric Power Consumption — Kaggle")
    st.write("""
    - **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)
    - Minute-level readings, 2006–2010
    - Columns: Global active/reactive power, Voltage, Intensity, Sub-metering (1–3)
    """)

    if df_raw is not None:
        st.subheader("Raw Sample")
        st.dataframe(df_raw.head(1000))
    else:
        st.warning("Kaggle data not loaded. Cannot display raw sample.")

    # --- Parquet Section ---
    st.markdown("---")
    st.header("2) Local Domestic Electricity Consumption")
    st.write("""
    - **Source:** [data.gov.my Parquet File](https://storage.data.gov.my/energy/electricity_consumption.parquet)
    - **Description:** Official data on local domestic electricity consumption.
    - **Format:** Parquet
    """)
    # st.caption("This data is loaded in addition to the Kaggle data for preview.")

    if df_parquet_raw is not None and not df_parquet_raw.empty:
        st.subheader("Raw Sample")
        st.dataframe(df_parquet_raw.head(100))
    else:
        st.warning("Local Domestic Parquet data not loaded. Cannot display raw sample.")


elif page == "Data Preparation & Cleaning":
    st.title("🧹 Data Preparation & Cleaning")

    # --- Household Section ---
    st.header("1. Household Power Consumption")
    st.write("""
**Steps performed**
- Download dataset directly from Kaggle using `kagglehub`
- Read semicolon-separated TXT
- Combine **Date + Time to Datetime Column**
- Treat `?` as missing values (`na_values='?'`)
- Set **Datetime** as the DataFrame index
- **Dropped all rows** containing any missing values
""")

    if df_cleaned is None or df_parsed is None:
        st.warning("Household dataset not loaded.")
    else:
        st.subheader("Cleaned Household Data")
        st.dataframe(df_cleaned.head(1000))
        st.write(f"**Original rows:** {len(df_parsed):,}")
        st.write(f"**Rows after cleaning:** {len(df_cleaned):,}")
        st.write(f"**Total rows removed:** {len(df_parsed) - len(df_cleaned):,}")

    # --- Parquet Section ---
    st.markdown("---")
    st.header("2. Local Domestic Electricity Consumption")
    st.write("""
**Steps performed**
- Download dataset directly from the `data.gov.my` URL
- Read Parquet file into a pandas DataFrame
- Convert **'date' column to a Datetime object**
- Set **'date'** as the DataFrame index
- Convert only the **'consumption'** column to numeric, coercing errors to `NaN`
- **Round 'consumption' data to 2 decimal places**
""")

    if df_parquet_cleaned is None or df_parquet_parsed is None or df_parquet_cleaned.empty:
        st.warning("Local Domestic dataset not loaded.")
    else:
        st.subheader("Cleaned Local Domestic Data")
        df_parquet_cleaned['consumption'] = df_parquet_cleaned['consumption'].round(2)
        st.dataframe(df_parquet_cleaned.head(100))
        st.write(f"**Original rows:** {len(df_parquet_parsed):,}")
        st.write(f"**Rows after cleaning:** {len(df_parquet_cleaned):,}")
        st.write(f"**Total rows removed:** {len(df_parquet_parsed) - len(df_parquet_cleaned):,}")


elif page == "Analysis (Trends)":
    st.title("📈 Analysis (Trends)")

    # --- Local Domestic Section ---
    st.markdown("---")
    st.header("1. Local Domestic Sector Deep-Dive")
    # st.info("This section analyzes the **Local Domestic** dataset.")

    if df_parquet_cleaned is None or df_parquet_cleaned.empty:
        st.warning("Local Domestic data is not loaded. Cannot display this dashboard.")
    else:
        st.write("""
        This is a linked chart focused only on the 'Local Domestic' sector.
        
        Use the top chart to click and drag a time period.
        The bottom chart will update to show the daily data for that selected period.
        """)

        try:
            # Prepare the data
            df_malay_altair = df_parquet_cleaned.reset_index()
            
            # Filter this data for 'Local Domestic' 
            df_localDomestic = df_malay_altair[df_malay_altair['sector'] == 'local_domestic'].copy()
            
            # 3. Check if have any data left
            if df_localDomestic.empty:
                st.warning("No data was found for the 'Local Domestic' sector.")
            
            else:
                # Create brush selection
                brush = alt.selection_interval(encodings=['x'])

                # Create the TOP chart (Monthly Trend)
                # Use the pre-filtered 'df_localDomestic' dataframe
                top_chart = alt.Chart(df_localDomestic).mark_line().encode(
                    # 'yearmonth(date):T' automatically groups the data by month
                    x=alt.X('yearmonth(date):T', axis=alt.Axis(title='Long-Term Trend (Monthly)')),
                    # 'aggregate="mean"' calculates the monthly average
                    y=alt.Y('consumption:Q', aggregate='mean', title='Mean Daily Consumption'),
                    tooltip=[
                        alt.Tooltip('yearmonth(date):T', title='Month'),
                        alt.Tooltip('mean(consumption):Q', title='Mean Consumption', format=',.2f')
                    ]
                ).properties(
                    title='Monthly Average Local Domestic Consumption (Click and Drag to Select)'
                ).add_selection(
                    brush  # Add the brush selection to this chart
                )

                # Create the BOTTOM chart (Daily Data)
                # Use the pre-filtered 'df_localDomestic' dataframe
                bottom_chart = alt.Chart(df_localDomestic).mark_line(point=True).encode(
                    x=alt.X('date:T', axis=alt.Axis(title='Date')),
                    y=alt.Y('consumption:Q', title='Daily Consumption'),
                    tooltip=[
                        alt.Tooltip('date:T', title='Date'),
                        alt.Tooltip('consumption:Q', title='Consumption', format=',.2f')
                    ]
                ).properties(
                    title='Daily Local Domestic Consumption (Filtered by selection)'
                ).transform_filter(
                    brush  # MAGIC: This chart is filtered by the brush from the top chart
                ).interactive() # Make the bottom chart zoomable

                # Combine and display the charts
                linked_chart = top_chart & bottom_chart
                
                st.altair_chart(linked_chart, use_container_width=True)
            
        except Exception as e:
            st.error(f"An error occurred while generating the local domestic linked chart: {e}")
            st.exception(e)
    
    
    # --- Household Section ---
    if df_cleaned is None:
        st.markdown("---")
        st.header("2. Household: Analysis")
        st.warning("Household data is not loaded. Please resolve the Kaggle authentication issue to proceed.")
    else:
        
        # Long_term Trends (2006-2010)
        st.markdown("---")
        st.header("2. Household: Long-Term Consumption Trends")
        st.write("""
            This chart shows the mean (average) consumption over the entire
            four-year period.
        """)
        
        
        # Column selection
        cols_to_plot = [
            "Global_active_power", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
        ]
        selected_col = st.selectbox("Select a metric to visualize:", cols_to_plot)
        
        # Frequency selection
        time_frequency = st.radio(
            "Select time frequency:",
            ("Hourly", "Daily", "Monthly"),
            index=0 # Default to Hourly
        )
        
        # Map frequency name to pandas resample code
        frequency_map = {
            "Hourly": "H",
            "Daily": "D",
            "Monthly": "M"
        }
        resample_code = frequency_map[time_frequency]
        
        if selected_col and time_frequency:
            try:
                with st.spinner(f"Resampling {selected_col} by {time_frequency.lower()}... This may take a moment."):
                    # Resample the selected column to the chosen frequency
                    df_resampled = df_cleaned[selected_col].resample(resample_code).mean().reset_index()
                    df_resampled.columns = ['Datetime', 'Mean Value'] # Rename for clarity
                
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
                    df_resampled, # Use the new resampled df
                    x='Datetime', # Updated column name
                    y='Mean Value', 
                    title=f"{time_frequency} Mean for {selected_col}"
                )
                
                fig.update_layout(
                    xaxis_title="Date / Time",
                    yaxis_title=y_label
                )
                
                # Display the interactive Plotly chart
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred while generating the trend plot for {selected_col}: {e}")

        st.markdown("---")
        st.header("3. Household: Hourly & Daily Patterns")
        
        # Chart: Average by Hour of Day 
        st.subheader("Average Consumption by Hour of Day")

        @st.cache_data(show_spinner="Calculating hourly average...")
        def get_hourly_avg(df):
            # Create an 'hour' column
            df_hourly = df.copy()
            df_hourly['hour'] = df_hourly.index.hour
            # Group by the hour and get the mean
            hourly_avg = df_hourly.groupby('hour')[['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].mean()
            return hourly_avg.reset_index()

        try:
            # Get the aggregated data
            df_hourly_avg = get_hourly_avg(df_cleaned)
            
            # Melt for easy plotting with Plotly
            df_hourly_melted = df_hourly_avg.melt(
                id_vars='hour', 
                value_vars=['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'],
                var_name='Metric',
                value_name='Mean Power'
            )
            
            # Create the plot
            fig_hourly = px.line(
                df_hourly_melted,
                x='hour',
                y='Mean Power',
                color='Metric',
                labels={'hour': 'Hour of Day (0-23)', 'Mean Power': 'Mean Power (kW / Watt-hour)'}
            )
            fig_hourly.update_layout(xaxis=dict(tickmode='linear')) # Ensure all 24 hours are shown
            st.plotly_chart(fig_hourly, use_container_width=True)
            
        except Exception as e:
            st.error(f"An error occurred while generating the hourly average plot: {e}")

        # Chart: Average by Day of Week 
        st.subheader("Average Consumption by Day of Week")

        @st.cache_data(show_spinner="Calculating daily average...")
        def get_daily_avg(df):
            # Create a 'day_of_week' column
            df_daily = df.copy()
            df_daily['day_of_week'] = df_daily.index.day_name()
            # Group by the day and get the mean
            daily_avg = df_daily.groupby('day_of_week')[['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].mean()
            
            # Sort the index to be in the correct weekday order
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            daily_avg = daily_avg.reindex(days_order)
            return daily_avg.reset_index()

        try:
            # Get the aggregated data
            df_daily_avg = get_daily_avg(df_cleaned)
            
            # Melt for easy plotting
            df_daily_melted = df_daily_avg.melt(
                id_vars='day_of_week', 
                value_vars=['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'],
                var_name='Metric',
                value_name='Mean Power'
            )
            
            # Create the plot
            fig_daily = px.bar(
                df_daily_melted,
                x='day_of_week',
                y='Mean Power',
                color='Metric',
                barmode='group', # Group bars side-by-side
                labels={'day_of_week': 'Day of Week', 'Mean Power': 'Mean Power (kW / Watt-hour)'}
            )
            st.plotly_chart(fig_daily, use_container_width=True)
            
        except Exception as e:
            st.error(f"An error occurred while generating the daily average plot: {e}")
        # --- MOVED CODE ENDS HERE ---


        # Variable Correlation 
        st.markdown("---")
        st.header("4. Household: Variable Correlation Heatmap")
        
        try:
            with st.spinner("Generating correlation heatmap..."):
                # Create the matplotlib figure and axes
                fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 7))

                # Define specific columns
                heatmap_cols = ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
                        
                # Calculate the correlation matrix for selected columns
                corr_matrix = df_cleaned[heatmap_cols].corr()
                        
                # Generate heatmap
                sns.heatmap(
                    corr_matrix, 
                    annot=True,      # Show the correlation values
                    cmap='coolwarm', # Use the 'coolwarm' color map
                    ax=ax_heatmap,   # Plot on the created axes
                    fmt=".2f"        # Format annotations to 2 decimal places
                )
                        
                ax_heatmap.set_title("Correlation Matrix of Energy Variables")
                        
                # Display
                st.pyplot(fig_heatmap)
        
        except Exception as e:
            st.error(f"An error occurred while generating the correlation heatmap: {e}")
            st.exception(e)

elif page == "Data Testing and Evaluation":
    st.title("📊 Data Testing and Evaluation Dashboard")
    st.write("""
    This section focuses on testing and evaluating predictive models using a 70/30 
    train-test split. We compare the accuracy of ARIMA, SARIMA, Random Forest, and 
    LSTM models based on Root Mean Square Error (RMSE) and Mean Absolute Error (MAE).
    """)
    
    # --- Data Preparation ---
    if df_cleaned is None or df_cleaned.empty:
        st.warning("Dataset not available. Please ensure data is loaded correctly in previous steps.")
    else:
        st.subheader("1️⃣ Data Split (70% Train, 30% Test)")

        df_eval = df_cleaned.copy()
        df_eval = df_eval[['Global_active_power']].resample('D').mean().dropna()

        train_size = int(len(df_eval) * 0.7)
        train, test = df_eval[:train_size], df_eval[train_size:]

        y_train, y_test = train['Global_active_power'], test['Global_active_power']

        st.write(f"✅ Training samples: {len(train)} | Testing samples: {len(test)}")

        # --- ARIMA Model ---
        st.subheader("2️⃣ ARIMA Model")
        arima_model = ARIMA(y_train, order=(2,1,2)).fit()
        arima_forecast = arima_model.forecast(steps=len(y_test))
        arima_rmse = np.sqrt(mean_squared_error(y_test, arima_forecast))
        arima_mae = mean_absolute_error(y_test, arima_forecast)
        st.write(f"**RMSE:** {arima_rmse:.4f}, **MAE:** {arima_mae:.4f}")

        # --- SARIMA Model ---
        st.subheader("3️⃣ SARIMA Model")
        sarima_model = SARIMAX(y_train, order=(2,1,2), seasonal_order=(1,1,1,12)).fit(disp=False)
        sarima_forecast = sarima_model.forecast(steps=len(y_test))
        sarima_rmse = np.sqrt(mean_squared_error(y_test, sarima_forecast))
        sarima_mae = mean_absolute_error(y_test, sarima_forecast)
        st.write(f"**RMSE:** {sarima_rmse:.4f}, **MAE:** {sarima_mae:.4f}")

        # --- Random Forest Model ---
        st.subheader("4️⃣ Random Forest Model")
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        X_train = np.arange(len(y_train)).reshape(-1,1)
        X_test = np.arange(len(y_train), len(y_train) + len(y_test)).reshape(-1,1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_mae = mean_absolute_error(y_test, rf_pred)
        st.write(f"**RMSE:** {rf_rmse:.4f}, **MAE:** {rf_mae:.4f}")

        # --- LSTM Model ---
        st.subheader("5️⃣ LSTM Model")
        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(df_eval[['Global_active_power']])
        train_scaled, test_scaled = y_scaled[:train_size], y_scaled[train_size:]

        X_train_lstm, y_train_lstm = [], []
        for i in range(3, len(train_scaled)):
            X_train_lstm.append(train_scaled[i-3:i])
            y_train_lstm.append(train_scaled[i])
        X_train_lstm, y_train_lstm = np.array(X_train_lstm), np.array(y_train_lstm)

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=32, verbose=0)

        X_test_lstm = []
        for i in range(len(test_scaled) - 3):
            X_test_lstm.append(test_scaled[i:i+3])
        X_test_lstm = np.array(X_test_lstm)

        lstm_pred = model.predict(X_test_lstm)
        lstm_pred_inv = scaler.inverse_transform(lstm_pred)
        y_test_inv = scaler.inverse_transform(test_scaled[3:])

        lstm_rmse = np.sqrt(mean_squared_error(y_test_inv, lstm_pred_inv))
        lstm_mae = mean_absolute_error(y_test_inv, lstm_pred_inv)
        st.write(f"**RMSE:** {lstm_rmse:.4f}, **MAE:** {lstm_mae:.4f}")

        # --- Summary Table ---
        st.markdown("### 📋 Model Comparison Summary")
        summary_df = pd.DataFrame({
            "Model": ["ARIMA", "SARIMA", "Random Forest", "LSTM"],
            "RMSE": [arima_rmse, sarima_rmse, rf_rmse, lstm_rmse],
            "MAE": [arima_mae, sarima_mae, rf_mae, lstm_mae]
        })
        st.dataframe(summary_df)

        # --- Visualization Section ---
        st.markdown("---")
        st.subheader("📊 Model Visualizations")

        visual_df = pd.DataFrame({
            "Actual": y_test.values,
            "ARIMA": arima_forecast.values,
            "SARIMA": sarima_forecast.values,
            "Random Forest": rf_pred
        }, index=df_eval.index[train_size:])

        # ARIMA Plot
        st.subheader("🔹 ARIMA Forecast")
        fig_arima = px.line(visual_df, y=["Actual", "ARIMA"], title="Actual vs Predicted (ARIMA)")
        st.plotly_chart(fig_arima, use_container_width=True)

        # SARIMA Plot
        st.subheader("🔹 SARIMA Forecast")
        fig_sarima = px.line(visual_df, y=["Actual", "SARIMA"], title="Actual vs Predicted (SARIMA)")
        st.plotly_chart(fig_sarima, use_container_width=True)

        # Random Forest Plot
        st.subheader("🔹 Random Forest Forecast")
        fig_rf = px.line(visual_df, y=["Actual", "Random Forest"], title="Actual vs Predicted (Random Forest)")
        st.plotly_chart(fig_rf, use_container_width=True)

        # LSTM Plot
        st.subheader("🔹 LSTM Forecast")
        lstm_visual_df = pd.DataFrame({
            "Actual": y_test_inv.flatten()[:len(lstm_pred_inv)],
            "LSTM": lstm_pred_inv.flatten()
        }, index=df_eval.index[-len(lstm_pred_inv):])

        fig_lstm = px.line(lstm_visual_df, y=["Actual", "LSTM"], title="Actual vs Predicted (LSTM)")
        st.plotly_chart(fig_lstm, use_container_width=True)

#elif page == "Dashboard (Interactive)":
#    st.title("📊 Dashboard (Interactive)")
#    st.info("This page is under construction.")

#elif page == "Recommendations":
#    st.title("💡 Recommendations")
#    st.info("This page is under construction.")

#elif page == "Outcomes / Results":
#   st.title("🏆 Outcomes / Results")
#     st.info("This page is under construction.")
