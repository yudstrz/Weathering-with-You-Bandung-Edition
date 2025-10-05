import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import joblib
import gzip
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Helper function for RMSE
def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Page configuration
st.set_page_config(
    page_title="Bandung Weather Dashboard",
    page_icon="🌤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .section-header {
        color: #1E3A8A;
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        font-weight: 600;
    }
    .prediction-box {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .info-box {
        background: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load Data with error handling
@st.cache_data
def load_data(path="weather_bandung_2020_2025_clean_data.xlsx"):
    try:
        df = pd.read_excel(path, parse_dates=["Date"])
        return df
    except FileNotFoundError:
        st.error(f"❌ File not found: {path}")
        return None

# Load or train model
@st.cache_resource
def load_or_train_model(df, model_path="rf_multi_weather_model_compressed.joblib.gz"):
    try:
        # Try to load the gzip model
        with gzip.open(model_path, "rb") as f:
            model = joblib.load(f)

        # Get feature and target names from the dataframe
        feature_names = df.select_dtypes(include=[np.number]).drop(
            columns=["T2M", "PRECTOTCORR", "RH2M"], errors='ignore'
        ).columns.tolist()
        target_names = ["T2M", "PRECTOTCORR", "RH2M"]

        # Predict for evaluation
        X_multi = df[feature_names]
        y_multi = df[target_names]
        X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)
        
        # Calculate metrics for each target
        mae_temp = mean_absolute_error(y_test.iloc[:, 0], y_pred[:, 0])
        rmse_temp = calculate_rmse(y_test.iloc[:, 0], y_pred[:, 0])
        r2_temp = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
        
        mae_rain = mean_absolute_error(y_test.iloc[:, 1], y_pred[:, 1])
        rmse_rain = calculate_rmse(y_test.iloc[:, 1], y_pred[:, 1])
        r2_rain = r2_score(y_test.iloc[:, 1], y_pred[:, 1])
        
        mae_hum = mean_absolute_error(y_test.iloc[:, 2], y_pred[:, 2])
        rmse_hum = calculate_rmse(y_test.iloc[:, 2], y_pred[:, 2])
        r2_hum = r2_score(y_test.iloc[:, 2], y_pred[:, 2])
        
        metrics = {
            "T2M": {"MAE": mae_temp, "RMSE": rmse_temp, "R2": r2_temp},
            "PRECTOTCORR": {"MAE": mae_rain, "RMSE": rmse_rain, "R2": r2_rain},
            "RH2M": {"MAE": mae_hum, "RMSE": rmse_hum, "R2": r2_hum}
        }

        st.sidebar.success("✅ Model loaded successfully from file")
        return model, feature_names, target_names, metrics

    except (FileNotFoundError, EOFError, OSError) as e:
        st.sidebar.warning(f"⚠ Model not found or failed to load ({str(e)}), training a new model...")

        X_multi = df.select_dtypes(include=[np.number]).drop(
            columns=["T2M", "PRECTOTCORR", "RH2M"], errors='ignore'
        )
        y_multi = df[["T2M", "PRECTOTCORR", "RH2M"]]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_multi, y_multi, test_size=0.2, random_state=42
        )

        # Train model
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
        with st.spinner("Training Random Forest model..."):
            model.fit(X_train, y_train)

        # Evaluation
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = calculate_rmse(y_test, y_pred)
        metrics = {'MAE': mae, 'RMSE': rmse} # Note: Simplified metrics for training case

        # Save gzip model
        with gzip.open(model_path, 'wb') as f:
            joblib.dump(model, f, compress=3)

        st.sidebar.success(f"✅ New model trained and saved to {model_path}")
        return model, X_multi.columns.tolist(), y_multi.columns.tolist(), metrics

df = load_data()

if df is None:
    st.stop()

# Load model
model, feature_names, target_names, metrics = load_or_train_model(df)

# Header
st.markdown('<h1 class="main-header">🌤 Bandung Weather & Climate Dashboard (2020–2025)</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar with improved styling
with st.sidebar:
    st.image("https://api.dicebear.com/7.x/shapes/svg?seed=weather", width=100)
    st.markdown("## 🎛 Dashboard Settings")
    
    # Year Filter
    st.markdown("### 📅 Period Filter")
    year_range = st.select_slider(
        "Year Range",
        options=sorted(df["Year"].unique()),
        value=(min(df["Year"].unique()), max(df["Year"].unique()))
    )
    
    # Month Filter
    month_filter = st.multiselect(
        "Select Month(s) (optional)",
        options=list(range(1, 13)),
        format_func=lambda x: datetime(2000, x, 1).strftime('%B')
    )
    
    st.markdown("---")
    st.markdown("### 📊 Visualization Options")
    show_trend = st.checkbox("Show Trendline", value=True)
    chart_type = st.radio("Chart Type", ["Interactive (Plotly)", "Static (Matplotlib)"])
    
    st.markdown("---")
    st.info("💡 Tip: Use the filters to analyze specific periods.")

# Filter data
df_filtered = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]
if month_filter:
    df_filtered = df_filtered[df_filtered["Month"].isin(month_filter)]

# Metrics Overview
st.markdown('<h2 class="section-header">📊 Statistical Summary</h2>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    avg_temp = df_filtered["T2M"].mean()
    st.metric(
        label="🌡 Average Temperature",
        value=f"{avg_temp:.1f}°C",
        delta=f"{avg_temp - df['T2M'].mean():.1f}°C"
    )

with col2:
    total_rain = df_filtered["PRECTOTCORR"].sum()
    st.metric(
        label="🌧 Total Rainfall",
        value=f"{total_rain:.0f} mm",
        delta=f"{(total_rain/len(df_filtered)*365):.0f} mm/year"
    )

with col3:
    avg_humidity = df_filtered["RH2M"].mean()
    st.metric(
        label="💧 Average Humidity",
        value=f"{avg_humidity:.1f}%"
    )

with col4:
    avg_wind = df_filtered["WS10M"].mean()
    st.metric(
        label="💨 Wind Speed",
        value=f"{avg_wind:.2f} m/s"
    )

with col5:
    avg_solar = df_filtered["ALLSKY_SFC_SW_DWN"].mean()
    st.metric(
        label="☀ Solar Radiation",
        value=f"{avg_solar:.1f} W/m²"
    )

st.markdown("---")

# Tabs for better content organization
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🌡 Temperature Analysis", 
    "🌧 Rainfall", 
    "💨 Wind & Energy", 
    "🌱 Agriculture", 
    "⚠ Extreme Weather",
    "🔮 Weather Prediction",
    "📖 Data Dictionary"
])

with tab1:
    st.markdown("### Temperature Trend Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if chart_type == "Interactive (Plotly)":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_filtered["Date"], 
                y=df_filtered["T2M"],
                mode='lines',
                name='Daily Temperature',
                line=dict(color='#EF4444', width=1),
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.1)'
            ))
            
            if show_trend:
                monthly = df_filtered.groupby(df_filtered["Date"].dt.to_period("M"))["T2M"].mean()
                fig.add_trace(go.Scatter(
                    x=monthly.index.to_timestamp(),
                    y=monthly.values,
                    mode='lines',
                    name='Monthly Trend',
                    line=dict(color='#1E3A8A', width=3)
                ))
            
            fig.update_layout(
                title="Daily Temperature in Bandung",
                xaxis_title="Date",
                yaxis_title="Temperature (°C)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df_filtered["Date"], df_filtered["T2M"], color='#EF4444', alpha=0.7)
            ax.set_ylabel("Temperature (°C)")
            ax.set_xlabel("Date")
            ax.set_title("Daily Temperature in Bandung")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    with col2:
        st.markdown("#### 📈 Temperature Statistics")
        temp_stats = df_filtered["T2M"].describe()
        st.write(f"Maximum: {temp_stats['max']:.1f}°C")
        st.write(f"Minimum: {temp_stats['min']:.1f}°C")
        st.write(f"Average: {temp_stats['mean']:.1f}°C")
        st.write(f"Standard Deviation: {temp_stats['std']:.2f}°C")
        
        # Monthly heatmap
        st.markdown("#### 🗓 Monthly Heatmap")
        pivot_data = df_filtered.groupby(["Year", "Month"])["T2M"].mean().reset_index()
        pivot_table = pivot_data.pivot(index="Month", columns="Year", values="T2M")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="RdYlBu_r", ax=ax, cbar_kws={'label': '°C'})
        ax.set_ylabel("Month")
        ax.set_xlabel("Year")
        st.pyplot(fig)

with tab2:
    st.markdown("### Rainfall Analysis")
    
    # Rainfall visualization
    if chart_type == "Interactive (Plotly)":
        fig = px.bar(
            df_filtered, 
            x="Date", 
            y="PRECTOTCORR",
            title="Daily Rainfall",
            labels={"PRECTOTCORR": "Rainfall (mm)", "Date": "Date"},
            color="PRECTOTCORR",
            color_continuous_scale="Blues"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(df_filtered["Date"], df_filtered["PRECTOTCORR"], color='#3B82F6', alpha=0.7)
        ax.set_ylabel("Rainfall (mm)")
        ax.set_xlabel("Date")
        ax.set_title("Daily Rainfall")
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
    
    # Seasonal analysis
    st.markdown("#### 📊 Seasonal Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_rain = df_filtered.groupby("Month")["PRECTOTCORR"].sum().reset_index()
        fig = px.bar(
            monthly_rain,
            x="Month",
            y="PRECTOTCORR",
            title="Total Rainfall per Month",
            labels={"PRECTOTCORR": "Rainfall (mm)", "Month": "Month"},
            color="PRECTOTCORR",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        yearly_rain = df_filtered.groupby("Year")["PRECTOTCORR"].sum().reset_index()
        fig = px.line(
            yearly_rain,
            x="Year",
            y="PRECTOTCORR",
            title="Annual Rainfall Trend",
            markers=True,
            labels={"PRECTOTCORR": "Rainfall (mm)", "Year": "Year"}
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Wind & Energy Potential Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💨 Wind Speed")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtered["Date"], y=df_filtered["WS2M"], name="2m", line=dict(color='#10B981')))
        fig.add_trace(go.Scatter(x=df_filtered["Date"], y=df_filtered["WS10M"], name="10m", line=dict(color='#3B82F6')))
        fig.add_trace(go.Scatter(x=df_filtered["Date"], y=df_filtered["WS50M"], name="50m", line=dict(color='#8B5CF6')))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Speed (m/s)",
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### ☀ Solar Radiation")
        fig = px.area(
            df_filtered,
            x="Date",
            y="ALLSKY_SFC_SW_DWN",
            title="Daily Solar Radiation",
            labels={"ALLSKY_SFC_SW_DWN": "Radiation (W/m²)", "Date": "Date"},
            color_discrete_sequence=["#F59E0B"]
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Renewable energy potential
    st.markdown("#### ⚡ Renewable Energy Potential")
    col1, col2 = st.columns(2)
    
    with col1:
        solar_yearly = df_filtered.groupby("Year")["ALLSKY_SFC_SW_DWN"].mean().reset_index()
        fig = px.bar(
            solar_yearly,
            x="Year",
            y="ALLSKY_SFC_SW_DWN",
            title="Solar Energy Potential (Annual Average)",
            color="ALLSKY_SFC_SW_DWN",
            color_continuous_scale="Oranges"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        wind_yearly = df_filtered.groupby("Year")["WS50M"].mean().reset_index()
        fig = px.bar(
            wind_yearly,
            x="Year",
            y="WS50M",
            title="Wind Energy Potential (Annual Average)",
            color="WS50M",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### Agricultural Recommendations")
    
    # Calculate recommendations
    rain_monthly = df_filtered.groupby(["Year", "Month"]).agg({
        "PRECTOTCORR": "sum",
        "T2M": "mean"
    }).reset_index()
    
    def get_planting_recommendation(row):
        if row["PRECTOTCORR"] > 150 and 20 <= row["T2M"] <= 30:
            return "🟢 Very Suitable for Planting"
        elif row["PRECTOTCORR"] > 100 and 18 <= row["T2M"] <= 32:
            return "🟡 Suitable for Planting"
        elif row["PRECTOTCORR"] < 50:
            return "🔴 Insufficient Water"
        else:
            return "🟠 Irrigation Needed"
    
    rain_monthly["Recommendation"] = rain_monthly.apply(get_planting_recommendation, axis=1)
    rain_monthly["Month_Name"] = rain_monthly["Month"].apply(lambda x: datetime(2000, x, 1).strftime('%B'))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📅 Planting Calendar")
        st.dataframe(
            rain_monthly[["Year", "Month_Name", "PRECTOTCORR", "T2M", "Recommendation"]].tail(24),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.markdown("#### 📊 Recommendation Distribution")
        recommendation_count = rain_monthly["Recommendation"].value_counts()
        fig = px.pie(
            values=recommendation_count.values,
            names=recommendation_count.index,
            title="Planting Condition Status"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.markdown("### Extreme Weather Detection")
    
    # Simple heat index calculation
    df_filtered["HeatIndex"] = 0.5 * (df_filtered["T2M"] + df_filtered["T2MDEW"])
    
    # Detect extreme weather
    extreme_heat = df_filtered[df_filtered["T2M"] > 35]
    extreme_rain = df_filtered[df_filtered["PRECTOTCORR"] > 50]
    high_wind = df_filtered[df_filtered["WS10M"] > 10]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔥 Extreme Heat Days", len(extreme_heat), f"{len(extreme_heat)/len(df_filtered)*100:.1f}%")
    with col2:
        st.metric("🌊 Heavy Rain Days", len(extreme_rain), f"{len(extreme_rain)/len(df_filtered)*100:.1f}%")
    with col3:
        st.metric("💨 High Wind Days", len(high_wind), f"{len(high_wind)/len(df_filtered)*100:.1f}%")
    
    # Visualize extreme weather
    st.markdown("#### 📍 Extreme Weather Events")
    
    tab_extreme1, tab_extreme2, tab_extreme3 = st.tabs(["Extreme Heat", "Heavy Rain", "High Wind"])
    
    with tab_extreme1:
        if len(extreme_heat) > 0:
            st.dataframe(
                extreme_heat[["Date", "T2M", "HeatIndex", "RH2M"]].sort_values("T2M", ascending=False),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No extreme heat events in the selected period")
    
    with tab_extreme2:
        if len(extreme_rain) > 0:
            st.dataframe(
                extreme_rain[["Date", "PRECTOTCORR", "T2M"]].sort_values("PRECTOTCORR", ascending=False),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No heavy rain events in the selected period")
    
    with tab_extreme3:
        if len(high_wind) > 0:
            st.dataframe(
                high_wind[["Date", "WS10M", "WS50M"]].sort_values("WS10M", ascending=False),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No high wind events in the selected period")

with tab6:
    st.markdown("### 🔮 Weather Prediction with Random Forest")
    
    # Model Information
    st.markdown("""
    <div style="
        background-color: black; 
        color: white; 
        padding: 15px; 
        border-radius: 8px;
        font-family: Arial, sans-serif;
    ">
        <h4>ℹ About the Prediction Model:</h4>
        <ul>
            <li><b>Algorithm</b>: Multi-target Random Forest Regressor</li>
            <li><b>Prediction Targets</b>: Temperature (T2M), Rainfall (PRECTOTCORR), Humidity (RH2M)</li>
            <li><b>Input Features</b>: Other weather parameters (wind, radiation, pressure, etc.)</li>
            <li><b>Model File</b>: rf_multi_weather_model_compressed.joblib.gz (compressed format)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance Metrics
    if metrics:
        st.markdown("#### 📊 Model Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("🌡 Temperature (T2M)")
            st.metric("RMSE", f"{metrics['T2M']['RMSE']:.3f}°C")
            st.metric("MAE", f"{metrics['T2M']['MAE']:.3f}°C")
            st.metric("R² Score", f"{metrics['T2M']['R2']:.3f}")
        
        with col2:
            st.markdown("🌧 Rainfall**")
            st.metric("RMSE", f"{metrics['PRECTOTCORR']['RMSE']:.3f} mm")
            st.metric("MAE", f"{metrics['PRECTOTCORR']['MAE']:.3f} mm")
            st.metric("R² Score", f"{metrics['PRECTOTCORR']['R2']:.3f}")
        
        with col3:
            st.markdown("💧 Humidity**")
            st.metric("RMSE", f"{metrics['RH2M']['RMSE']:.3f}%")
            st.metric("MAE", f"{metrics['RH2M']['MAE']:.3f}%")
            st.metric("R² Score", f"{metrics['RH2M']['R2']:.3f}")
    
    st.markdown("---")
    
    # Prediction Interface
    st.markdown("#### 🎯 Make a Prediction")
    
    prediction_mode = st.radio(
        "Select Prediction Mode",
        ["Predict from Historical Data", "Manual Parameter Input", "Batch Prediction (7 Days)"],
        horizontal=True
    )
    
    if prediction_mode == "Predict from Historical Data":
        st.markdown("Select a date from historical data to see predicted vs. actual values:")
        
        selected_date = st.date_input(
            "Select Date",
            value=df["Date"].max(),
            min_value=df["Date"].min().date(),
            max_value=df["Date"].max().date()
        )
        
        # Get data for the selected date
        selected_data = df[df["Date"] == pd.Timestamp(selected_date)]
        
        if len(selected_data) > 0:
            selected_data = selected_data.iloc[0]
            
            # Prepare input for prediction
            X_input = pd.DataFrame([selected_data[feature_names]])
            
            # Make prediction
            prediction = model.predict(X_input)[0]
            
            # Display results
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### 📍 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🌡 Temperature")
                st.markdown(f"Predicted: {prediction[0]:.2f}°C")
                st.markdown(f"Actual: {selected_data['T2M']:.2f}°C")
                diff_temp = prediction[0] - selected_data['T2M']
                st.markdown(f"Difference: {diff_temp:+.2f}°C")
            
            with col2:
                st.markdown("#### 🌧 Rainfall")
                st.markdown(f"Predicted: {prediction[1]:.2f} mm")
                st.markdown(f"Actual: {selected_data['PRECTOTCORR']:.2f} mm")
                diff_rain = prediction[1] - selected_data['PRECTOTCORR']
                st.markdown(f"Difference: {diff_rain:+.2f} mm")
            
            with col3:
                st.markdown("#### 💧 Humidity")
                st.markdown(f"Predicted: {prediction[2]:.2f}%")
                st.markdown(f"Actual: {selected_data['RH2M']:.2f}%")
                diff_hum = prediction[2] - selected_data['RH2M']
                st.markdown(f"Difference: {diff_hum:+.2f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Comparison visualization
            st.markdown("#### 📊 Comparison Visualization")
            
            comparison_data = pd.DataFrame({
                'Parameter': ['Temperature (°C)', 'Rainfall (mm)', 'Humidity (%)'],
                'Predicted': [prediction[0], prediction[1], prediction[2]],
                'Actual': [selected_data['T2M'], selected_data['PRECTOTCORR'], selected_data['RH2M']]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Predicted', x=comparison_data['Parameter'], y=comparison_data['Predicted'], marker_color='#6366f1'))
            fig.add_trace(go.Bar(name='Actual', x=comparison_data['Parameter'], y=comparison_data['Actual'], marker_color='#10b981'))
            fig.update_layout(barmode='group', height=400, title='Predicted vs. Actual Comparison')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Data not available for the selected date")
    
    elif prediction_mode == "Manual Parameter Input":
        st.markdown("Enter weather parameters for prediction:")
        
        col1, col2, col3 = st.columns(3)
        
        # Manual input for important features
        input_data = {}
        
        with col1:
            st.markdown("🌡 Temperature Parameters**")
            input_data['T2M_MAX'] = st.number_input("Max Temperature (°C)", value=30.0, min_value=15.0, max_value=40.0)
            input_data['T2M_MIN'] = st.number_input("Min Temperature (°C)", value=20.0, min_value=10.0, max_value=30.0)
            input_data['T2MDEW'] = st.number_input("Dew Point (°C)", value=22.0, min_value=10.0, max_value=30.0)
        
        with col2:
            st.markdown("💨 Wind Parameters**")
            input_data['WS2M'] = st.number_input("Wind Speed 2m (m/s)", value=2.0, min_value=0.0, max_value=20.0)
            input_data['WS10M'] = st.number_input("Wind Speed 10m (m/s)", value=3.0, min_value=0.0, max_value=25.0)
            input_data['WS50M'] = st.number_input("Wind Speed 50m (m/s)", value=5.0, min_value=0.0, max_value=30.0)
        
        with col3:
            st.markdown("☀ Radiation Parameters**")
            input_data['ALLSKY_SFC_SW_DWN'] = st.number_input("Solar Radiation (W/m²)", value=180.0, min_value=0.0, max_value=400.0)
            input_data['PS'] = st.number_input("Surface Pressure (kPa)", value=95.0, min_value=90.0, max_value=105.0)
        
        # Fill remaining features with mean values
        for feature in feature_names:
            if feature not in input_data:
                input_data[feature] = df[feature].mean()
        
        # Prediction button
        if st.button("🔮 Make Prediction", type="primary"):
            X_input = pd.DataFrame([input_data])[feature_names]
            prediction = model.predict(X_input)[0]
            
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### 📍 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🌡 Temperature")
                st.markdown(f"### {prediction[0]:.2f}°C")
                if prediction[0] > 30:
                    st.warning("⚠ High temperature")
                elif prediction[0] < 20:
                    st.info("❄ Cool temperature")
                else:
                    st.success("✅ Normal temperature")
            
            with col2:
                st.markdown("#### 🌧 Rainfall")
                st.markdown(f"### {prediction[1]:.2f} mm")
                if prediction[1] > 20:
                    st.warning("⚠ Heavy rain")
                elif prediction[1] > 5:
                    st.info("🌦 Light rain")
                else:
                    st.success("☀ Clear")
            
            with col3:
                st.markdown("#### 💧 Humidity")
                st.markdown(f"### {prediction[2]:.1f}%")
                if prediction[2] > 80:
                    st.info("💧 High humidity")
                elif prediction[2] < 60:
                    st.warning("🏜 Dry")
                else:
                    st.success("✅ Normal")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("#### 💡 Recommendations")
            recommendations = []
            
            if prediction[0] > 32:
                recommendations.append("🌡 High temperature - Avoid midday outdoor activities")
            if prediction[1] > 10:
                recommendations.append("☔ Chance of rain - Bring an umbrella")
            if prediction[2] > 85:
                recommendations.append("💧 High humidity - Air will feel stuffy")
            
            if recommendations:
                for rec in recommendations:
                    st.info(rec)
            else:
                st.success("✅ Good weather conditions for outdoor activities")
    
    else:  # Batch Prediction (7 Days)
        st.markdown("Forecast weather for the next 7 days based on recent data patterns:")
        
        if st.button("🔮 Generate 7-Day Forecast", type="primary"):
            # Get the last 7 days of data
            last_7_days = df.tail(7).copy()
            
            predictions_list = []
            dates_list = []
            
            for i in range(7):
                # Use the last day's data as a base
                base_data = last_7_days.iloc[-1]
                
                # Add small variations for simulation
                future_data = {}
                for feature in feature_names:
                    if feature in base_data.index:
                        # Add small noise
                        noise = np.random.normal(0, 0.02 * abs(base_data[feature]))
                        future_data[feature] = base_data[feature] + noise
                    else:
                        future_data[feature] = df[feature].mean()
                
                X_input = pd.DataFrame([future_data])[feature_names]
                prediction = model.predict(X_input)[0]
                
                future_date = last_7_days['Date'].iloc[-1] + timedelta(days=i+1)
                
                predictions_list.append({
                    'Date': future_date,
                    'Temperature (°C)': prediction[0],
                    'Rainfall (mm)': prediction[1],
                    'Humidity (%)': prediction[2]
                })
                dates_list.append(future_date)
            
            pred_df = pd.DataFrame(predictions_list)
            
            # Display prediction table
            st.markdown("#### 📅 7-Day Forecast Table")
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
            
            # Visualize predictions
            st.markdown("#### 📊 Forecast Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pred_df['Date'],
                    y=pred_df['Temperature (°C)'],
                    mode='lines+markers',
                    name='Temperature',
                    line=dict(color='#ef4444', width=3),
                    marker=dict(size=10)
                ))
                fig.update_layout(
                    title='7-Day Temperature Forecast',
                    xaxis_title='Date',
                    yaxis_title='Temperature (°C)',
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=pred_df['Date'],
                    y=pred_df['Rainfall (mm)'],
                    name='Rainfall',
                    marker_color='#3b82f6'
                ))
                fig.update_layout(
                    title='7-Day Rainfall Forecast',
                    xaxis_title='Date',
                    yaxis_title='Rainfall (mm)',
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.markdown("#### 📈 Forecast Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_temp = pred_df['Temperature (°C)'].mean()
                st.metric("Average Temperature", f"{avg_temp:.1f}°C")
            
            with col2:
                total_rain = pred_df['Rainfall (mm)'].sum()
                st.metric("Total Rainfall", f"{total_rain:.1f} mm")
            
            with col3:
                avg_humidity = pred_df['Humidity (%)'].mean()
                st.metric("Average Humidity", f"{avg_humidity:.1f}%")
            
            # Weather warnings
            st.markdown("#### ⚠ Weather Alerts")
            warnings = []
            
            if (pred_df['Temperature (°C)'] > 33).any():
                hot_days = (pred_df['Temperature (°C)'] > 33).sum()
                warnings.append(f"🌡 {hot_days} day(s) with very high temperatures (>33°C)")
            
            if (pred_df['Rainfall (mm)'] > 20).any():
                rainy_days = (pred_df['Rainfall (mm)'] > 20).sum()
                warnings.append(f"🌧 {rainy_days} day(s) with heavy rain (>20mm)")
            
            if warnings:
                for warning in warnings:
                    st.warning(warning)
            else:
                st.success("✅ No extreme weather alerts in the next 7 days")
    
    # Feature Importance
    st.markdown("---")
    st.markdown("#### 🎯 Feature Importance in the Model")
    
    try:
        # Check model type
        if isinstance(model, MultiOutputRegressor):
            # If MultiOutputRegressor, get from the first estimator
            if hasattr(model.estimators_[0], 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.estimators_[0].feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Top 10 Most Influential Features (for Temperature Prediction)',
                    color='Importance',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        elif hasattr(model, 'feature_importances_'):
            # If a regular RandomForest
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Most Influential Features',
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance is not available for this model type")
    except Exception as e:
        st.warning(f"Could not display feature importance: {str(e)}")
    
    # Download prediction results
    st.markdown("---")
    st.markdown("#### 💾 Export Results")
    
    if st.button("📥 Download Model Info"):
        model_info = {
            "Model": "Multi-target Random Forest",
            "Features": feature_names,
            "Targets": target_names,
            "Metrics": metrics
        }
        
        st.json(model_info)
        st.success("✅ Model information displayed above")

with tab7:
    st.markdown("### 📖 Data Dictionary & Documentation")
    
    st.markdown("""
    <div style="
        background-color: black; 
        color: white; 
        padding: 15px; 
        border-radius: 8px;
        font-family: Arial, sans-serif;
    ">
        <h4>ℹ About the Dataset</h4>
        <p>
            This dataset contains daily weather data for the city of Bandung, sourced from 
            <strong>NASA POWER</strong> (Prediction of Worldwide Energy Resources). It covers various 
            meteorological and solar radiation parameters from the year 2020 to 2025.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data categories
    st.markdown("### 📊 Parameter Categories")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3>🌡</h3>
            <h4>Temperature</h4>
            <p>5 Parameters</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3>💨</h3>
            <h4>Wind</h4>
            <p>3 Parameters</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3>☀</h3>
            <h4>Radiation</h4>
            <p>2 Parameters</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data dictionary table
    st.markdown("### 📋 Data Parameter Details")
    
    # Data dictionary
    data_dict = {
        "Column Name": [
            "Date", "City", "T2M", "T2M_MIN", "T2M_MAX", "T2MDEW", "T2MWET",
            "RH2M", "PRECTOTCORR", "WS2M", "WS10M", "WS50M", 
            "PS", "ALLSKY_SFC_SW_DWN", "TOA_SW_DWN"
        ],
        "Description": [
            "Observation date",
            "Location name (Bandung)",
            "Temperature at 2 Meters – average daily temperature at 2m",
            "Temperature at 2 Meters Minimum – minimum daily temperature at 2m",
            "Temperature at 2 Meters Maximum – maximum daily temperature at 2m",
            "Dew Point Temperature at 2 Meters – dew point temperature at 2m",
            "Wet Bulb Temperature at 2 Meters – wet bulb temperature (humidity indicator)",
            "Relative Humidity at 2 Meters – relative humidity at 2m",
            "Precipitation Corrected – total rainfall (data quality corrected)",
            "Wind Speed at 2 Meters – wind speed at 2m height",
            "Wind Speed at 10 Meters – wind speed at 10m height",
            "Wind Speed at 50 Meters – wind speed at 50m height",
            "Surface Pressure – air pressure at the surface",
            "All Sky Surface Shortwave Downward Irradiance – shortwave radiation reaching the surface",
            "Top of Atmosphere Shortwave Downward Irradiance – shortwave radiation at the top of the atmosphere"
        ],
        "Unit": [
            "-", "-", "°C", "°C", "°C", "°C", "°C",
            "%", "mm/day", "m/s", "m/s", "m/s",
            "kPa", "MJ/m²/day", "MJ/m²/day"
        ],
        "Category": [
            "Metadata", "Metadata", "Meteorology", "Meteorology", "Meteorology", 
            "Meteorology", "Meteorology", "Meteorology", "Meteorology",
            "Meteorology", "Meteorology", "Meteorology", "Meteorology",
            "Radiation", "Radiation"
        ],
        "Source": [
            "Metadata", "Metadata", "NASA POWER", "NASA POWER", "NASA POWER",
            "NASA POWER", "NASA POWER", "NASA POWER", "NASA POWER",
            "NASA POWER", "NASA POWER", "NASA POWER", "NASA POWER",
            "NASA POWER", "NASA POWER"
        ]
    }
    
    df_dict = pd.DataFrame(data_dict)
    
    # Filter by category
    category_filter = st.multiselect(
        "Filter by category:",
        options=["All", "Metadata", "Meteorology", "Radiation"],
        default=["All"]
    )
    
    if "All" not in category_filter and len(category_filter) > 0:
        df_dict_filtered = df_dict[df_dict["Category"].isin(category_filter)]
    else:
        df_dict_filtered = df_dict
    
    # Display table with styling
    st.dataframe(
        df_dict_filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Column Name": st.column_config.TextColumn("Column Name", width="medium"),
            "Description": st.column_config.TextColumn("Description", width="large"),
            "Unit": st.column_config.TextColumn("Unit", width="small"),
            "Category": st.column_config.TextColumn("Category", width="small"),
            "Source": st.column_config.TextColumn("Source", width="small")
        }
    )
    
    st.markdown("---")
    
    # Detailed category explanations
    st.markdown("### 🔍 Detailed Category Explanations")
    
    detail_tab1, detail_tab2, detail_tab3 = st.tabs([
        "🌡 Temperature Parameters", 
        "💨 Wind Parameters", 
        "☀ Radiation Parameters"
    ])
    
    with detail_tab1:
        st.markdown("""
        #### Temperature Parameters
        
        T2M (Temperature at 2 Meters)
        - Average daily temperature measured at 2 meters above ground level.
        - An international standard for meteorological measurement.
        - Used for: climate trend analysis, weather forecasting, agricultural planning.
        
        T2M_MIN & T2M_MAX
        - Minimum and maximum daily temperatures.
        - Important for: calculating the diurnal temperature range.
        - An indicator of thermal comfort and energy needs.
        
        T2MDEW (Dew Point Temperature)
        - The temperature at which air becomes saturated and dew begins to form.
        - An indicator of absolute humidity.
        - High values (>20°C) = humid and stuffy air.
        
        T2MWET (Wet Bulb Temperature)
        - The lowest temperature that can be reached through evaporation.
        - Used for: calculating the heat index and thermal comfort.
        - Important for: outdoor work safety and sports.
        """)
        
        # Example visualization
        if len(df_filtered) > 0:
            st.markdown("##### 📊 Example Temperature Distribution")
            fig = go.Figure()
            fig.add_trace(go.Box(y=df_filtered["T2M"], name="T2M", marker_color='#ef4444'))
            fig.add_trace(go.Box(y=df_filtered["T2M_MIN"], name="T2M_MIN", marker_color='#3b82f6'))
            fig.add_trace(go.Box(y=df_filtered["T2M_MAX"], name="T2M_MAX", marker_color='#f59e0b'))
            fig.update_layout(
                title="Temperature Parameter Distribution",
                yaxis_title="Temperature (°C)",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with detail_tab2:
        st.markdown("""
        #### Wind Speed Parameters
        
        WS2M (Wind Speed at 2 Meters)
        - Wind speed at 2 meters height.
        - Relevant for: human activities on the surface, agriculture, air pollution.
        - Usually lower due to friction with the surface.
        
        WS10M (Wind Speed at 10 Meters)
        - Wind speed at 10 meters height.
        - The standard for weather reporting and aviation.
        - Used for: estimating wind loads on buildings.
        
        WS50M (Wind Speed at 50 Meters)
        - Wind speed at 50 meters height.
        - Crucial for: wind turbine planning and renewable energy.
        - Higher values = greater wind energy potential.
        
        Wind Speed Scale:
        - 0-2 m/s: Calm to light breeze
        - 2-5 m/s: Gentle breeze
        - 5-10 m/s: Moderate wind
        - >10 m/s: Strong wind
        """)
        
        if len(df_filtered) > 0:
            st.markdown("##### 📊 Wind Speed Comparison")
            wind_avg = df_filtered[["WS2M", "WS10M", "WS50M"]].mean()
            fig = go.Figure(data=[
                go.Bar(
                    x=["WS2M (2m)", "WS10M (10m)", "WS50M (50m)"],
                    y=wind_avg.values,
                    marker_color=['#10b981', '#3b82f6', '#8b5cf6'],
                    text=[f"{v:.2f} m/s" for v in wind_avg.values],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="Average Wind Speed by Height",
                yaxis_title="Speed (m/s)",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with detail_tab3:
        st.markdown("""
        #### Solar Radiation Parameters
        
        ALLSKY_SFC_SW_DWN (All Sky Surface Shortwave Downward Irradiance)
        - Shortwave radiation that reaches the Earth's surface.
        - Accounts for all sky conditions (clear, cloudy, rainy).
        - Unit: MJ/m²/day, can be converted to kWh/m²/day.
        - Primary applications: 
          - Designing solar panel (PV) systems
          - Estimating solar energy production
          - Agricultural planning (photosynthesis)
        
        TOA_SW_DWN (Top of Atmosphere Shortwave Downward Irradiance)
        - Radiation that reaches the top of the atmosphere (before being absorbed/reflected).
        - The theoretical maximum radiation value.
        - The difference with ALLSKY_SFC_SW_DWN shows the atmospheric effect.
        
        Value Interpretation:
        - <3 MJ/m²/day: Very low (very bad weather/night)
        - 3-10 MJ/m²/day: Low (overcast)
        - 10-20 MJ/m²/day: Moderate (partly cloudy)
        - >20 MJ/m²/day: High (clear)
        
        Conversion:
        - 1 MJ/m²/day ≈ 0.278 kWh/m²/day
        - Example: 18 MJ/m²/day = ~5 kWh/m²/day (good for solar panels)
        """)
        
        if len(df_filtered) > 0:
            st.markdown("##### 📊 Solar Radiation Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    df_filtered,
                    x="ALLSKY_SFC_SW_DWN",
                    nbins=30,
                    title="Surface Radiation Distribution",
                    labels={"ALLSKY_SFC_SW_DWN": "Radiation (MJ/m²/day)"},
                    color_discrete_sequence=['#f59e0b']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                monthly_rad = df_filtered.groupby("Month")["ALLSKY_SFC_SW_DWN"].mean().reset_index()
                fig = px.line(
                    monthly_rad,
                    x="Month",
                    y="ALLSKY_SFC_SW_DWN",
                    title="Average Radiation per Month",
                    markers=True,
                    labels={"ALLSKY_SFC_SW_DWN": "Radiation (MJ/m²/day)", "Month": "Month"}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Additional information
    st.markdown("### 📚 Sources & References")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        🌐 NASA POWER
        - Website: [power.larc.nasa.gov](https://power.larc.nasa.gov)
        - Temporal Resolution: Daily
        - Spatial Resolution: 0.5° x 0.625°
        - Method: Assimilation of satellite and model data
        
        📖 Documentation
        - [NASA POWER Documentation](https://power.larc.nasa.gov/docs/)
        - [Data Access Guide](https://power.larc.nasa.gov/docs/services/api/)
        """)
    
    with col2:
        st.markdown("""
        🎯 Data Utility
        - ☀ Renewable energy planning
        - 🌱 Agricultural planting schedule optimization
        - 🏗 Energy-efficient building design
        - 🌡 Climate change analysis
        - 📊 Meteorological research
        
        ⚠ Important Notes
        - Data has undergone quality control.
        - PRECTOTCORR is the corrected precipitation data.
        - Minimal missing values (<1%).
        """)
    
    st.markdown("---")
    
    # Download data dictionary
    st.markdown("### 💾 Download Documentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_dict.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data Dictionary (CSV)",
            data=csv,
            file_name="data_dictionary_bandung_weather.csv",
            mime="text/csv",
            type="primary"
        )
    
    with col2:
        sample_data = df.head(100).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download Sample Data",
            data=sample_data,
            file_name="sample_data_bandung_weather.csv",
            mime="text/csv",
            type="secondary"
        )

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("📊 *Data Source*: NASA POWER")
with col2:
    st.markdown(f"📅 *Last Update*: {df['Date'].max().strftime('%d %B %Y')}")
with col3:
    st.markdown(f"📈 *Total Records*: {len(df_filtered):,}")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280; padding: 1rem; font-size: 0.9rem;'>
        Made with using Streamlit | Bandung Weather Dashboard | Powered by Random Forest ML
        <br><br>
        <b>Team:</b> Failed to Graduate on Time
        <br>
        Dharmmesti Mayda • Wahyudi • Geryl • Sayyidina Gusti • Satrio
    </div>
    """,
    unsafe_allow_html=True
)
