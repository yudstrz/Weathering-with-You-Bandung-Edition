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

# Helper function untuk RMSE
def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Cuaca Bandung",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
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

# Load Data dengan error handling
@st.cache_data
def load_data(path="weather_bandung_2020_2025_clean_data.xlsx"):
    try:
        df = pd.read_excel(path, parse_dates=["Date"])
        return df
    except FileNotFoundError:
        st.error(f"❌ File tidak ditemukan: {path}")
        return None

# Load atau train model
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================
# Fungsi helper RMSE
# =========================
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# =========================
# Fungsi load dataset
# =========================
@st.cache_data
def load_data(path="weather_bandung_2020_2025_clean_data.xlsx"):
    df = pd.read_excel(path)
    df = df.dropna()
    return df

# =========================
# Fungsi load atau train model (versi sama seperti kode asli)
# =========================
@st.cache_resource
def load_or_train_model(df, model_path="rf_multi_weather_model_compressed.joblib.gz"):
    try:
        # Coba load model gzip
        with gzip.open(model_path, "rb") as f:
            model = joblib.load(f)

        # Ambil feature dan target names dari dataframe
        feature_names = df.select_dtypes(include=[np.number]).drop(
            columns=["T2M", "PRECTOTCORR", "RH2M"], errors='ignore'
        ).columns.tolist()
        target_names = ["T2M", "PRECTOTCORR", "RH2M"]

# Prediksi untuk evaluasi
        X_multi = df[feature_names]
        y_multi = df[target_names]
        X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)
        
        # Hitung metrics untuk setiap target
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

        st.sidebar.success("✅ Model berhasil dimuat dari file")
        return model, feature_names, target_names, metrics

    except (FileNotFoundError, EOFError, OSError) as e:
        st.sidebar.warning(f"⚠️ Model tidak ditemukan atau gagal load ({str(e)}), melatih model baru...")

        X_multi = df.select_dtypes(include=[np.number]).drop(
            columns=["T2M", "PRECTOTCORR", "RH2M"], errors='ignore'
        )
        y_multi = df[["T2M", "PRECTOTCORR", "RH2M"]]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_multi, y_multi, test_size=0.2, random_state=42
        )

        # Train model persis seperti kode asli
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
        with st.spinner("Melatih model Random Forest..."):
            model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = calculate_rmse(y_test, y_pred)
        metrics = {'MAE': mae, 'RMSE': rmse}

        # Simpan model gzip
        with gzip.open(model_path, 'wb') as f:
            joblib.dump(model, f, compress=3)

        st.sidebar.success(f"✅ Model baru berhasil dilatih dan disimpan ke {model_path}")
        return model, X_multi.columns.tolist(), y_multi.columns.tolist(), metrics

df = load_data()

if df is None:
    st.stop()

# Load model
model, feature_names, target_names, metrics = load_or_train_model(df)

# Header
st.markdown('<h1 class="main-header">🌤️ Dashboard Cuaca & Iklim Bandung (2000–2025)</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar dengan styling lebih baik
with st.sidebar:
    st.image("https://api.dicebear.com/7.x/shapes/svg?seed=weather", width=100)
    st.markdown("## 🎛️ Pengaturan Dashboard")
    
    # Filter Tahun
    st.markdown("### 📅 Filter Periode")
    year_range = st.select_slider(
        "Rentang Tahun",
        options=sorted(df["Year"].unique()),
        value=(min(df["Year"].unique()), max(df["Year"].unique()))
    )
    
    # Filter Bulan
    month_filter = st.multiselect(
        "Pilih Bulan (opsional)",
        options=list(range(1, 13)),
        format_func=lambda x: datetime(2000, x, 1).strftime('%B')
    )
    
    st.markdown("---")
    st.markdown("### 📊 Opsi Visualisasi")
    show_trend = st.checkbox("Tampilkan Trendline", value=True)
    chart_type = st.radio("Tipe Chart", ["Interaktif (Plotly)", "Statis (Matplotlib)"])
    
    st.markdown("---")
    st.info("💡 **Tips**: Gunakan filter untuk menganalisis periode spesifik")

# Filter data
df_filtered = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]
if month_filter:
    df_filtered = df_filtered[df_filtered["Month"].isin(month_filter)]

# Metrics Overview
st.markdown('<h2 class="section-header">📊 Ringkasan Statistik</h2>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    avg_temp = df_filtered["T2M"].mean()
    st.metric(
        label="🌡️ Suhu Rata-rata",
        value=f"{avg_temp:.1f}°C",
        delta=f"{avg_temp - df['T2M'].mean():.1f}°C"
    )

with col2:
    total_rain = df_filtered["PRECTOTCORR"].sum()
    st.metric(
        label="🌧️ Total Curah Hujan",
        value=f"{total_rain:.0f} mm",
        delta=f"{(total_rain/len(df_filtered)*365):.0f} mm/tahun"
    )

with col3:
    avg_humidity = df_filtered["RH2M"].mean()
    st.metric(
        label="💧 Kelembaban Rata-rata",
        value=f"{avg_humidity:.1f}%"
    )

with col4:
    avg_wind = df_filtered["WS10M"].mean()
    st.metric(
        label="💨 Kecepatan Angin",
        value=f"{avg_wind:.2f} m/s"
    )

with col5:
    avg_solar = df_filtered["ALLSKY_SFC_SW_DWN"].mean()
    st.metric(
        label="☀️ Radiasi Matahari",
        value=f"{avg_solar:.1f} W/m²"
    )

st.markdown("---")

# Tabs untuk organisasi konten yang lebih baik
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🌡️ Analisis Suhu", 
    "🌧️ Curah Hujan", 
    "💨 Angin & Energi", 
    "🌱 Pertanian", 
    "⚠️ Cuaca Ekstrem",
    "🔮 Prediksi Cuaca",
    "📖 Kamus Data"
])

with tab1:
    st.markdown("### Analisis Tren Suhu")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if chart_type == "Interaktif (Plotly)":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_filtered["Date"], 
                y=df_filtered["T2M"],
                mode='lines',
                name='Suhu Harian',
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
                    name='Tren Bulanan',
                    line=dict(color='#1E3A8A', width=3)
                ))
            
            fig.update_layout(
                title="Suhu Harian Bandung",
                xaxis_title="Tanggal",
                yaxis_title="Suhu (°C)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df_filtered["Date"], df_filtered["T2M"], color='#EF4444', alpha=0.7)
            ax.set_ylabel("Suhu (°C)")
            ax.set_xlabel("Tanggal")
            ax.set_title("Suhu Harian Bandung")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    with col2:
        st.markdown("#### 📈 Statistik Suhu")
        temp_stats = df_filtered["T2M"].describe()
        st.write(f"**Maksimum**: {temp_stats['max']:.1f}°C")
        st.write(f"**Minimum**: {temp_stats['min']:.1f}°C")
        st.write(f"**Rata-rata**: {temp_stats['mean']:.1f}°C")
        st.write(f"**Standar Deviasi**: {temp_stats['std']:.2f}°C")
        
        # Heatmap bulanan
        st.markdown("#### 🗓️ Heatmap Bulanan")
        pivot_data = df_filtered.groupby(["Year", "Month"])["T2M"].mean().reset_index()
        pivot_table = pivot_data.pivot(index="Month", columns="Year", values="T2M")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="RdYlBu_r", ax=ax, cbar_kws={'label': '°C'})
        ax.set_ylabel("Bulan")
        ax.set_xlabel("Tahun")
        st.pyplot(fig)

with tab2:
    st.markdown("### Analisis Curah Hujan")
    
    # Visualisasi curah hujan
    if chart_type == "Interaktif (Plotly)":
        fig = px.bar(
            df_filtered, 
            x="Date", 
            y="PRECTOTCORR",
            title="Curah Hujan Harian",
            labels={"PRECTOTCORR": "Curah Hujan (mm)", "Date": "Tanggal"},
            color="PRECTOTCORR",
            color_continuous_scale="Blues"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(df_filtered["Date"], df_filtered["PRECTOTCORR"], color='#3B82F6', alpha=0.7)
        ax.set_ylabel("Curah Hujan (mm)")
        ax.set_xlabel("Tanggal")
        ax.set_title("Curah Hujan Harian")
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
    
    # Analisis musiman
    st.markdown("#### 📊 Distribusi Musiman")
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_rain = df_filtered.groupby("Month")["PRECTOTCORR"].sum().reset_index()
        fig = px.bar(
            monthly_rain,
            x="Month",
            y="PRECTOTCORR",
            title="Total Curah Hujan per Bulan",
            labels={"PRECTOTCORR": "Curah Hujan (mm)", "Month": "Bulan"},
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
            title="Tren Tahunan Curah Hujan",
            markers=True,
            labels={"PRECTOTCORR": "Curah Hujan (mm)", "Year": "Tahun"}
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Analisis Angin & Potensi Energi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💨 Kecepatan Angin")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtered["Date"], y=df_filtered["WS2M"], name="2m", line=dict(color='#10B981')))
        fig.add_trace(go.Scatter(x=df_filtered["Date"], y=df_filtered["WS10M"], name="10m", line=dict(color='#3B82F6')))
        fig.add_trace(go.Scatter(x=df_filtered["Date"], y=df_filtered["WS50M"], name="50m", line=dict(color='#8B5CF6')))
        fig.update_layout(
            xaxis_title="Tanggal",
            yaxis_title="Kecepatan (m/s)",
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### ☀️ Radiasi Matahari")
        fig = px.area(
            df_filtered,
            x="Date",
            y="ALLSKY_SFC_SW_DWN",
            title="Radiasi Matahari Harian",
            labels={"ALLSKY_SFC_SW_DWN": "Radiasi (W/m²)", "Date": "Tanggal"},
            color_discrete_sequence=["#F59E0B"]
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Potensi energi terbarukan
    st.markdown("#### ⚡ Potensi Energi Terbarukan")
    col1, col2 = st.columns(2)
    
    with col1:
        solar_yearly = df_filtered.groupby("Year")["ALLSKY_SFC_SW_DWN"].mean().reset_index()
        fig = px.bar(
            solar_yearly,
            x="Year",
            y="ALLSKY_SFC_SW_DWN",
            title="Potensi Energi Solar (Rata-rata Tahunan)",
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
            title="Potensi Energi Angin (Rata-rata Tahunan)",
            color="WS50M",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### Rekomendasi Pertanian")
    
    # Hitung rekomendasi
    rain_monthly = df_filtered.groupby(["Year", "Month"]).agg({
        "PRECTOTCORR": "sum",
        "T2M": "mean"
    }).reset_index()
    
    def get_planting_recommendation(row):
        if row["PRECTOTCORR"] > 150 and 20 <= row["T2M"] <= 30:
            return "🟢 Sangat Cocok Tanam"
        elif row["PRECTOTCORR"] > 100 and 18 <= row["T2M"] <= 32:
            return "🟡 Cocok Tanam"
        elif row["PRECTOTCORR"] < 50:
            return "🔴 Kurang Air"
        else:
            return "🟠 Perlu Irigasi"
    
    rain_monthly["Rekomendasi"] = rain_monthly.apply(get_planting_recommendation, axis=1)
    rain_monthly["Bulan"] = rain_monthly["Month"].apply(lambda x: datetime(2000, x, 1).strftime('%B'))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📅 Kalender Tanam")
        st.dataframe(
            rain_monthly[["Year", "Bulan", "PRECTOTCORR", "T2M", "Rekomendasi"]].tail(24),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.markdown("#### 📊 Distribusi Rekomendasi")
        recommendation_count = rain_monthly["Rekomendasi"].value_counts()
        fig = px.pie(
            values=recommendation_count.values,
            names=recommendation_count.index,
            title="Status Kondisi Tanam"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.markdown("### Deteksi Cuaca Ekstrem")
    
    # Hitung heat index sederhana
    df_filtered["HeatIndex"] = 0.5 * (df_filtered["T2M"] + df_filtered["T2MDEW"])
    
    # Deteksi cuaca ekstrem
    extreme_heat = df_filtered[df_filtered["T2M"] > 35]
    extreme_rain = df_filtered[df_filtered["PRECTOTCORR"] > 50]
    high_wind = df_filtered[df_filtered["WS10M"] > 10]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔥 Hari Panas Ekstrem", len(extreme_heat), f"{len(extreme_heat)/len(df_filtered)*100:.1f}%")
    with col2:
        st.metric("🌊 Hari Hujan Lebat", len(extreme_rain), f"{len(extreme_rain)/len(df_filtered)*100:.1f}%")
    with col3:
        st.metric("💨 Hari Angin Kencang", len(high_wind), f"{len(high_wind)/len(df_filtered)*100:.1f}%")
    
    # Visualisasi cuaca ekstrem
    st.markdown("#### 📍 Kejadian Cuaca Ekstrem")
    
    tab_extreme1, tab_extreme2, tab_extreme3 = st.tabs(["Panas Ekstrem", "Hujan Lebat", "Angin Kencang"])
    
    with tab_extreme1:
        if len(extreme_heat) > 0:
            st.dataframe(
                extreme_heat[["Date", "T2M", "HeatIndex", "RH2M"]].sort_values("T2M", ascending=False),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Tidak ada kejadian panas ekstrem pada periode yang dipilih")
    
    with tab_extreme2:
        if len(extreme_rain) > 0:
            st.dataframe(
                extreme_rain[["Date", "PRECTOTCORR", "T2M"]].sort_values("PRECTOTCORR", ascending=False),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Tidak ada kejadian hujan lebat pada periode yang dipilih")
    
    with tab_extreme3:
        if len(high_wind) > 0:
            st.dataframe(
                high_wind[["Date", "WS10M", "WS50M"]].sort_values("WS10M", ascending=False),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Tidak ada kejadian angin kencang pada periode yang dipilih")

with tab6:
    st.markdown("### 🔮 Prediksi Cuaca dengan Random Forest")
    
    # Informasi Model
    st.markdown("""
    <div style="
        background-color: black; 
        color: white; 
        padding: 15px; 
        border-radius: 8px;
        font-family: Arial, sans-serif;
    ">
        <h4>ℹ️ Tentang Model Prediksi:</h4>
        <ul>
            <li><b>Algoritma</b>: Multi-target Random Forest Regressor</li>
            <li><b>Target Prediksi</b>: Suhu (T2M), Curah Hujan (PRECTOTCORR), Kelembaban (RH2M)</li>
            <li><b>Fitur Input</b>: Parameter cuaca lainnya (angin, radiasi, tekanan, dll)</li>
            <li><b>File Model</b>: rf_multi_weather_model_compressed.joblib.gz (compressed format)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance Metrics
    if metrics:
        st.markdown("#### 📊 Performa Model")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🌡️ Suhu (T2M)**")
            st.metric("RMSE", f"{metrics['T2M']['RMSE']:.3f}°C")
            st.metric("MAE", f"{metrics['T2M']['MAE']:.3f}°C")
            st.metric("R² Score", f"{metrics['T2M']['R2']:.3f}")
        
        with col2:
            st.markdown("**🌧️ Curah Hujan**")
            st.metric("RMSE", f"{metrics['PRECTOTCORR']['RMSE']:.3f} mm")
            st.metric("MAE", f"{metrics['PRECTOTCORR']['MAE']:.3f} mm")
            st.metric("R² Score", f"{metrics['PRECTOTCORR']['R2']:.3f}")
        
        with col3:
            st.markdown("**💧 Kelembaban**")
            st.metric("RMSE", f"{metrics['RH2M']['RMSE']:.3f}%")
            st.metric("MAE", f"{metrics['RH2M']['MAE']:.3f}%")
            st.metric("R² Score", f"{metrics['RH2M']['R2']:.3f}")
    
    st.markdown("---")
    
    # Prediksi Interface
    st.markdown("#### 🎯 Lakukan Prediksi")
    
    prediction_mode = st.radio(
        "Pilih Mode Prediksi",
        ["Prediksi dari Data Historis", "Input Manual Parameter", "Prediksi Batch (7 Hari)"],
        horizontal=True
    )
    
    if prediction_mode == "Prediksi dari Data Historis":
        st.markdown("Pilih tanggal dari data historis untuk melihat prediksi vs nilai aktual:")
        
        selected_date = st.date_input(
            "Pilih Tanggal",
            value=df["Date"].max(),
            min_value=df["Date"].min().date(),
            max_value=df["Date"].max().date()
        )
        
        # Ambil data untuk tanggal yang dipilih
        selected_data = df[df["Date"] == pd.Timestamp(selected_date)]
        
        if len(selected_data) > 0:
            selected_data = selected_data.iloc[0]
            
            # Persiapkan input untuk prediksi
            X_input = pd.DataFrame([selected_data[feature_names]])
            
            # Lakukan prediksi
            prediction = model.predict(X_input)[0]
            
            # Tampilkan hasil
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### 📍 Hasil Prediksi")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🌡️ Suhu")
                st.markdown(f"**Prediksi**: {prediction[0]:.2f}°C")
                st.markdown(f"**Aktual**: {selected_data['T2M']:.2f}°C")
                diff_temp = prediction[0] - selected_data['T2M']
                st.markdown(f"**Selisih**: {diff_temp:+.2f}°C")
            
            with col2:
                st.markdown("#### 🌧️ Curah Hujan")
                st.markdown(f"**Prediksi**: {prediction[1]:.2f} mm")
                st.markdown(f"**Aktual**: {selected_data['PRECTOTCORR']:.2f} mm")
                diff_rain = prediction[1] - selected_data['PRECTOTCORR']
                st.markdown(f"**Selisih**: {diff_rain:+.2f} mm")
            
            with col3:
                st.markdown("#### 💧 Kelembaban")
                st.markdown(f"**Prediksi**: {prediction[2]:.2f}%")
                st.markdown(f"**Aktual**: {selected_data['RH2M']:.2f}%")
                diff_hum = prediction[2] - selected_data['RH2M']
                st.markdown(f"**Selisih**: {diff_hum:+.2f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualisasi perbandingan
            st.markdown("#### 📊 Visualisasi Perbandingan")
            
            comparison_data = pd.DataFrame({
                'Parameter': ['Suhu (°C)', 'Curah Hujan (mm)', 'Kelembaban (%)'],
                'Prediksi': [prediction[0], prediction[1], prediction[2]],
                'Aktual': [selected_data['T2M'], selected_data['PRECTOTCORR'], selected_data['RH2M']]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Prediksi', x=comparison_data['Parameter'], y=comparison_data['Prediksi'], marker_color='#6366f1'))
            fig.add_trace(go.Bar(name='Aktual', x=comparison_data['Parameter'], y=comparison_data['Aktual'], marker_color='#10b981'))
            fig.update_layout(barmode='group', height=400, title='Perbandingan Prediksi vs Aktual')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Data tidak tersedia untuk tanggal yang dipilih")
    
    elif prediction_mode == "Input Manual Parameter":
        st.markdown("Masukkan parameter cuaca untuk prediksi:")
        
        col1, col2, col3 = st.columns(3)
        
        # Input manual untuk fitur-fitur penting
        input_data = {}
        
        with col1:
            st.markdown("**🌡️ Parameter Suhu**")
            input_data['T2M_MAX'] = st.number_input("Suhu Maksimum (°C)", value=30.0, min_value=15.0, max_value=40.0)
            input_data['T2M_MIN'] = st.number_input("Suhu Minimum (°C)", value=20.0, min_value=10.0, max_value=30.0)
            input_data['T2MDEW'] = st.number_input("Titik Embun (°C)", value=22.0, min_value=10.0, max_value=30.0)
        
        with col2:
            st.markdown("**💨 Parameter Angin**")
            input_data['WS2M'] = st.number_input("Kecepatan Angin 2m (m/s)", value=2.0, min_value=0.0, max_value=20.0)
            input_data['WS10M'] = st.number_input("Kecepatan Angin 10m (m/s)", value=3.0, min_value=0.0, max_value=25.0)
            input_data['WS50M'] = st.number_input("Kecepatan Angin 50m (m/s)", value=5.0, min_value=0.0, max_value=30.0)
        
        with col3:
            st.markdown("**☀️ Parameter Radiasi**")
            input_data['ALLSKY_SFC_SW_DWN'] = st.number_input("Radiasi Matahari (W/m²)", value=180.0, min_value=0.0, max_value=400.0)
            input_data['PS'] = st.number_input("Tekanan Permukaan (kPa)", value=95.0, min_value=90.0, max_value=105.0)
        
        # Isi fitur yang tersisa dengan nilai rata-rata
        for feature in feature_names:
            if feature not in input_data:
                input_data[feature] = df[feature].mean()
        
        # Tombol prediksi
        if st.button("🔮 Lakukan Prediksi", type="primary"):
            X_input = pd.DataFrame([input_data])[feature_names]
            prediction = model.predict(X_input)[0]
            
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### 📍 Hasil Prediksi")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🌡️ Suhu")
                st.markdown(f"### {prediction[0]:.2f}°C")
                if prediction[0] > 30:
                    st.warning("⚠️ Suhu tinggi")
                elif prediction[0] < 20:
                    st.info("❄️ Suhu sejuk")
                else:
                    st.success("✅ Suhu normal")
            
            with col2:
                st.markdown("#### 🌧️ Curah Hujan")
                st.markdown(f"### {prediction[1]:.2f} mm")
                if prediction[1] > 20:
                    st.warning("⚠️ Hujan lebat")
                elif prediction[1] > 5:
                    st.info("🌦️ Hujan ringan")
                else:
                    st.success("☀️ Cerah")
            
            with col3:
                st.markdown("#### 💧 Kelembaban")
                st.markdown(f"### {prediction[2]:.1f}%")
                if prediction[2] > 80:
                    st.info("💧 Lembab tinggi")
                elif prediction[2] < 60:
                    st.warning("🏜️ Kering")
                else:
                    st.success("✅ Normal")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Rekomendasi
            st.markdown("#### 💡 Rekomendasi")
            recommendations = []
            
            if prediction[0] > 32:
                recommendations.append("🌡️ Suhu tinggi - Hindari aktivitas outdoor siang hari")
            if prediction[1] > 10:
                recommendations.append("☔ Kemungkinan hujan - Bawa payung")
            if prediction[2] > 85:
                recommendations.append("💧 Kelembaban tinggi - Udara terasa pengap")
            
            if recommendations:
                for rec in recommendations:
                    st.info(rec)
            else:
                st.success("✅ Kondisi cuaca baik untuk aktivitas outdoor")
    
    else:  # Prediksi Batch (7 Hari)
        st.markdown("Prediksi cuaca untuk 7 hari ke depan berdasarkan pola data terakhir:")
        
        if st.button("🔮 Generate Prediksi 7 Hari", type="primary"):
            # Ambil data 7 hari terakhir
            last_7_days = df.tail(7).copy()
            
            predictions_list = []
            dates_list = []
            
            for i in range(7):
                # Gunakan data hari terakhir sebagai basis
                base_data = last_7_days.iloc[-1]
                
                # Tambahkan variasi kecil untuk simulasi
                future_data = {}
                for feature in feature_names:
                    if feature in base_data.index:
                        # Tambahkan noise kecil
                        noise = np.random.normal(0, 0.02 * abs(base_data[feature]))
                        future_data[feature] = base_data[feature] + noise
                    else:
                        future_data[feature] = df[feature].mean()
                
                X_input = pd.DataFrame([future_data])[feature_names]
                prediction = model.predict(X_input)[0]
                
                future_date = last_7_days['Date'].iloc[-1] + timedelta(days=i+1)
                
                predictions_list.append({
                    'Tanggal': future_date,
                    'Suhu (°C)': prediction[0],
                    'Curah Hujan (mm)': prediction[1],
                    'Kelembaban (%)': prediction[2]
                })
                dates_list.append(future_date)
            
            pred_df = pd.DataFrame(predictions_list)
            
            # Tampilkan tabel prediksi
            st.markdown("#### 📅 Tabel Prediksi 7 Hari")
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
            
            # Visualisasi prediksi
            st.markdown("#### 📊 Visualisasi Prediksi")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pred_df['Tanggal'],
                    y=pred_df['Suhu (°C)'],
                    mode='lines+markers',
                    name='Suhu',
                    line=dict(color='#ef4444', width=3),
                    marker=dict(size=10)
                ))
                fig.update_layout(
                    title='Prediksi Suhu 7 Hari',
                    xaxis_title='Tanggal',
                    yaxis_title='Suhu (°C)',
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=pred_df['Tanggal'],
                    y=pred_df['Curah Hujan (mm)'],
                    name='Curah Hujan',
                    marker_color='#3b82f6'
                ))
                fig.update_layout(
                    title='Prediksi Curah Hujan 7 Hari',
                    xaxis_title='Tanggal',
                    yaxis_title='Curah Hujan (mm)',
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistik
            st.markdown("#### 📈 Ringkasan Prediksi")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_temp = pred_df['Suhu (°C)'].mean()
                st.metric("Suhu Rata-rata", f"{avg_temp:.1f}°C")
            
            with col2:
                total_rain = pred_df['Curah Hujan (mm)'].sum()
                st.metric("Total Curah Hujan", f"{total_rain:.1f} mm")
            
            with col3:
                avg_humidity = pred_df['Kelembaban (%)'].mean()
                st.metric("Kelembaban Rata-rata", f"{avg_humidity:.1f}%")
            
            # Peringatan cuaca
            st.markdown("#### ⚠️ Peringatan Cuaca")
            warnings = []
            
            if (pred_df['Suhu (°C)'] > 33).any():
                hot_days = (pred_df['Suhu (°C)'] > 33).sum()
                warnings.append(f"🌡️ {hot_days} hari dengan suhu sangat tinggi (>33°C)")
            
            if (pred_df['Curah Hujan (mm)'] > 20).any():
                rainy_days = (pred_df['Curah Hujan (mm)'] > 20).sum()
                warnings.append(f"🌧️ {rainy_days} hari dengan hujan lebat (>20mm)")
            
            if warnings:
                for warning in warnings:
                    st.warning(warning)
            else:
                st.success("✅ Tidak ada peringatan cuaca ekstrem dalam 7 hari ke depan")
    
    # Feature Importance
    st.markdown("---")
    st.markdown("#### 🎯 Pentingnya Fitur dalam Model")
    
    try:
        # Cek tipe model
        from sklearn.multioutput import MultiOutputRegressor
        
        if isinstance(model, MultiOutputRegressor):
            # Jika MultiOutputRegressor, ambil dari estimator pertama
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
                    title='Top 10 Fitur Paling Berpengaruh (untuk Prediksi Suhu)',
                    color='Importance',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        elif hasattr(model, 'feature_importances_'):
            # Jika RandomForest biasa
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Fitur Paling Berpengaruh',
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance tidak tersedia untuk model ini")
    except Exception as e:
        st.warning(f"Tidak dapat menampilkan feature importance: {str(e)}")
    
    # Download hasil prediksi
    st.markdown("---")
    st.markdown("#### 💾 Export Hasil")
    
    if st.button("📥 Download Model Info"):
        model_info = {
            "Model": "Multi-target Random Forest",
            "Features": feature_names,
            "Targets": target_names,
            "Metrics": metrics
        }
        
        st.json(model_info)
        st.success("✅ Informasi model ditampilkan di atas")

with tab7:
    st.markdown("### 📖 Kamus Data & Dokumentasi")
    
    st.markdown("""
    <div style="
        background-color: black; 
        color: white; 
        padding: 15px; 
        border-radius: 8px;
        font-family: Arial, sans-serif;
    ">
        <h4>ℹ️ Tentang Dataset</h4>
        <p>
            Dataset ini berisi data cuaca harian untuk kota Bandung yang bersumber dari 
            <strong>NASA POWER</strong> (Prediction of Worldwide Energy Resources), mencakup berbagai 
            parameter meteorologi dan radiasi matahari dari tahun 2000 hingga 2025.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Kategori data
    st.markdown("### 📊 Kategori Parameter")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3>🌡️</h3>
            <h4>Suhu</h4>
            <p>5 Parameter</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3>💨</h3>
            <h4>Angin</h4>
            <p>3 Parameter</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3>☀️</h3>
            <h4>Radiasi</h4>
            <p>2 Parameter</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabel kamus data
    st.markdown("### 📋 Detail Parameter Data")
    
    # Data dictionary
    data_dict = {
        "Nama Kolom": [
            "Date", "City", "T2M", "T2M_MIN", "T2M_MAX", "T2MDEW", "T2MWET",
            "RH2M", "PRECTOTCORR", "WS2M", "WS10M", "WS50M", 
            "PS", "ALLSKY_SFC_SW_DWN", "TOA_SW_DWN"
        ],
        "Deskripsi": [
            "Tanggal observasi",
            "Nama lokasi (Bandung)",
            "Temperature at 2 Meters – suhu rata-rata harian di 2m",
            "Temperature at 2 Meters Minimum – suhu minimum harian di 2m",
            "Temperature at 2 Meters Maximum – suhu maksimum harian di 2m",
            "Dew Point Temperature at 2 Meters – suhu titik embun di 2m",
            "Wet Bulb Temperature at 2 Meters – suhu bola basah (indikasi kelembapan)",
            "Relative Humidity at 2 Meters – kelembaban relatif di 2m",
            "Precipitation Corrected – total curah hujan (telah dikoreksi kualitas data)",
            "Wind Speed at 2 Meters – kecepatan angin di ketinggian 2m",
            "Wind Speed at 10 Meters – kecepatan angin di ketinggian 10m",
            "Wind Speed at 50 Meters – kecepatan angin di ketinggian 50m",
            "Surface Pressure – tekanan udara di permukaan",
            "All Sky Surface Shortwave Downward Irradiance – radiasi gelombang pendek ke permukaan",
            "Top of Atmosphere Shortwave Downward Irradiance – radiasi gelombang pendek di puncak atmosfer"
        ],
        "Satuan": [
            "-", "-", "°C", "°C", "°C", "°C", "°C",
            "%", "mm/day", "m/s", "m/s", "m/s",
            "kPa", "MJ/m²/day", "MJ/m²/day"
        ],
        "Kategori": [
            "Metadata", "Metadata", "Meteorology", "Meteorology", "Meteorology", 
            "Meteorology", "Meteorology", "Meteorology", "Meteorology",
            "Meteorology", "Meteorology", "Meteorology", "Meteorology",
            "Radiation", "Radiation"
        ],
        "Sumber": [
            "Metadata", "Metadata", "NASA POWER", "NASA POWER", "NASA POWER",
            "NASA POWER", "NASA POWER", "NASA POWER", "NASA POWER",
            "NASA POWER", "NASA POWER", "NASA POWER", "NASA POWER",
            "NASA POWER", "NASA POWER"
        ]
    }
    
    df_dict = pd.DataFrame(data_dict)
    
    # Filter berdasarkan kategori
    category_filter = st.multiselect(
        "Filter berdasarkan kategori:",
        options=["Semua", "Metadata", "Meteorology", "Radiation"],
        default=["Semua"]
    )
    
    if "Semua" not in category_filter and len(category_filter) > 0:
        df_dict_filtered = df_dict[df_dict["Kategori"].isin(category_filter)]
    else:
        df_dict_filtered = df_dict
    
    # Tampilkan tabel dengan styling
    st.dataframe(
        df_dict_filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Nama Kolom": st.column_config.TextColumn("Nama Kolom", width="medium"),
            "Deskripsi": st.column_config.TextColumn("Deskripsi", width="large"),
            "Satuan": st.column_config.TextColumn("Satuan", width="small"),
            "Kategori": st.column_config.TextColumn("Kategori", width="small"),
            "Sumber": st.column_config.TextColumn("Sumber", width="small")
        }
    )
    
    st.markdown("---")
    
    # Penjelasan kategori detail
    st.markdown("### 🔍 Penjelasan Detail Kategori")
    
    detail_tab1, detail_tab2, detail_tab3 = st.tabs([
        "🌡️ Parameter Suhu", 
        "💨 Parameter Angin", 
        "☀️ Parameter Radiasi"
    ])
    
    with detail_tab1:
        st.markdown("""
        #### Parameter Suhu (Temperature)
        
        **T2M (Temperature at 2 Meters)**
        - Suhu rata-rata harian yang diukur pada ketinggian 2 meter dari permukaan tanah
        - Standar pengukuran meteorologi internasional
        - Digunakan untuk: analisis tren iklim, prediksi cuaca, perencanaan pertanian
        
        **T2M_MIN & T2M_MAX**
        - Suhu minimum dan maksimum harian
        - Penting untuk: menghitung rentang suhu harian (diurnal temperature range)
        - Indikator kenyamanan termal dan kebutuhan energi
        
        **T2MDEW (Dew Point Temperature)**
        - Suhu di mana udara menjadi jenuh dan embun mulai terbentuk
        - Indikator kelembaban absolut
        - Nilai tinggi (>20°C) = udara lembab dan pengap
        
        **T2MWET (Wet Bulb Temperature)**
        - Suhu bola basah, terendah yang bisa dicapai melalui penguapan
        - Digunakan untuk: menghitung heat index dan kenyamanan termal
        - Penting untuk: keselamatan kerja outdoor dan olahraga
        """)
        
        # Contoh visualisasi
        if len(df_filtered) > 0:
            st.markdown("##### 📊 Contoh Distribusi Suhu")
            fig = go.Figure()
            fig.add_trace(go.Box(y=df_filtered["T2M"], name="T2M", marker_color='#ef4444'))
            fig.add_trace(go.Box(y=df_filtered["T2M_MIN"], name="T2M_MIN", marker_color='#3b82f6'))
            fig.add_trace(go.Box(y=df_filtered["T2M_MAX"], name="T2M_MAX", marker_color='#f59e0b'))
            fig.update_layout(
                title="Distribusi Parameter Suhu",
                yaxis_title="Suhu (°C)",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with detail_tab2:
        st.markdown("""
        #### Parameter Angin (Wind Speed)
        
        **WS2M (Wind Speed at 2 Meters)**
        - Kecepatan angin pada ketinggian 2 meter
        - Relevan untuk: aktivitas manusia di permukaan, pertanian, polusi udara
        - Biasanya lebih rendah karena gesekan dengan permukaan
        
        **WS10M (Wind Speed at 10 Meters)**
        - Kecepatan angin pada ketinggian 10 meter
        - Standar untuk pelaporan cuaca dan aviation
        - Digunakan untuk: estimasi beban angin pada bangunan
        
        **WS50M (Wind Speed at 50 Meters)**
        - Kecepatan angin pada ketinggian 50 meter
        - Sangat penting untuk: **perencanaan turbin angin** dan energi terbarukan
        - Nilai lebih tinggi = potensi energi angin lebih besar
        
        **Skala Kecepatan Angin:**
        - 0-2 m/s: Tenang hingga angin sepoi-sepoi
        - 2-5 m/s: Angin ringan
        - 5-10 m/s: Angin sedang
        - >10 m/s: Angin kencang
        """)
        
        if len(df_filtered) > 0:
            st.markdown("##### 📊 Perbandingan Kecepatan Angin")
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
                title="Rata-rata Kecepatan Angin Berdasarkan Ketinggian",
                yaxis_title="Kecepatan (m/s)",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with detail_tab3:
        st.markdown("""
        #### Parameter Radiasi (Solar Radiation)
        
        **ALLSKY_SFC_SW_DWN (All Sky Surface Shortwave Downward Irradiance)**
        - Radiasi gelombang pendek yang sampai ke permukaan bumi
        - Mencakup semua kondisi langit (cerah, berawan, hujan)
        - Satuan: MJ/m²/day atau bisa dikonversi ke kWh/m²/day
        - **Aplikasi utama**: 
          - Desain sistem panel surya (PLTS)
          - Estimasi produksi energi surya
          - Perencanaan pertanian (fotosintesis)
        
        **TOA_SW_DWN (Top of Atmosphere Shortwave Downward Irradiance)**
        - Radiasi yang sampai di puncak atmosfer (sebelum diserap/dipantulkan)
        - Nilai teoritis maksimum radiasi
        - Selisih dengan ALLSKY_SFC_SW_DWN menunjukkan efek atmosfer
        
        **Interpretasi Nilai:**
        - <3 MJ/m²/day: Sangat rendah (cuaca sangat buruk/malam)
        - 3-10 MJ/m²/day: Rendah (mendung)
        - 10-20 MJ/m²/day: Sedang (cerah berawan)
        - >20 MJ/m²/day: Tinggi (cerah)
        
        **Konversi:**
        - 1 MJ/m²/day ≈ 0.278 kWh/m²/day
        - Contoh: 18 MJ/m²/day = ~5 kWh/m²/day (cukup untuk panel surya)
        """)
        
        if len(df_filtered) > 0:
            st.markdown("##### 📊 Distribusi Radiasi Matahari")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    df_filtered,
                    x="ALLSKY_SFC_SW_DWN",
                    nbins=30,
                    title="Distribusi Radiasi Permukaan",
                    labels={"ALLSKY_SFC_SW_DWN": "Radiasi (MJ/m²/day)"},
                    color_discrete_sequence=['#f59e0b']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                monthly_rad = df_filtered.groupby("Month")["ALLSKY_SFC_SW_DWN"].mean().reset_index()
                fig = px.line(
                    monthly_rad,
                    x="Month",
                    y="ALLSKY_SFC_SW_DWN",
                    title="Rata-rata Radiasi per Bulan",
                    markers=True,
                    labels={"ALLSKY_SFC_SW_DWN": "Radiasi (MJ/m²/day)", "Month": "Bulan"}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Informasi tambahan
    st.markdown("### 📚 Sumber & Referensi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🌐 NASA POWER**
        - Website: [power.larc.nasa.gov](https://power.larc.nasa.gov)
        - Resolusi Temporal: Harian
        - Resolusi Spasial: 0.5° x 0.625°
        - Metode: Assimilasi data satelit dan model
        
        **📖 Dokumentasi**
        - [NASA POWER Documentation](https://power.larc.nasa.gov/docs/)
        - [Data Access Guide](https://power.larc.nasa.gov/docs/services/api/)
        """)
    
    with col2:
        st.markdown("""
        **🎯 Kegunaan Data**
        - ☀️ Perencanaan energi terbarukan
        - 🌱 Optimasi jadwal tanam pertanian
        - 🏗️ Desain bangunan hemat energi
        - 🌡️ Analisis perubahan iklim
        - 📊 Penelitian meteorologi
        
        **⚠️ Catatan Penting**
        - Data sudah melalui quality control
        - PRECTOTCORR adalah data terkoreksi
        - Missing values minimal (<1%)
        """)
    
    st.markdown("---")
    
    # Download kamus data
    st.markdown("### 💾 Download Dokumentasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Kamus Data (CSV)", type="primary"):
            csv = df_dict.to_csv(index=False)
            st.download_button(
                label="💾 Klik untuk Download",
                data=csv,
                file_name="kamus_data_cuaca_bandung.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Download Data Sample", type="secondary"):
            sample_data = df.head(100).to_csv(index=False)
            st.download_button(
                label="💾 Klik untuk Download",
                data=sample_data,
                file_name="sample_data_cuaca_bandung.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("📊 **Data Source**: NASA POWER")
with col2:
    st.markdown(f"📅 **Last Update**: {df['Date'].max().strftime('%d %B %Y')}")
with col3:
    st.markdown(f"📈 **Total Records**: {len(df_filtered):,}")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280; padding: 1rem;'>
        Made with ❤️ using Streamlit | Dashboard Cuaca Bandung v2.0 | Powered by Random Forest ML
    </div>
    """,
    unsafe_allow_html=True
)



