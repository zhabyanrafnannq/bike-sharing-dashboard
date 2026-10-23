from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

sns.set(style='dark')

st.set_page_config(page_title="Bike Sharing Dashboard", page_icon="🚲", layout="wide")


# ================= Helper Functions =================
def create_daily_rentals_df(df):
    """Menyiapkan daily_rentals_df: jumlah penyewaan harian."""
    daily_rentals_df = df.resample(rule='D', on='dteday').agg({
        "cnt": "sum",
        "casual": "sum",
        "registered": "sum"
    })
    daily_rentals_df = daily_rentals_df.reset_index()
    daily_rentals_df.rename(columns={
        "cnt": "rental_count",
        "casual": "casual_count",
        "registered": "registered_count"
    }, inplace=True)

    return daily_rentals_df


def create_monthly_performance_df(df):
    """Menyiapkan monthly_df: rata-rata penyewaan per bulan."""
    monthly_df = df.groupby("month_name").agg(
        avg_cnt=("cnt", "mean")
    ).reset_index()
    monthly_df["avg_cnt"] = monthly_df["avg_cnt"].round(0).astype(int)
    monthly_df = monthly_df.sort_values(by="avg_cnt", ascending=False).reset_index(drop=True)

    return monthly_df


def create_byweather_df(df):
    """Menyiapkan byweather_df: rata-rata penyewaan per kondisi cuaca."""
    byweather_df = df.groupby("weather_name").agg(
        avg_cnt=("cnt", "mean")
    ).reset_index()
    byweather_df["avg_cnt"] = byweather_df["avg_cnt"].round(0).astype(int)
    byweather_df = byweather_df.sort_values(by="avg_cnt", ascending=False).reset_index(drop=True)

    return byweather_df


def create_byseason_df(df):
    """Menyiapkan byseason_df: rata-rata penyewaan per musim."""
    season_order = ["Spring", "Summer", "Fall", "Winter"]
    byseason_df = df.groupby("season_name").agg(
        avg_cnt=("cnt", "mean")
    ).reset_index()
    byseason_df["avg_cnt"] = byseason_df["avg_cnt"].round(0).astype(int)
    byseason_df['season_name'] = pd.Categorical(byseason_df['season_name'], season_order)
    byseason_df = byseason_df.sort_values(by="season_name").reset_index(drop=True)

    return byseason_df


def create_hourly_pattern_df(df):
    """Menyiapkan hourly_df: rata-rata penyewaan per jam, hari kerja vs akhir pekan."""
    working = df[df['workingday'] == 1].groupby('hr')['cnt'].mean().round(0).astype(int)
    weekend = df[df['workingday'] == 0].groupby('hr')['cnt'].mean().round(0).astype(int)
    hourly_df = pd.DataFrame({
        "hr": working.index,
        "workingday": working.values,
        "weekend": weekend.values
    }).reset_index(drop=True)

    return hourly_df


def create_peak_hours_df(df, workingday):
    """Menyiapkan peak_df: 5 jam dengan penyewaan tertinggi berdasarkan tipe hari."""
    subset = df[df['workingday'] == workingday]
    peak_df = subset.groupby('hr')['cnt'].mean().round(0).astype(int).reset_index()
    peak_df = peak_df.sort_values(by='cnt', ascending=False).head(5).reset_index(drop=True)

    return peak_df


def create_byusertype_df(df):
    """Menyiapkan bytype_df: total penyewaan registered vs casual."""
    bytype_df = pd.DataFrame({
        "user_type": ["Registered", "Casual"],
        "total": [df["registered"].sum(), df["casual"].sum()]
    })

    return bytype_df


# ================= Menyiapkan DataFrame =================
@st.cache_data
def load_data():
    # Satu berkas hasil ekspor notebook: all_data.csv (data per jam + weather_day)
    hour_df = pd.read_csv(BASE_DIR / "all_data.csv")
    hour_df["dteday"] = pd.to_datetime(hour_df["dteday"])

    weather_map = {1: "Cerah", 2: "Kabut", 3: "Hujan Ringan", 4: "Hujan Lebat"}

    # Turunkan data harian dari data per jam (jumlah penyewaan per hari = jumlah per jam)
    day_df = hour_df.groupby("dteday").agg(
        cnt=("cnt", "sum"),
        casual=("casual", "sum"),
        registered=("registered", "sum"),
        season_name=("season_name", "first"),
        weather_day=("weather_day", "first"),
        workingday=("workingday", "first")
    ).reset_index()
    day_df["weather_name"] = day_df["weather_day"].map(weather_map)
    day_df["month_name"] = day_df["dteday"].dt.strftime('%b')

    return day_df, hour_df


all_df, hour_df = load_data()

datetime_columns = ["dteday"]
all_df.sort_values(by="dteday", inplace=True)
all_df.reset_index(inplace=True)
hour_df.sort_values(by="dteday", inplace=True)
hour_df.reset_index(inplace=True)

for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])
    hour_df[column] = pd.to_datetime(hour_df[column])


# ================= Membuat Komponen Filter =================
min_date = all_df["dteday"].min()
max_date = all_df["dteday"].max()

with st.sidebar:
    # Menambahkan logo perusahaan
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png")

    st.markdown("### Bike Sharing Dataset")
    st.caption("Fanaee-T & Gama, 2014 | Periode 2011-2012")

    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Rentang Waktu', min_value=min_date.date(),
        max_value=max_date.date(),
        value=[min_date.date(), max_date.date()]
    )

main_df = all_df[(all_df["dteday"] >= str(start_date)) &
                 (all_df["dteday"] <= str(end_date))]
main_hour_df = hour_df[(hour_df["dteday"] >= str(start_date)) &
                       (hour_df["dteday"] <= str(end_date))]

daily_rentals_df = create_daily_rentals_df(main_df)
monthly_df = create_monthly_performance_df(main_df)
byweather_df = create_byweather_df(main_df)
byseason_df = create_byseason_df(main_df)
hourly_df = create_hourly_pattern_df(main_hour_df)
peak_working_df = create_peak_hours_df(main_hour_df, workingday=1)
peak_weekend_df = create_peak_hours_df(main_hour_df, workingday=0)
bytype_df = create_byusertype_df(main_df)


# ================= Header =================
st.header('Bike Sharing Dashboard :sparkles:')

if main_df.empty:
    st.warning("Tidak ada data pada rentang waktu yang dipilih.")
    st.stop()


# ================= Daily Rentals =================
st.subheader('Daily Rentals')

col1, col2, col3 = st.columns(3)

with col1:
    total_rentals = daily_rentals_df.rental_count.sum()
    st.metric("Total Rentals", value=f"{total_rentals:,}".replace(",", "."))

with col2:
    avg_daily = round(daily_rentals_df.rental_count.mean())
    st.metric("Average Daily Rentals", value=f"{avg_daily:,}".replace(",", "."))

with col3:
    registered_share = 100 * bytype_df.loc[0, "total"] / bytype_df["total"].sum()
    st.metric("Registered User Share", value=f"{registered_share:.1f}%")

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    daily_rentals_df["dteday"],
    daily_rentals_df["rental_count"],
    marker='o',
    markersize=3,
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)

st.pyplot(fig)


# ================= Best & Worst Performing Month =================
st.subheader("Best & Worst Performing Month")

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))

colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

best_df = monthly_df.head(5)
sns.barplot(x="avg_cnt", y="month_name", data=best_df, palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Average Rentals per Day", fontsize=30)
ax[0].set_title("Best Performing Month", loc="center", fontsize=50)
ax[0].tick_params(axis='y', labelsize=35)
ax[0].tick_params(axis='x', labelsize=30)

worst_df = monthly_df.sort_values(by="avg_cnt", ascending=True).head(5)
sns.barplot(x="avg_cnt", y="month_name", data=worst_df, palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Average Rentals per Day", fontsize=30)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Worst Performing Month", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=35)
ax[1].tick_params(axis='x', labelsize=30)

st.pyplot(fig)


# ================= Rental Characteristics =================
st.subheader("Rental Characteristics")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(20, 10))

    sns.barplot(
        y="avg_cnt",
        x="weather_name",
        data=byweather_df,
        palette=colors,
        ax=ax
    )
    ax.set_title("Average Rentals by Weather", loc="center", fontsize=50)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=30)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(20, 10))

    colors_season = ["#D3D3D3", "#90CAF9", "#D3D3D3", "#D3D3D3"]

    sns.barplot(
        y="avg_cnt",
        x="season_name",
        data=byseason_df,
        palette=colors_season,
        ax=ax
    )
    ax.set_title("Average Rentals by Season", loc="center", fontsize=50)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=30)
    st.pyplot(fig)

fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(hourly_df["hr"], hourly_df["workingday"], marker='o', markersize=6,
        linewidth=3, color="#90CAF9", label="Weekday")
ax.plot(hourly_df["hr"], hourly_df["weekend"], marker='o', markersize=6,
        linewidth=3, color="#D3D3D3", label="Weekend")
ax.set_title("Average Rentals per Hour: Weekday vs Weekend", loc="center", fontsize=30)
ax.set_ylabel(None)
ax.set_xlabel("Hour", fontsize=20)
ax.set_xticks(range(0, 24))
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
ax.legend(fontsize=20)
ax.grid(axis='y', linestyle='--', alpha=0.4)
st.pyplot(fig)


# ================= Peak Hour & User Type =================
st.subheader("Peak Hour & User Type")

col1, col2, col3 = st.columns(3)

with col1:
    busiest_hour = int(hourly_df.loc[hourly_df['workingday'].idxmax(), 'hr'])
    st.metric("Busiest Hour (Weekday)", value=f"{busiest_hour}.00")

with col2:
    peak_avg = int(hourly_df['workingday'].max())
    st.metric("Peak Average (rentals/hour)", value=f"{peak_avg:,}".replace(",", "."))

with col3:
    weekend_peak_hour = int(hourly_df.loc[hourly_df['weekend'].idxmax(), 'hr'])
    st.metric("Busiest Hour (Weekend)", value=f"{weekend_peak_hour}.00")

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(35, 15))
colors_peak = ["#90CAF9", "#90CAF9", "#90CAF9", "#90CAF9", "#90CAF9"]

sns.barplot(y="cnt", x="hr", data=peak_working_df, palette=colors_peak, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Hour", fontsize=30)
ax[0].set_title("Top 5 Rush Hours (Weekday)", loc="center", fontsize=50)
ax[0].tick_params(axis='y', labelsize=30)
ax[0].tick_params(axis='x', labelsize=35)

sns.barplot(y="cnt", x="hr", data=peak_weekend_df, palette=colors_peak, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Hour", fontsize=30)
ax[1].set_title("Top 5 Hours (Weekend)", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=30)
ax[1].tick_params(axis='x', labelsize=35)

sns.barplot(y="total", x="user_type", data=bytype_df, palette=colors_peak, ax=ax[2])
ax[2].set_ylabel(None)
ax[2].set_xlabel("Total Rentals", fontsize=30)
ax[2].set_title("Registered vs Casual", loc="center", fontsize=50)
ax[2].tick_params(axis='y', labelsize=30)
ax[2].tick_params(axis='x', labelsize=35)

st.pyplot(fig)

st.caption('Copyright (c) Zhabyan Rafnan Nur Qohharsya 2026')
