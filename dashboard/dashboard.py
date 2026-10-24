from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent

# Palet: #90CAF9 = penanda (highlight), #D3D3D3 = konteks. Highlight ditentukan dari data,
# bukan dari urutan bar, agar tidak pernah "nyangkut" di kategori yang salah.
COLOR_HIGHLIGHT = "#90CAF9"
COLOR_CONTEXT = "#D3D3D3"

px.defaults.template = "plotly_white"

st.set_page_config(page_title="Bike Sharing Dashboard", page_icon="🚲", layout="wide")


# ================= Helper Functions =================
def highlight_colors(values):
    """Warnai batang tertinggi dengan warna highlight, sisanya warna konteks."""
    values = list(values)
    top = max(values)
    return [COLOR_HIGHLIGHT if val == top else COLOR_CONTEXT for val in values]


def thous(value):
    """Format angka gaya Indonesia: 64544 -> '64.544'."""
    return f"{value:,.0f}".replace(",", ".")


def create_daily_rentals_df(df):
    """Menyiapkan daily_rentals_df: jumlah penyewaan harian."""
    daily_rentals_df = df.resample(rule='D', on='dteday').agg({
        "cnt": "sum",
        "casual": "sum",
        "registered": "sum"
    })
    daily_rentals_df = daily_rentals_df.reset_index()
    daily_rentals_df = daily_rentals_df.rename(columns={
        "cnt": "rental_count",
        "casual": "casual_count",
        "registered": "registered_count"
    })

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
        avg_cnt=("cnt", "mean"),
        jumlah_hari=("cnt", "count")
    ).reset_index()
    byweather_df["avg_cnt"] = byweather_df["avg_cnt"].round(0).astype(int)
    byweather_df = byweather_df.sort_values(by="avg_cnt", ascending=False).reset_index(drop=True)

    return byweather_df


def create_byseason_df(df):
    """Menyiapkan byseason_df: rata-rata penyewaan per musim."""
    season_order = ["Spring", "Summer", "Fall", "Winter"]
    byseason_df = df.groupby("season_name").agg(
        avg_cnt=("cnt", "mean"),
        jumlah_hari=("cnt", "count")
    ).reset_index()
    byseason_df["avg_cnt"] = byseason_df["avg_cnt"].round(0).astype(int)
    byseason_df['season_name'] = pd.Categorical(byseason_df['season_name'], season_order)
    byseason_df = byseason_df.sort_values(by="season_name").reset_index(drop=True)

    return byseason_df


def create_hourly_pattern_df(df):
    """Menyiapkan hourly_df: rata-rata penyewaan per jam, hari kerja vs akhir pekan.

    Jam tanpa data (mis. rentang tanggal yang isinya hanya hari kerja) diisi NaN
    lewat reindex(range(24)) supaya panjang kedua seri selalu 24 dan tidak
    memicu error "All arrays must be of the same length".
    """
    working = df[df['workingday'] == 1].groupby('hr')['cnt'].mean().reindex(range(24))
    weekend = df[df['workingday'] == 0].groupby('hr')['cnt'].mean().reindex(range(24))
    hourly_df = pd.DataFrame({
        "hr": list(range(24)),
        "workingday": working.to_numpy(),
        "weekend": weekend.to_numpy()
    })

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


def create_demand_cluster_df(df):
    """Menyiapkan cluster_df: profil hari per kelompok permintaan hasil binning."""
    cluster_df = df.groupby("demand_level", observed=True).agg(
        jumlah_hari=("cnt", "count"),
        rata_rata_cnt=("cnt", "mean"),
        pct_akhir_pekan=("workingday", lambda x: 100 * (x == 0).mean()),
        pct_cuaca_buruk=("weathersit", lambda x: 100 * (x >= 2).mean()),
        total_casual=("casual", "sum"),
        total_registered=("registered", "sum")
    )
    cluster_df["rata_rata_cnt"] = cluster_df["rata_rata_cnt"].round(0)
    cluster_df["pct_akhir_pekan"] = cluster_df["pct_akhir_pekan"].round(1)
    cluster_df["pct_cuaca_buruk"] = cluster_df["pct_cuaca_buruk"].round(1)
    cluster_df["pct_casual"] = (
        100 * cluster_df["total_casual"]
        / (cluster_df["total_casual"] + cluster_df["total_registered"])
    ).round(1)
    cluster_df = cluster_df.drop(columns=["total_casual", "total_registered"])
    cluster_df.index = cluster_df.index.astype(str)

    return cluster_df.reset_index()


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
    day_df["weathersit"] = day_df["weather_day"]
    day_df["weather_name"] = day_df["weather_day"].map(weather_map)
    day_df["month_name"] = day_df["dteday"].dt.strftime('%b')

    # Clustering binning dihitung pada SELURUH periode (731 hari) lebih dulu, bukan pada
    # hasil filter, supaya label Low/Medium/High Demand tetap konsisten di rentang apa pun.
    q1, q3 = day_df["cnt"].quantile([0.33, 0.66])
    day_df["demand_level"] = pd.cut(
        day_df["cnt"], bins=[0, q1, q3, day_df["cnt"].max()],
        labels=["Low Demand", "Medium Demand", "High Demand"], include_lowest=True
    )

    return day_df, hour_df


all_df, hour_df = load_data()

all_df = all_df.sort_values(by="dteday").reset_index(drop=True)
hour_df = hour_df.sort_values(by="dteday").reset_index(drop=True)


# ================= Membuat Komponen Filter =================
min_date = all_df["dteday"].min()
max_date = all_df["dteday"].max()

with st.sidebar:
    # Menambahkan logo perusahaan
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png")

    st.markdown("### Bike Sharing Dataset")
    st.caption("Fanaee-T & Gama, 2014 | Periode 2011-2012")

    # Mengambil start_date & end_date dari date_input
    date_range = st.date_input(
        label='Rentang Waktu', min_value=min_date.date(),
        max_value=max_date.date(),
        value=[min_date.date(), max_date.date()]
    )

# date_input mengembalikan 2 tanggal, atau 1 tanggal saja ketika pengguna baru memilih
# satu ujung rentang. Kedua bentuk itu ditangani di sini agar tidak muncul
# "not enough values to unpack".
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
    start_date = end_date = date_range[0]
else:
    start_date, end_date = min_date.date(), max_date.date()

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

main_df = all_df[(all_df["dteday"] >= start_ts) & (all_df["dteday"] <= end_ts)]
main_hour_df = hour_df[(hour_df["dteday"] >= start_ts) & (hour_df["dteday"] <= end_ts)]


# ================= Header =================
st.header('Bike Sharing Dashboard :sparkles:')

if main_df.empty:
    st.warning("Tidak ada data pada rentang waktu yang dipilih. Silakan perlebar rentang tanggalnya.")
    st.stop()

st.caption(
    f"Periode terpilih: **{start_ts:%d %b %Y}** s.d. **{end_ts:%d %b %Y}** "
    f"({main_df.shape[0]} hari, {thous(main_hour_df.shape[0])} baris data per jam)"
)

# Semua agregasi dipanggil SETELAH pengecekan data kosong, supaya tidak ada
# helper function yang dijalankan di atas DataFrame kosong.
daily_rentals_df = create_daily_rentals_df(main_df)
monthly_df = create_monthly_performance_df(main_df)
byweather_df = create_byweather_df(main_df)
byseason_df = create_byseason_df(main_df)
hourly_df = create_hourly_pattern_df(main_hour_df)
peak_working_df = create_peak_hours_df(main_hour_df, workingday=1)
peak_weekend_df = create_peak_hours_df(main_hour_df, workingday=0)
bytype_df = create_byusertype_df(main_df)
cluster_df = create_demand_cluster_df(main_df)


# ================= Daily Rentals =================
st.subheader('Daily Rentals')

col1, col2, col3 = st.columns(3)

with col1:
    total_rentals = daily_rentals_df.rental_count.sum()
    st.metric("Total Rentals", value=thous(total_rentals), help="Total penyewaan sepeda pada periode terpilih.")

with col2:
    avg_daily = round(daily_rentals_df.rental_count.mean())
    st.metric("Average Daily Rentals", value=thous(avg_daily), help="Satuan: sepeda/hari.")

with col3:
    registered_share = 100 * bytype_df.loc[0, "total"] / bytype_df["total"].sum()
    st.metric("Registered User Share", value=f"{registered_share:.1f}%")


fig = go.Figure()
fig.add_trace(go.Scatter(
    x=daily_rentals_df["dteday"], y=daily_rentals_df["rental_count"],
    mode="lines+markers", name="Total",
    line=dict(color=COLOR_HIGHLIGHT, width=2), marker=dict(size=5),
    hovertemplate="%{x|%d %b %Y}<br>Total: %{y:,} sepeda<extra></extra>"
))
# Klik legenda untuk menampilkan/menyembunyikan dua seri di bawah ini
fig.add_trace(go.Scatter(
    x=daily_rentals_df["dteday"], y=daily_rentals_df["registered_count"],
    mode="lines", name="Registered", visible="legendonly",
    line=dict(color="#1f77b4", width=1.5),
    hovertemplate="%{x|%d %b %Y}<br>Registered: %{y:,} sepeda<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=daily_rentals_df["dteday"], y=daily_rentals_df["casual_count"],
    mode="lines", name="Casual", visible="legendonly",
    line=dict(color="#ff7f0e", width=1.5),
    hovertemplate="%{x|%d %b %Y}<br>Casual: %{y:,} sepeda<extra></extra>"
))

fig.update_layout(
    title="Penyewaan Sepeda Harian (2011-2012)",
    xaxis_title="Tanggal",
    yaxis_title="Jumlah sepeda disewa (sepeda/hari)",
    hovermode="x unified",
    height=460,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=90, b=10)
)

st.plotly_chart(fig)
st.caption("Tip: tarik (drag) pada grafik untuk zoom ke periode tertentu, atau klik legenda untuk menyembunyikan serinya.")


# ================= Best & Worst Performing Month =================
st.subheader("Best & Worst Performing Month")

best_df = monthly_df.head(5)
worst_df = monthly_df.sort_values(by="avg_cnt", ascending=True).head(5)

fig = make_subplots(
    rows=1, cols=2, horizontal_spacing=0.12,
    subplot_titles=("Best Performing Month", "Worst Performing Month")
)

fig.add_trace(go.Bar(
    x=best_df["avg_cnt"], y=best_df["month_name"], orientation="h",
    marker_color=highlight_colors(best_df["avg_cnt"]),
    text=[thous(v) for v in best_df["avg_cnt"]], textposition="outside",
    hovertemplate="%{y}<br>Rata-rata: %{x:,} sepeda/hari<extra></extra>",
    showlegend=False
), row=1, col=1)

fig.add_trace(go.Bar(
    x=worst_df["avg_cnt"], y=worst_df["month_name"], orientation="h",
    marker_color=COLOR_CONTEXT,
    text=[thous(v) for v in worst_df["avg_cnt"]], textposition="outside",
    hovertemplate="%{y}<br>Rata-rata: %{x:,} sepeda/hari<extra></extra>",
    showlegend=False
), row=1, col=2)

fig.update_xaxes(title_text="Rata-rata sepeda/hari", row=1, col=1)
fig.update_xaxes(title_text="Rata-rata sepeda/hari", autorange="reversed", row=1, col=2)
fig.update_yaxes(title_text=None, row=1, col=1)
fig.update_yaxes(title_text=None, row=1, col=2)
fig.update_layout(
    height=460,
    margin=dict(l=10, r=10, t=70, b=10)
)

st.plotly_chart(fig)


# ================= Rental Characteristics =================
st.subheader("Rental Characteristics")

col1, col2 = st.columns(2)

with col1:
    fig = go.Figure(go.Bar(
        x=byweather_df["weather_name"], y=byweather_df["avg_cnt"],
        marker_color=highlight_colors(byweather_df["avg_cnt"]),
        text=[thous(v) for v in byweather_df["avg_cnt"]], textposition="outside",
        customdata=byweather_df[["jumlah_hari"]],
        hovertemplate=("%{x}<br>Rata-rata: %{y:,} sepeda/hari"
                       "<br>Jumlah hari: %{customdata[0]} hari<extra></extra>"),
        showlegend=False
    ))
    fig.update_layout(
        title=dict(text="Average Rentals by Weather", x=0.5, xanchor="center"),
        xaxis_title="Kondisi cuaca",
        yaxis_title="Rata-rata sepeda disewa (sepeda/hari)",
        height=460,
        margin=dict(l=10, r=10, t=70, b=10)
    )
    st.plotly_chart(fig)

with col2:
    fig = go.Figure(go.Bar(
        x=byseason_df["season_name"], y=byseason_df["avg_cnt"],
        marker_color=highlight_colors(byseason_df["avg_cnt"]),
        text=[thous(v) for v in byseason_df["avg_cnt"]], textposition="outside",
        customdata=byseason_df[["jumlah_hari"]],
        hovertemplate=("%{x}<br>Rata-rata: %{y:,} sepeda/hari"
                       "<br>Jumlah hari: %{customdata[0]} hari<extra></extra>"),
        showlegend=False
    ))
    fig.update_layout(
        title=dict(text="Average Rentals by Season", x=0.5, xanchor="center"),
        xaxis_title="Musim",
        yaxis_title="Rata-rata sepeda disewa (sepeda/hari)",
        height=460,
        margin=dict(l=10, r=10, t=70, b=10)
    )
    st.plotly_chart(fig)


# Tinggi batang tertinggi ditentukan dari data (avg_cnt), jadi highlight selalu jatuh tepat
peak_weather = byweather_df.loc[byweather_df["avg_cnt"].idxmax(), "weather_name"]
peak_season = byseason_df.loc[byseason_df["avg_cnt"].idxmax(), "season_name"]
st.caption(
    f"Batang berwarna terang menandai nilai tertinggi pada data periode terpilih, "
    f"yaitu **{peak_weather}** untuk cuaca dan **{peak_season}** untuk musim."
)


# ================= Hourly Pattern =================
st.subheader("Hourly Pattern: Weekday vs Weekend")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hourly_df["hr"], y=hourly_df["workingday"],
    mode="lines+markers", name="Weekday",
    line=dict(color=COLOR_HIGHLIGHT, width=3), marker=dict(size=7),
    hovertemplate="Jam %{x}.00<br>Weekday: %{y:.0f} sepeda/jam<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=hourly_df["hr"], y=hourly_df["weekend"],
    mode="lines+markers", name="Weekend",
    line=dict(color="#1f77b4", width=3), marker=dict(size=7),
    hovertemplate="Jam %{x}.00<br>Weekend: %{y:.0f} sepeda/jam<extra></extra>"
))

fig.update_layout(
    title="Rata-rata Penyewaan per Jam: Weekday vs Weekend",
    xaxis_title="Jam dalam sehari",
    yaxis_title="Rata-rata sepeda disewa (sepeda/jam)",
    hovermode="x unified",
    height=460,
    xaxis=dict(dtick=1, range=[-0.5, 23.5]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=90, b=10)
)

st.plotly_chart(fig)
st.caption("Tip: klik legenda **Weekday**/**Weekend** untuk menampilkan atau menyembunyikan salah satu seri.")


# ================= Peak Hour & User Type =================
st.subheader("Peak Hour & User Type")

has_weekday = hourly_df["workingday"].notna().any()
has_weekend = hourly_df["weekend"].notna().any()

col1, col2, col3 = st.columns(3)

with col1:
    if has_weekday:
        busiest_hour = int(hourly_df.loc[hourly_df['workingday'].idxmax(), 'hr'])
        st.metric("Busiest Hour (Weekday)", value=f"{busiest_hour}.00")
    else:
        st.metric("Busiest Hour (Weekday)", value="—", help="Tidak ada data hari kerja pada rentang ini.")

with col2:
    if has_weekday:
        peak_avg = int(round(hourly_df['workingday'].max()))
        st.metric("Peak Average (rentals/hour)", value=thous(peak_avg))
    else:
        st.metric("Peak Average (rentals/hour)", value="—")

with col3:
    if has_weekend:
        weekend_peak_hour = int(hourly_df.loc[hourly_df['weekend'].idxmax(), 'hr'])
        st.metric("Busiest Hour (Weekend)", value=f"{weekend_peak_hour}.00")
    else:
        st.metric("Busiest Hour (Weekend)", value="—", help="Tidak ada data akhir pekan pada rentang ini.")

if not has_weekend:
    st.info("Rentang tanggal yang dipilih hanya berisi hari kerja, sehingga seri Weekend tidak tersedia.")

panels = []
if not peak_working_df.empty:
    panels.append(("Top 5 Rush Hours (Weekday)", peak_working_df["hr"], peak_working_df["cnt"],
                   "Jam", "Rata-rata sepeda/jam"))
if not peak_weekend_df.empty:
    panels.append(("Top 5 Hours (Weekend)", peak_weekend_df["hr"], peak_weekend_df["cnt"],
                   "Jam", "Rata-rata sepeda/jam"))
panels.append(("Registered vs Casual", bytype_df["user_type"], bytype_df["total"],
               "Tipe pengguna", "Total sepeda"))

fig = make_subplots(rows=1, cols=len(panels), horizontal_spacing=0.10,
                    subplot_titles=[panel[0] for panel in panels])

for i, (title, x_values, y_values, x_title, y_title) in enumerate(panels, start=1):
    if title == "Registered vs Casual":
        x_axis = [str(v) for v in x_values]
        marker_color = highlight_colors(y_values)
    else:
        x_axis = [f"{int(v)}.00" for v in x_values]
        marker_color = COLOR_HIGHLIGHT
    fig.add_trace(go.Bar(
        x=x_axis, y=[float(v) for v in y_values],
        marker_color=marker_color,
        text=[thous(v) for v in y_values], textposition="outside",
        hovertemplate=f"{x_title}: %{{x}}<br>{y_title}: %{{y:,.0f}}<extra></extra>",
        showlegend=False
    ), row=1, col=i)
    fig.update_xaxes(title_text=x_title, row=1, col=i)
    fig.update_yaxes(title_text=y_title, row=1, col=i)

fig.update_layout(height=460, margin=dict(l=10, r=10, t=70, b=10))

st.plotly_chart(fig)


# ================= Demand Cluster (Analisis Lanjutan) =================
st.subheader("Demand Cluster: Low / Medium / High Demand")

st.caption(
    "Hasil analisis lanjutan (clustering manual / binning) dari notebook. Ambang batas "
    "kelompok dihitung dari seluruh periode 2011-2012 (kuantil 33% dan 66% dari penyewaan harian), "
    "lalu hari pada rentang tanggal terpilih dipetakan ke kelompoknya."
)

col1, col2 = st.columns(2)

with col1:
    fig = go.Figure(go.Bar(
        x=cluster_df["demand_level"], y=cluster_df["rata_rata_cnt"],
        marker_color=highlight_colors(cluster_df["rata_rata_cnt"]),
        text=[thous(v) for v in cluster_df["rata_rata_cnt"]], textposition="outside",
        customdata=cluster_df[["jumlah_hari"]],
        hovertemplate=("%{x}<br>Rata-rata: %{y:,.0f} sepeda/hari"
                       "<br>Jumlah hari: %{customdata[0]} hari<extra></extra>"),
        showlegend=False
    ))
    fig.update_layout(
        title=dict(text="Rata-rata Penyewaan per Kelompok Permintaan", x=0.5, xanchor="center"),
        xaxis_title="Kelompok permintaan",
        yaxis_title="Rata-rata sepeda disewa (sepeda/hari)",
        height=430,
        margin=dict(l=10, r=10, t=70, b=10)
    )
    st.plotly_chart(fig)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cluster_df["demand_level"], y=cluster_df["pct_cuaca_buruk"],
        name="Hari cuaca kabut/hujan (%)",
        marker_color=COLOR_HIGHLIGHT,
        text=[f"{v:.1f}%" for v in cluster_df["pct_cuaca_buruk"]], textposition="outside",
        hovertemplate="%{x}<br>Cuaca kabut/hujan: %{y:.1f}% hari<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=cluster_df["demand_level"], y=cluster_df["pct_casual"],
        name="Porsi pengguna casual (%)",
        marker_color=COLOR_CONTEXT,
        text=[f"{v:.1f}%" for v in cluster_df["pct_casual"]], textposition="outside",
        hovertemplate="%{x}<br>Porsi casual: %{y:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="Komposisi Cuaca Buruk & Pengguna Casual", x=0.5, xanchor="center"),
        xaxis_title="Kelompok permintaan",
        yaxis_title="Persentase (%)",
        barmode="group",
        height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=90, b=10)
    )
    st.plotly_chart(fig)

st.dataframe(
    cluster_df.rename(columns={
        "demand_level": "Kelompok",
        "jumlah_hari": "Jumlah hari",
        "rata_rata_cnt": "Rata-rata sepeda/hari",
        "pct_akhir_pekan": "Hari akhir pekan (%)",
        "pct_cuaca_buruk": "Hari cuaca kabut/hujan (%)",
        "pct_casual": "Porsi pengguna casual (%)"
    }),
    hide_index=True,
    column_config={
        "Rata-rata sepeda/hari": st.column_config.NumberColumn(format="%.0f"),
        "Hari akhir pekan (%)": st.column_config.NumberColumn(format="%.1f%%"),
        "Hari cuaca kabut/hujan (%)": st.column_config.NumberColumn(format="%.1f%%"),
        "Porsi pengguna casual (%)": st.column_config.NumberColumn(format="%.1f%%")
    }
)

st.caption(
    "Semakin tinggi kelompok permintaan, semakin kecil porsi hari dengan cuaca kabut/hujan — "
    "konsisten dengan temuan bahwa cuaca adalah penekan permintaan terkuat."
)

st.caption('Copyright (c) Zhabyan Rafnan Nur Qohharsya 2026')
