# Bike Sharing Dashboard ✨

## Setup environment
```
conda create --name main-ds python=3.11
conda activate main-ds
pip install numpy pandas matplotlib seaborn jupyter nbconvert streamlit
```

## Run notebook
Buka `notebook.ipynb` pada Colab/Jupyter, upload `day.csv` dan `hour.csv` dari folder `data` ke working directory, lalu Run All. Cell terakhir akan menghasilkan `all_data.csv` (data yang sudah dibersihkan) — pindahkan ke folder `dashboard/`.

## Run streamlit app
```
cd dashboard
streamlit run dashboard.py
```
