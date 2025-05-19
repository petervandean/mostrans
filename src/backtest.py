import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def evaluate_forecast(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return mae, rmse, mape

def backtest(df, period=7, plot_output_dir=None):
    """
    Backtest dua metode (STL-HW vs Mean) dengan:
      - Test periode = 7 hari terakhir
      - Train = sisa data sebelum 7 hari terakhir
    Jika plot_output_dir diset, simpan grafik per outlet–product di sana.
    """
    results = []

    for (outlet, product), group in df.groupby(['outlet','product']):
        ts = (group
              .sort_values('date')
              .set_index('date')['qty_box']
              .asfreq('D')
              .fillna(method='ffill')
              .fillna(method='bfill'))

        # skip jika data kurang dari seminggu + 1 titik
        if len(ts) <= period:
            continue

        # split: terakhir 7 hari untuk test, sisanya train
        train, test = ts[:-period], ts[-period:]
        h = len(test)  # = period

        # === 1) STL + Holt–Winters ===
        stl = STL(train, period=period, robust=True).fit()
        trend, seasonal = stl.trend, stl.seasonal
        hw = ExponentialSmoothing(trend.dropna(),
                                  trend='add',
                                  initialization_method='estimated').fit()
        trend_fc = hw.forecast(h)
        last_season = seasonal[-period:]
        seasonal_fc = np.tile(last_season.values, int(np.ceil(h/period)))[:h]
        fc_stl = trend_fc.values + seasonal_fc

        mae1, rmse1, mape1 = evaluate_forecast(test.values, fc_stl)
        results.append({
            'outlet': outlet, 'product': product, 'method': 'STL-HW',
            'MAE': mae1, 'RMSE': rmse1, 'MAPE': mape1, 'n_test': h
        })

        # === 2) Mean Forecast ===
        mean_val = train.mean()
        fc_mean = np.full(h, mean_val)
        mae2, rmse2, mape2 = evaluate_forecast(test.values, fc_mean)
        results.append({
            'outlet': outlet, 'product': product, 'method': 'Mean',
            'MAE': mae2, 'RMSE': rmse2, 'MAPE': mape2, 'n_test': h
        })

        # === 3) Plot per seri ===
        if plot_output_dir:
            plt.figure(figsize=(8,6))
            # Plot train + actual test
            plt.plot(train.index, train.values, label='Train')
            plt.plot(test.index, test.values, label='Actual Test', linewidth=1.5)
            # Plot forecasts
            plt.plot(test.index, fc_stl, '--', label='STL-HW Forecast')
            plt.plot(test.index, fc_mean, ':', label='Mean Forecast')
            plt.title(f'Backtest: {outlet} - {product}')
            plt.xlabel('Date')
            plt.ylabel('Qty Box')
            plt.legend()
            plt.tight_layout()

    metrics_df = pd.DataFrame(results)

    # Tambah overall metrics per method
    overall = []
    for method, grp in metrics_df.groupby('method'):
        overall.append({
            'outlet': 'ALL', 'product': 'ALL', 'method': method,
            'MAE': grp['MAE'].mean(),
            'RMSE': grp['RMSE'].mean(),
            'MAPE': grp['MAPE'].mean(),
            'n_test': grp['n_test'].sum()
        })
    metrics_df = pd.concat([metrics_df, pd.DataFrame(overall)], ignore_index=True)
    return metrics_df

def main():
    file_path = 'D:/project_mostrans/data/historical_sales.xlsx'
    df = pd.read_excel(file_path, sheet_name='historical_orders')
    df['date'] = pd.to_datetime(df['date'])

    # Folder untuk menyimpan plot (pastikan sudah dibuat)
    plot_dir = 'D:/project_mostrans/outputs/plots_backtest'

    # Jalankan backtest (7 hari terakhir sebagai test)
    metrics = backtest(df, period=7, plot_output_dir=plot_dir)

    # Simpan metrics
    output_file = 'D:/project_mostrans/outputs/backtest_metrics_comparison.xlsx'
    metrics.to_excel(output_file, index=False)

    print(f"Backtest selesai. Metrics di {output_file}, plots di {plot_dir}")

if __name__ == '__main__':
    main()
