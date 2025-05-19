import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

def main():
    # Load historical orders data
    file_path = 'D:/project_mostrans/data/historical_sales.xlsx'
    df = pd.read_excel(file_path, sheet_name='historical_orders')
    df['date'] = pd.to_datetime(df['date'])

    # Forecast horizon
    horizon = 7  # days ahead
    forecasts = []

    # Loop over each outlet-product combination
    for (outlet, product), group in df.groupby(['outlet', 'product']):
        ts = group.set_index('date')['qty_box'].asfreq('D')
        ts = ts.fillna(method='ffill').fillna(method='bfill')

        # STL decomposition
        stl = STL(ts, period=7, robust=True)
        res = stl.fit()
        trend = res.trend
        seasonal = res.seasonal
        resid = res.resid

        # Forecast trend with Exponential Smoothing
        hw = ExponentialSmoothing(trend.dropna(), trend='add', initialization_method='estimated')
        hw_fit = hw.fit()
        trend_fc = hw_fit.forecast(horizon)

        # Forecast seasonal by repeating the last seasonal cycle
        last_season = seasonal[-7:]
        seasonal_fc = np.tile(last_season.values, int(np.ceil(horizon / len(last_season))))[:horizon]

        # Combine components
        fc_values = trend_fc.values + seasonal_fc

        # Future dates index
        future_dates = pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')

        # Confidence intervals assuming normal residuals
        resid_std = resid.std()
        ci_lower = fc_values - 1.96 * resid_std
        ci_upper = fc_values + 1.96 * resid_std

        # Build forecast DataFrame
        df_fc = pd.DataFrame({
            'date': future_dates,
            'outlet': outlet,
            'product': product,
            'forecast_qty_box': fc_values.astype(float),
            'ci_lower': ci_lower.astype(float),
            'ci_upper': ci_upper.astype(float)
        })
        forecasts.append(df_fc)

    # Concatenate and save forecast results
    df_forecast = pd.concat(forecasts).reset_index(drop=True)
    output_file = 'D:/project_mostrans/outputs/forecast_7d.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        df_forecast.to_excel(writer, sheet_name='forecast', index=False)
    print(f"Forecast completed and saved to {output_file}")

    # Aggregate historical and forecast data per product
    hist_agg = df.groupby(['date', 'product'])['qty_box'].sum().reset_index()
    fc_agg = df_forecast.groupby(['date', 'product']).agg({
        'forecast_qty_box': 'sum',
        'ci_lower': 'sum',
        'ci_upper': 'sum'
    }).reset_index()

    # Plot historical vs forecast for each product
    products = hist_agg['product'].unique()
    for prod in products:
        # Historical series
        hist = hist_agg[hist_agg['product'] == prod].set_index('date')['qty_box']
        # Forecast series
        fc = fc_agg[fc_agg['product'] == prod].set_index('date')

        plt.figure()
        plt.plot(hist.index, hist.values, label='Historical')
        plt.plot(fc.index, fc['forecast_qty_box'].values, '--', label='Forecast')
        # Confidence interval
        plt.fill_between(fc.index,
                         fc['ci_lower'].values,
                         fc['ci_upper'].values,
                         color='gray', alpha=0.2,
                         label='95% CI')
        plt.title(f'Historical and Forecast for {prod}')
        plt.xlabel('Date')
        plt.ylabel('Quantity (boxes)')
        plt.legend()
        plt.tight_layout()
        # Save plot per product
        plt.savefig(f'D:/project_mostrans/outputs/forecast_{prod}.png')
        plt.close()

if __name__ == "__main__":
    main()
