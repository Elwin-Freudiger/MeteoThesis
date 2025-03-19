import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import RMoK
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
from neuralforecast.losses.pytorch import MSE
from neuralforecast.losses.pytorch import MAE

"""
Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = RMoK(h=12,
             input_size=24,
             n_series=2,
             taylor_order=3,
             jacobi_degree=6,
             wavelet_function='mexican_hat',
             dropout=0.1,
             revin_affine=True,
             loss=MSE(),
             valid_loss=MAE(),
             early_stop_patience_steps=3,
             batch_size=32)

fcst = NeuralForecast(models=[model], freq='ME')
fcst.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
forecasts = fcst.predict(futr_df=Y_test_df)

# Plot predictions
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['RMoK'], c='blue', label='Forecast')
ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Year', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
plt.show()
"""

# Load the data
df_crypto = pd.read_csv("../Crypto_data.csv")

# Prepare data in NeuralForecast format
Y_train_df = pd.DataFrame({
    'ds': pd.date_range(start='2025-01-01', periods=1296, freq='min'),
    'y': df_crypto['ETH'][:1296],
    'unique_id': 'ETH'
})

Y_test_df = pd.DataFrame({
    'ds': pd.date_range(start='2025-01-01', periods=1440, freq='min')[1296:],
    'y': df_crypto['ETH'][1296:],
    'unique_id': 'ETH'
})

# Model definition
model = RMoK(h=144,
             input_size=144,
             n_series=1,
             taylor_order=3,
             jacobi_degree=6,
             wavelet_function='mexican_hat',
             dropout=0.1,
             revin_affine=True,
             loss=MSE(),
             valid_loss=MAE(),
             early_stop_patience_steps=5,
             batch_size=100)

# Fit and predict
fcst = NeuralForecast(models=[model], freq='min')
fcst.fit(df=Y_train_df, val_size=144)
forecasts = fcst.predict(futr_df=Y_test_df)

# Prepare plot data
Y_hat_df = forecasts.reset_index(drop=False)
plot_df = pd.concat([Y_train_df, Y_test_df], axis=1)

# Plot predictions
fig, ax = plt.subplots(1, 1, figsize=(20, 7))
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(Y_hat_df['ds'], Y_hat_df['RMoK'], c='blue', label='Predicted')
ax.set_title('ETH Forecast', fontsize=22)
ax.set_ylabel('ETH Price', fontsize=20)
ax.set_xlabel('Time', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
plt.show()