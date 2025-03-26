import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from neuralforecast import NeuralForecast
from neuralforecast.models import KAN
from neuralforecast.losses.pytorch import DistributionLoss

#Load the data
valais_dataset = pd.read_csv('../data/clean/valais_imputed.csv')
valais_small = valais_dataset[['station', 'time', 'precip']]
valais_small['ds'] = pd.to_datetime(valais_small['time'], format='%Y%m%d%H%M')
valais_small = valais_small.drop(columns=['time'])
valais_small = valais_small.rename(columns={'precip': 'y', 'station':'unique_id'})

#Split that data
valais_novalidf =  valais_small[valais_small['ds']<=pd.to_datetime('2022-12-31 23:50')]

Y_train_df = valais_small[valais_small['ds']<=pd.to_datetime('2021-12-31 23:50')]
Y_test_df =  valais_small[(valais_small['ds']>pd.to_datetime('2021-12-31 23:50')) & (valais_small['ds']<=pd.to_datetime('2022-12-31 23:50'))]
Y_validate_df = valais_small[valais_small['ds']>=pd.to_datetime('2023-01-01 00:00')]

#model definition
fcst = NeuralForecast(
    models=[
            KAN(h=72,
                input_size=144,
                loss = DistributionLoss(distribution="Normal"),
                max_steps=100,
                scaler_type='standard',
                futr_exog_list=None,
                hist_exog_list=None,
                stat_exog_list=None,
                ),     
    ],
    freq='10min'
)
fcst.fit(df=Y_train_df)
forecasts = fcst.predict(futr_df=Y_test_df)

# Prepare plot data
Y_hat_df = forecasts.reset_index(drop=False)
plot_df = pd.concat([Y_train_df, Y_test_df], axis=1)

# Plot predictions
fig, ax = plt.subplots(1, 1, figsize=(20, 7))
plt.plot(plot_df['ds'], plot_df['precip'], c='black', label='True')
plt.plot(Y_hat_df['ds'], Y_hat_df['KAN-median'], c='blue', label='Predicted')
ax.set_title('Rainfall Forecast', fontsize=22)
ax.set_ylabel('Rainfall (mm)', fontsize=20)
ax.set_xlabel('Time', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
plt.show()