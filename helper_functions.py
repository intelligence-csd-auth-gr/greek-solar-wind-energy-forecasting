from imports import *

class Evaluation():
   def __init__(self, train_data=None, test_data=None):
      self.model_comparison_dict = {}
      self.config = json.loads(open('config.json', 'r').read())
      self.target_dates = self.config['target_dates']
      

      if train_data is not None and test_data is not None:
         self.train_data = train_data
         self.test_data = test_data

         print_df_specs(self.train_data, 'train set')
         print_df_specs(self.test_data, 'test set')

# --------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------
   def set_data(self, train_data, test_data):
      self.train_data = train_data.copy()
      self.test_data = test_data.copy()
     

   def ts_plot(self, x, y, title, ylim=None):
      fig = go.Figure()
      fig.add_trace(go.Scatter(x=x, y=y, line_color='deepskyblue', opacity=.8))
      #fig.add_trace(go)
      fig.update_layout(title_text=title,
                  xaxis_rangeslider_visible=True)
      if ylim:
         fig.update_layout(yaxis=dict(range=ylim))
      fig.show()
      

   def ts_dualplot(self, x, y1, y2, title, l1='Line 1', l2='Line 2', color1=None, color2=None, opacity1=None, opacity2=None, ylim=None, save=True):
      fig = go.Figure()
      fig.add_trace(go.Scatter(x=x, y=y1, name=l1, 
                              line_color=color1, opacity=opacity1))

      fig.add_trace(go.Scatter(x=x, y=y2, name=l2, 
                              line_color=color2, opacity=opacity2))

      #fig.add_trace(go)
      fig.update_layout(title_text=title,
                        xaxis_rangeslider_visible=True)   
      fig.update_layout(showlegend=True)
                        
      if ylim:
         fig.update_layout(yaxis=dict(range=ylim))
      fig.show()
      if save:
         img_dir = os.path.join('data', 'images')
         if not os.path.exists(img_dir):      
            os.makedirs(img_dir)
         fig_path = os.path.join(img_dir, '%s_correlation.png' % self.model_name)
         # fig.write_image(fig_path)

   def plot_decomposition(self, decomp_result):
      recomp = (decomp_result._resid + decomp_result.seasonal + decomp_result.trend)
      self.ts_plot(recomp.index, recomp, 'Observed')
      self.ts_plot(recomp.index, decomp_result.seasonal, 'STL Seasonal')
      self.ts_plot(recomp.index, decomp_result.trend, 'STL Trend')
      self.ts_plot(recomp.index, decomp_result._resid, 'STL Residuals')
      
   # plotting predicted vs actual energy values
   def plot_prediction(self, y_test, y_pred, model_name=None, feature_importance=None, save=True):
      """
      Plots the observed energy consumption against the predicted energy consumption
      """
      self.ts_dualplot(y_test.index,
                  y_test.values,
                  y_pred,
                  title='Observed vs predicted energy values using %s' % model_name,
                  l1='Observed',
                  l2='Predicted')
      
      #---------
      # Prediction vs observed energy correlation
      fig = go.Figure()
      # Add traces
      fig.add_trace(go.Scatter(x=y_test.values, y=y_pred,
                        mode='markers',
                        name='markers'))
      x = np.linspace(0, 1)                
      fig.add_trace(go.Scatter(x=x, y=x,
                    mode='lines',
                    line=dict(color='firebrick', width=2)))

      fig.update_layout(title_text='Observed vs Predicted energy correlation {}'.format(model_name),
                 xaxis_title="Observed energy",
                 yaxis_title="Predicted energy")
      fig.show()

      if save:
         img_dir = os.path.join('data', 'images')
         if not os.path.exists(img_dir):      
            os.makedirs(img_dir)
         fig_path = os.path.join(img_dir, '%s_forecast.png' % self.model_name)
         # fig.write_image(fig_path)


   def final_feature_selection(self, model, x_train, y_train):
      # for time-series cross-validation set 5 folds
      tscv = TimeSeriesSplit(n_splits=5)
      # Train RFE feature selectior with cross validation
      selector = RFECV(model, 
                     step=1, 
                     cv=tscv,
                     min_features_to_select = 10,
                     scoring='neg_mean_squared_error')

      selector.fit(x_train, y_train)
      features_selected = x_train.columns[selector.support_].tolist() 
      return features_selected


   def features_eval(self, feat_names, feat_values, model_name):
      # Plotting the coefficients to check the importance of each coefficient 
      # Plot the coefficients
      _ = plt.figure(figsize = (16, 7))
      _ = plt.bar(range(len(feat_names)), feat_values)
      _ = plt.xticks(range(len(feat_names)), feat_names, rotation = 85)
      _ = plt.margins(0.02)
      _ = plt.axhline(0, linewidth = 0.5, color = 'r')
      _ = plt.title('Feature importance of %s' % model_name)
      _ = plt.ylabel('lm_coeff')
      _ = plt.xlabel('Features')
      _ = plt.show()     

   # Metrics function
   def get_metrics(self, y_test, y_pred):
      rmse = np.sqrt(mean_squared_error(y_test, y_pred))
      print('Root Mean Squared Error (RMSE): %.5f' % rmse)

      mae = mean_absolute_error(y_test, y_pred)
      print('Mean Absolute Error (MAE): %.5f' % mae)

      return {'MAE':np.round(mae, 5), 
            'RMSE':np.round(rmse, 5),
            'time':y_test.index,
            'absolute_diff': np.abs(y_pred.values - y_test.values),
            'squared_root_diff': np.sqrt(np.abs(y_pred.values**2 - y_test.values**2))}

   # wrapper function for model evaluation
   def model_eval(self, forecast_df, model_name, plot=False, save=True):
      self.model_name = model_name
      y_test, y_pred = forecast_df['y_test'], forecast_df['y_pred']
      if plot:
         self.plot_prediction(y_test, y_pred, model_name = model_name)
      self.model_comparison_dict[model_name] = self.get_metrics(y_test, y_pred)
      if save:
         self.save_forecast(forecast_df, model_name)


   def plot_ts_decomp(self, stl_result):   
      fig, ax = plt.subplots(4,1, figsize=(15,10))
      _ = ax[0].plot(stl_result.observed)
      _ = ax[0].set_title('Observed')
      _ = ax[1].plot(stl_result.trend)
      _ = ax[1].set_title('Trend')
      _ = ax[2].plot(stl_result.seasonal)
      _ = ax[2].set_title('Seasonal')
      _ = ax[3].plot(stl_result.resid)
      _ = ax[3].set_title('Residual')
      _ = fig.tight_layout()
      plt.show()


   def plot_corr_heatmap(self, df, features, interval=None, vmin=None, vmax=None):
      # Generate a mask for the upper triangle
      corr_mat = df[features].corr().round(2)
      mask = np.triu(np.ones_like(corr_mat, dtype=bool))

      # Set up the matplotlib figure
      _, _ = plt.subplots(figsize=(11, 9))

      # Generate a custom diverging colormap
      cmap = sns.diverging_palette(230, 20, as_cmap=True)

      # Draw the heatmap with the mask and correct aspect ratio
      sns.heatmap(corr_mat, mask=mask, cmap=cmap, annot= True, vmin=vmin, vmax=vmax,
                  square=True, linewidths=.5, cbar_kws={"shrink": .5})


   def save_forecast(self, df, model_name):
      energy_type = 'solar' if 'solar' in model_name.lower() else 'wind'

      forecast_dir = os.path.join('data', 'forecast_output', energy_type)
      if not os.path.exists(forecast_dir):      
         os.makedirs(forecast_dir)
      df.to_csv(os.path.join(forecast_dir, '%s.csv' % model_name))
      


   def hourly_train_split(self, hour, target_column, train_data=[]):
      if len(train_data)==0:
         train_data = self.train_data.copy()

      train_subset = train_data.loc[train_data.index.hour==hour].copy()
      x_train, y_train = train_subset.drop(columns=target_column), train_subset[target_column]
      return x_train, y_train


   def hourly_test_split(self, hour, target_column, test_data=[]):
      if len(test_data)==0:
         test_data = self.test_data.copy()

      test_subset = test_data.loc[test_data.index.hour==hour].copy()
      x_test, y_test = test_subset.drop(columns=target_column), test_subset[target_column]
      return x_test, y_test

   
   def train_split(self, target_column, train_data=[]):
      if len(train_data)==0:
         train_data = self.train_data.copy()
      train_subset = train_data.copy()
      x_train, y_train = train_subset.drop(columns=target_column), train_subset[target_column]
      return x_train, y_train


   def get_model(self, model_name):
      model_path = [model['model_path'] for model in self.config['models']['models_list'] 
                     if model_name == model['model_name']][0]
      model = pickle.load(open(str(model_path).replace('\\', os.sep), 'rb'))
      return model
      
# -------------------------------
# Forecasting Functions
# -------------------------------      
   def forecast_hour(self, model_type, curr_timestamp, target_column):
      # testing set split
      test_subset = self.test_data.loc[curr_timestamp].copy()
      x_test, y_test = test_subset.drop(columns=target_column), test_subset[target_column]
      
      hour = pd.to_datetime(curr_timestamp).hour
      model = self.get_model('%s-%sh' %(model_type,hour))
      y_pred = model.predict(x_test.values.reshape(1, -1))[0]      
      return y_test, y_pred   


   def forecast_instance(self, model_type, instance, target_column):
      # testing set split
      if 'Hourly' in model_type:
         test_subset = self.test_data.loc[instance].copy()
         x_test, y_test = test_subset.drop(target_column), test_subset[target_column]
         hour = pd.to_datetime(instance).hour
         model = self.get_model('%s-%sh' %(model_type,hour))
         y_pred = model.predict(x_test.values.reshape(1, -1))[0]     
      else:
         test_subset = self.test_data.loc[instance].copy()
         x_test, y_test = test_subset.drop(target_column), test_subset[target_column]      
         model = self.get_model('%s' %(model_type))
         y_pred = model.predict(x_test.values.reshape(1, -1))
      return y_test, y_pred


   def forecast_hour2year(self, model_type, target_column, eval=True, correction=True):
      print("")
      print("Starting yearly forcast with %s..." % model_type)
      start = datetime.now()
      test_days = sorted(set(self.test_data.index.date))

      pred_dict = {'y_pred': [], 'y_test': [], 'time': []}
      for test_day in test_days:    
         for curr_timestamp in self.test_data[str(test_day)].index:
            # Apply forecast
            y_test, y_pred = self.forecast_instance(model_type, curr_timestamp, target_column)
            try:
               pred_dict['y_test'].append(y_test.values)
            except:
               pred_dict['y_test'].append(y_test)
            pred_dict['y_pred'].append(y_pred)
            pred_dict['time'].append(curr_timestamp)
            
      forecast_df = pd.DataFrame.from_dict(pred_dict)
      forecast_df.set_index('time', inplace=True)
      forecast_df['y_pred'] = np.hstack(forecast_df['y_pred'].values)
      forecast_df['y_test'] = np.hstack(forecast_df['y_test'].values)

      # Prediction correction
      if correction:
         forecast_df.loc[forecast_df['y_pred']<0, 'y_pred'] = 0.

      if eval: 
         self.model_eval(forecast_df, model_type)
      print('Finished forecasting in %s' % str(datetime.now()-start)) 
      return forecast_df


   def forecast_year(self, model_type, target_column, freq, eval=True, correction=True):
         print("")
         print("Starting yearly forcast with %s..." % model_type)
         start = datetime.now()
         test_days = pd.date_range(start=self.test_data.index[0], end=self.test_data.index[-1], freq='%sH' % freq)

         pred_dict = {'y_pred': [], 'y_test': [], 'time': []}
         for test_day in test_days:
            # Apply forecast
            y_test, y_pred = self.forecast_instance(model_type, test_day, target_column)
            if type(y_test) == np.array:
                  pred_dict['y_test'] += y_test.flatten()
            elif type(y_test) == pd.Series:
               pred_dict['y_test'] += y_test.values.flatten().tolist()
            else:
               pred_dict['y_test'] += [y_test]
            pred_dict['y_pred'] += y_pred.flatten().tolist()
            date_range = pd.date_range(start=test_day, end=test_day+timedelta(hours=freq-1), freq='1H')
            pred_dict['time'] += date_range

         # Create forecast
         forecast_df = pd.DataFrame.from_dict(pred_dict)      
         forecast_df['y_pred'] = np.hstack(forecast_df['y_pred'].values)
         forecast_df['y_test'] = np.hstack(forecast_df['y_test'].values)
         forecast_df['time'] = np.hstack(forecast_df['time'].values)
         forecast_df.set_index('time', inplace=True)
         forecast_df.dropna(inplace=True)
         # Prediction correction
         if correction:
            forecast_df.loc[forecast_df['y_pred']<0, 'y_pred'] = 0.

         if eval: 
            self.model_eval(forecast_df, model_type, plot=True)
         print('Finished forecasting in %s' % str(datetime.now()-start)) 
         return forecast_df

   def forecast_year_raw_prophet(self, model_type, target_column, hybrid=False, hybrid_model=None, eval=True, training_flag_prophet=True, training_flag=True):
      print("")
      print("Starting yearly forcast with %s..." % model_type)      
      start = datetime.now()

      # Store original training and testing sets
      orig_train_data = self.train_data.copy()
      orig_test_data = self.test_data.copy()

      # Prepare prediction dataframe
      pred_dict = {'y_test':[], 'y_pred':[]}
      # Initiate FB Prophet
      prophet, p_train_data, p_test_data = self.raw_fbProphet_init(target_column)
      prophet_name = 'Raw Solar FB Prophet' if 'Solar' in model_type else 'Raw Wind FB Prophet'
      if training_flag_prophet:
         print('Training FB Prophet')
         prophet.fit(p_train_data)         
         model_path = self.set_model_info(prophet_name)
         pickle.dump(prophet, open(model_path, 'wb'))
         print('Finished training %s in %s' % (model_type, str(datetime.now()-start)) )

      prophet = self.get_model(prophet_name)
      merged_df = pd.concat([p_train_data, p_test_data]).reset_index()
      y_pred_prophet = prophet.predict(merged_df)

      if hybrid:
         # Detrend data         
         trend = y_pred_prophet.trend + y_pred_prophet.additive_terms

         merged_df.loc[:, 'y'] = merged_df['y'] - trend            
         self.train_data.loc[:, target_column] = merged_df.loc[:len(self.train_data)-1, 'y'].values
         self.test_data.loc[:, target_column] = merged_df.loc[len(self.train_data): , 'y'].values

         # Training classic regressor
         if training_flag:
            x_train, y_train = self.train_split(target_column)
            hybrid_model.fit(x_train, y_train)               
            model_path = self.set_model_info('%s' % model_type)
            pickle.dump(hybrid_model, open(model_path, 'wb'))

         # Predict
         test_subset = self.test_data.copy()
         x_test, y_test = test_subset.drop(columns=target_column), test_subset[target_column]      
         model = self.get_model('%s' %(model_type))
         y_pred = model.predict(x_test.values)
         pred_dict = {'y_pred': y_pred, 'y_test': y_test, 'time': y_test.index}
         # Create forecast
         forecast_df = pd.DataFrame.from_dict(pred_dict)      
         forecast_df['y_pred'] = forecast_df['y_pred'].values.ravel()
         forecast_df['time'] = forecast_df['time'].values.ravel()
         forecast_df.set_index('time', inplace=True)
         forecast_df.dropna(inplace=True)
         forecast_df['y_pred'] += trend.loc[len(self.train_data):].values
         forecast_df['y_test'] = orig_test_data[target_column].values.ravel()

      else:
         # Add prediction lower and upper bounds at 0
         for col in y_pred_prophet.columns[1:]:
            y_pred_prophet.loc[y_pred_prophet[col] < p_train_data['y'].min(), col] = p_train_data['y'].min()
            # y_pred_prophet.loc[y_pred_prophet[col] > p_train_data['y'].max(), col] = p_train_data['y'].max()
         y_pred = y_pred_prophet.set_index('ds').iloc[len(p_train_data):]['yhat']
         y_pred.index = y_pred.index.tz_localize('UTC').tz_convert('CET')
         pred_dict['y_pred'] = y_pred.values
         pred_dict['y_test'] = orig_test_data[target_column]
         pred_dict['time'] = orig_test_data.index
         forecast_df = pd.DataFrame.from_dict(pred_dict).set_index('time')
      
      if eval: 
         # Prediction correction
         forecast_df.loc[forecast_df['y_pred']<0, 'y_pred'] = 0
         self.model_eval(forecast_df, model_type)
      print('Finished forecasting in %s' % str(datetime.now()-start)) 
      return forecast_df

   def forecast_year_prophet(self, regressors, model_type, target_column, hybrid=False, hybrid_model=None, eval=True, training_flag_prophet=True, training_flag=True):
      print("")
      print("Starting yearly forcast with %s..." % model_type)      
      start = datetime.now()

      # Store original training and testing sets
      orig_train_data = self.train_data.copy()
      orig_test_data = self.test_data.copy()

      # Prepare prediction dataframe
      pred_dict = {'y_test':[], 'y_pred':[]}
      # Initiate FB Prophet
      prophet, p_train_data, p_test_data = self.fbProphet_init(regressors, self.train_data.columns, target_column)
      prophet_name = 'Solar FB Prophet' if 'Solar' in model_type else 'Wind FB Prophet'
      if training_flag_prophet:
         print('Training FB Prophet')
         prophet.fit(p_train_data)         
         model_path = self.set_model_info(prophet_name)
         pickle.dump(prophet, open(model_path, 'wb'))
         print('Finished training %s in %s' % (model_type, str(datetime.now()-start)) )

      prophet = self.get_model(prophet_name)
      merged_df = pd.concat([p_train_data, p_test_data]).reset_index()
      y_pred_prophet = prophet.predict(merged_df)

      if hybrid:
         # Detrend data   
         if 'wind' in model_type.lower():       
            trend = y_pred_prophet.trend + y_pred_prophet.additive_terms
         else:
            trend = y_pred_prophet.trend + y_pred_prophet.daily + y_pred_prophet.morning \
                  + y_pred_prophet.night + y_pred_prophet.noon + y_pred_prophet.summer + y_pred_prophet.sunrise \
                  + y_pred_prophet.sunset + y_pred_prophet.weekly + y_pred_prophet.winter + y_pred_prophet.yearly
         merged_df.loc[:, 'y'] = merged_df['y'] - trend            
         self.train_data.loc[:, target_column] = merged_df.loc[:len(self.train_data)-1, 'y'].values
         self.test_data.loc[:, target_column] = merged_df.loc[len(self.train_data): , 'y'].values

         # Training classic regressor
         if training_flag:
            x_train, y_train = self.train_split(target_column)
            hybrid_model.fit(x_train, y_train)               
            model_path = self.set_model_info('%s' % model_type)
            pickle.dump(hybrid_model, open(model_path, 'wb'))

         # Predict
         test_subset = self.test_data.copy()
         x_test, y_test = test_subset.drop(columns=target_column), test_subset[target_column]      
         model = self.get_model('%s' %(model_type))
         y_pred = model.predict(x_test.values)
         pred_dict = {'y_pred': y_pred, 'y_test': y_test, 'time': y_test.index}
         # Create forecast
         forecast_df = pd.DataFrame.from_dict(pred_dict)      
         forecast_df['y_pred'] = forecast_df['y_pred'].values.ravel()
         forecast_df['time'] = forecast_df['time'].values.ravel()
         forecast_df.set_index('time', inplace=True)
         forecast_df.dropna(inplace=True)
         forecast_df['y_pred'] += trend.loc[len(self.train_data):].values
         forecast_df['y_test'] = orig_test_data[target_column].values.ravel()

      else:
         # Add prediction lower and upper bounds at 0
         for col in y_pred_prophet.columns[1:]:
            y_pred_prophet.loc[y_pred_prophet[col] < p_train_data['y'].min(), col] = p_train_data['y'].min()
            # y_pred_prophet.loc[y_pred_prophet[col] > p_train_data['y'].max(), col] = p_train_data['y'].max()
         y_pred = y_pred_prophet.set_index('ds').iloc[len(p_train_data):]['yhat']
         y_pred.index = y_pred.index.tz_localize('UTC').tz_convert('CET')
         pred_dict['y_pred'] = y_pred.values
         pred_dict['y_test'] = orig_test_data[target_column]
         pred_dict['time'] = orig_test_data.index
         forecast_df = pd.DataFrame.from_dict(pred_dict).set_index('time')
      
      if eval: 
         # Prediction correction
         forecast_df.loc[forecast_df['y_pred']<0, 'y_pred'] = 0
         self.model_eval(forecast_df, model_type)
      print('Finished forecasting in %s' % str(datetime.now()-start)) 
      return forecast_df


   def set_model_info(self, model_name):
      models_dir = self.config['models']['models_dir']
      model_dir = os.path.join(models_dir, model_name.split('-')[0])
      model_path = os.path.join(model_dir, '%s.pkl' % model_name)
      models_dict = {'model_type': model_name.split('-')[0], 'model_name': model_name,'model_path': model_path}
      
      # Create model directory
      if not os.path.exists(model_dir):      
         os.makedirs(model_dir)
      # remove previous record
      if models_dict in self.config['models']['models_list']:
         self.config['models']['models_list'].remove(models_dict )
      # Update model list
      self.config['models']['models_list'].append(models_dict)
      with open('config.json', 'w') as config_file:
         json.dump(self.config, config_file)
      return model_path



   def raw_fbProphet_init(self, target_column):
      prophet = Prophet(
          growth='linear',
          daily_seasonality=True,
          weekly_seasonality=True,
          yearly_seasonality=True,
          changepoint_prior_scale=0.001,
          seasonality_mode='additive',
      )

      prophet.add_seasonality(
          name='daily',
          period=1,
          fourier_order=2,
      )

      prophet.add_seasonality(
          name='weekly',
          period=7,
          fourier_order=10,
      )

      prophet.add_seasonality(
          name='yearly',
          period=366,
          fourier_order=40,
      )


      # Localize timezone
      train_data = self.train_data.copy()
      train_data = (train_data.reset_index()
                        .rename(columns = {target_column:'y', 'time':'ds'}))
      train_data.loc[:, 'ds'] = self.train_data.index.tz_localize(None)
      test_data = self.test_data.copy()
      test_data = (test_data.reset_index()
                        .rename(columns = {target_column:'y', 'time':'ds'}))
      test_data.loc[:, 'ds'] = self.test_data.index.tz_localize(None)

      return prophet, train_data, test_data

   def fbProphet_init(self, regressors, features, target_column):
      prophet = Prophet(
          growth='linear',
          daily_seasonality=True,
          weekly_seasonality=True,
          yearly_seasonality=True,
          changepoint_prior_scale=0.001,
          seasonality_mode='additive',
      )

      # Adding seasonalities
      if 'season_summer' in features:
         prophet.add_seasonality(
            name='summer', 
            period=6,
            fourier_order=2, 
            condition_name='season_summer')

      if 'season_winter' in features:
         prophet.add_seasonality(
            name='winter', 
            period=6, 
            fourier_order=2, 
            condition_name='season_winter')

      if 'timeofday_morning' in features:
         prophet.add_seasonality(
            name='morning', 
            period=.5, 
            fourier_order=20, 
            condition_name='timeofday_morning')

      if 'timeofday_night' in features:
         prophet.add_seasonality(
            name='night', 
            period=.5, 
            fourier_order=20, 
            condition_name='timeofday_night')

      if 'timeofday_noon' in features:
         prophet.add_seasonality(
            name='noon', 
            period=.5, 
            fourier_order=20, 
            condition_name='timeofday_noon')

      if 'timeofday_sunrise' in features:
         prophet.add_seasonality(
            name='sunrise', 
            period=.5, 
            fourier_order=20, 
            condition_name='timeofday_sunrise')

      if 'timeofday_sunset' in features:
         prophet.add_seasonality(
            name='sunset', 
            period=.5, 
            fourier_order=20, 
            condition_name='timeofday_sunset')

      prophet.add_seasonality(
          name='daily',
          period=1,
          fourier_order=2,
      )

      prophet.add_seasonality(
          name='weekly',
          period=7,
          fourier_order=10,
      )

      prophet.add_seasonality(
          name='yearly',
          period=366,
          fourier_order=40,
      )

      # Adding external regressors
      for reg in regressors:
         prophet.add_regressor(reg, prior_scale=20, mode='additive', standardize='auto')

      # Localize timezone
      train_data = self.train_data.copy()
      train_data = (train_data.reset_index()
                        .rename(columns = {target_column:'y', 'time':'ds'}))
      train_data.loc[:, 'ds'] = self.train_data.index.tz_localize(None)
      test_data = self.test_data.copy()
      test_data = (test_data.reset_index()
                        .rename(columns = {target_column:'y', 'time':'ds'}))
      test_data.loc[:, 'ds'] = self.test_data.index.tz_localize(None)

      return prophet, train_data, test_data


   def _stl_decompose(self, data, seasonal, period, trend, stl_name, robust=True, training_stl=True):      
      print('Starting STL decomposition')
      start = datetime.now()
      if training_stl:         
         stl_result = STL(data, 
                  seasonal=seasonal, 
                  period=period, 
                  trend=trend,
                  robust=robust).fit()
         model_path = self.set_model_info(stl_name)
         pickle.dump(stl_result, open(model_path, 'wb'))

      stl_result = self.get_model(stl_name)      
      print('Finished STL decomposition in %s' % (str(datetime.now()-start))) 
      return stl_result


   def _seasonal_forecast(self,  stl_result, period, steps, index=None):
      """
      Get the seasonal component of the forecast
      Parameters
      ----------
      steps : int
         The number of steps required.
      index : pd.Index
         A pandas index to use. If None, returns an ndarray.
      offset : int
         The index of the first out-of-sample observation. If None, uses
         nobs.
      Returns
      -------
      seasonal : {ndarray, Series}
         The seasonal component.
      """
      seasonal = np.asarray(stl_result.seasonal)
      period = period
      seasonal.shape

      seasonal = np.tile(seasonal, steps // period + ((steps % period) != 0))
      seasonal = seasonal[:steps]
      if index is not None:
         seasonal = pd.Series(seasonal, index=index)
      return seasonal


   def stl_forecast(self, target_column, seasonal, period, trend, model, model_type, robust=True, training_stl=True, training_flag=True, correction=True):
      stl_name = 'Solar STL Decomposition' if 'Solar' in model_type else 'Wind STL Decomposition'
      # STL decomposition
      stl_result = self._stl_decompose(self.train_data[target_column], seasonal, period, trend, stl_name, robust, training_stl)
      
      # Seasonal component forecast
      seasonal = self._seasonal_forecast( stl_result, period, self.test_data.shape[0], self.test_data.index)
      
      # De-season data
      seasoned_cols = self.test_data.filter(regex='t[\+\-].+energy').columns
      for col in seasoned_cols:
         self.train_data['%s-deseasoned' % col] = self.train_data[col].values - stl_result.seasonal
         self.test_data['%s-deseasoned' % col] = self.test_data[col].values - seasonal
      
      # Set training and test data
      deseasoned_cols = self.test_data.filter(regex='t[\+\-].+deseasoned').columns
      new_target_variable = self.train_data.filter(regex='t\+.+deseasoned').columns.tolist()
      new_columns = sorted(set(self.train_data.columns) - set(seasoned_cols) - set(new_target_variable))
      og_columns = sorted(set(self.train_data.columns) - set(deseasoned_cols) - set(target_column))

      x_train, y_train = self.train_data[new_columns], self.train_data[new_target_variable]
      x_test, y_test = self.test_data[new_columns], self.test_data[new_target_variable]

      # Train model
      if training_flag:
         print('Training model %s...' % (model_type))
         start = datetime.now()
         model.fit(x_train, y_train)
         print('Finished training %s in %s' % (model_type, str(datetime.now()-start))) 

         start = datetime.now()
         print('Saving model %s...' % (model_type))
         model_path = self.set_model_info('%s' % model_type)
         pickle.dump(model, open(model_path, 'wb'))
         print('Finished saving %s in %s\n' % (model_type, str(datetime.now()-start))) 
     
      # Forecast
      start = datetime.now()
      model = self.get_model(model_type)
      y_pred = model.predict(x_test)
      y_pred_seasoned = seasonal + y_pred.ravel()
      y_test = self.test_data[target_column]
      pred_dict = {'y_pred': y_pred_seasoned, 'y_test': y_test.values.ravel(), 'time': y_test.index}

      forecast_df = pd.DataFrame.from_dict(pred_dict)
      forecast_df.set_index('time', inplace=True)
      if correction:
         forecast_df.loc[forecast_df['y_pred']<0, 'y_pred'] = 0.
      self.model_eval(forecast_df, model_type) 
      print('Finished forecasting in %s' % str(datetime.now()-start)) 
      return forecast_df

# ------------------------------------------
# Baseline Forecasting Functions
# ------------------------------------------
   def persistence_forecast(self, target_column, model_type, lag, data=[]):
      if len(data)==0:
         data = self.test_data.copy()
      delta = timedelta(days=lag)
      data['y_test'] = data[target_column]
      data['y_pred'] = data['y_test'].shift(freq=delta)
      forecast_df = data[['y_test', 'y_pred']].dropna()
      
      self.model_eval(forecast_df, model_type)
      return forecast_df
    
   def historic_average_forecast(self, target_column, model_type):
      test_dates = np.unique(self.test_data.index)
      pred_dic = {'y_pred': [], 'y_test': [], 'time': []}
      for instance in test_dates:
         pred_dic['y_pred'] += self.train_data.loc[((self.train_data.index.hour == instance.hour) & 
                                                    (self.train_data.index.day == instance.day) & 
                                                    (self.train_data.index.month == instance.month)), 
                                                            target_column].mean().values.tolist()
         pred_dic['y_test'] += self.test_data.loc[instance, target_column].values.tolist()
         pred_dic['time'] += [instance]
      forecast_df = pd.DataFrame.from_dict(pred_dic)
      forecast_df['y_test'] = forecast_df['y_test'].values.ravel()
      forecast_df['y_pred'] = forecast_df['y_pred'].values.ravel()
      forecast_df.set_index('time', inplace=True)
      forecast_df.dropna(inplace=True)
      self.model_eval(forecast_df, model_type)
      return forecast_df

   def ewma_forecast(self, target_column, model_type, alpha, adjust):
      # Exponential window moving average
      if isinstance(target_column, list) and len(target_column)==1:
         target_column = target_column[0]
      # Set data
      y_train = self.train_data[target_column]
      y_train = y_train.reset_index().drop(columns='time').squeeze()
      y_test = self.test_data[target_column]
      fh = len(y_test)
       
      # Predict
      y_pred = y_train.ewm(alpha=alpha, adjust=adjust).mean()[:fh]
      pred_dic = {'y_pred': y_pred.values, 'y_test': y_test.values, 'time': y_test.index}
      forecast_df = pd.DataFrame.from_dict(pred_dic)
      forecast_df.set_index('time', inplace=True)
      self.model_eval(forecast_df, model_type)
      return forecast_df

   def hw_exponential_smoothing_forecast(self, target_column, model_type, 
                                           seasonal_periods=24, trend=None, seasonal=None, 
                                           smoothing_level=None, smoothing_trend=None, smoothing_seasonal=None):
      if isinstance(target_column, list) and len(target_column)==1:
         target_column = target_column[0]
      # Set data
      y_train = self.train_data[target_column]
      y_train = y_train.reset_index().drop(columns='time').squeeze()
      y_test = self.test_data[target_column]
      fh = len(y_test)
      
      # Predict
      model = ExponentialSmoothing(y_train, 
                                 seasonal_periods=seasonal_periods, 
                                 trend=trend, seasonal=seasonal, damped_trend=True)
      predict = model.fit(use_brute=True, optimized=True, 
                          smoothing_level=smoothing_level, 
                          smoothing_trend=smoothing_trend, 
                          smoothing_seasonal=smoothing_seasonal)
      y_pred = predict.forecast(fh)
      pred_dic = {'y_pred': y_pred.values, 'y_test': y_test.values, 'time': y_test.index}
      forecast_df = pd.DataFrame.from_dict(pred_dic)
      forecast_df.set_index('time', inplace=True)
      self.model_eval(forecast_df, model_type)
      return forecast_df
 

   def naive_forecast(self, target_column, model_type, strategy='last', window_length=None, sp=1):
      if isinstance(target_column, list) and len(target_column)==1:
         target_column = target_column[0]
      # Set data
      y_train = self.train_data[target_column]
      y_train = y_train.reset_index().drop(columns='time').squeeze()
      y_test = self.test_data[target_column]
      fh = list(range(y_train.index[-1] + 1, y_train.index[-1] + len(y_test) + 1))
      
      # Train
      forecaster = NaiveForecaster(strategy=strategy, window_length=window_length, sp=sp)
      forecaster.fit(y_train)
      
      # Predict
      y_pred = forecaster.predict(fh)
      pred_dic = {'y_pred': y_pred.values, 'y_test': y_test.values, 'time': y_test.index}
      forecast_df = pd.DataFrame.from_dict(pred_dic)
      forecast_df.set_index('time', inplace=True)
      self.model_eval(forecast_df, model_type)
      return forecast_df
   

def print_df_specs(df, message=''):
   print('Shape of %s dataframe: %s,\nSize in memory (MB): %.2f' % (message, 
                                                                   df.shape, 
                                                                   df.memory_usage().sum()/1e6))

def floor_date(date):
   date = pd.to_datetime(date)
   return date - timedelta(hours=date.time().hour, 
                         minutes=date.time().minute, 
                         seconds=date.time().second, 
                         microseconds=date.time().microsecond)

def ceil_date(date):
   date = floor_date(date)   
   return date + timedelta(hours=23)
