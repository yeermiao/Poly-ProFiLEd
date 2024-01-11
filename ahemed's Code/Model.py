import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
from sklearn.svm import SVR
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError

    
def plot_history(history, key):
  plt.plot(history.history[key])
  plt.plot(history.history['val_'+key])
  plt.xlabel("Epochs")
  plt.ylabel(key)
  plt.legend([key, 'val_'+key])
  plt.show()
  
pd.options.mode.chained_assignment = None  # default='warn'
# _____________________________________Datenbereinigung/ Vorbereitung_____________________________________
# Funktion zur Datenbereinigung
def clean_data(data):
    condition_σ_surf = (data['σ_surf'] <= -300) | (data['σ_surf'] >= 300)
    condition_R_squared = data['R²'] < 0.9
    data = data[~(condition_σ_surf | condition_R_squared)]
    data['Spannungskomp.'] = data['Spannungskomp.'].astype('category')
    data['Versuch'] = data['Versuch'].astype('category')
    data['Spannungskomp.'] = data['Spannungskomp.'].cat.codes # Das Hinzufügen von .loc[:, ...] stellt sicher, dass die Operationen direkt auf dem Originaldatenrahmen durchgeführt werden
    data['Versuch'] = data['Versuch'].cat.codes
    columns_to_convert = ['Zahnvorschub']
    data[columns_to_convert] = data[columns_to_convert].apply(lambda x: x.str.replace(',', '.').astype(float))
    return data

# Funktion zur Datenaufteilung
def split_data(data):
    data_D16 = data[data['Durchmesser'] == 16]
    data_D50 = data[data['Durchmesser'] == 50]
    return data_D16, data_D50

# Funktion zur Ausgabe der Modellmetriken
def print_model_metrics(model_name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name}:")
    print("MSE:", mse)
    print("MAE:", mae)
    print("R2:", r2)

# _____________________________________Modelle_____________________________________
# rf_param_dist, xgb_param_dist
# Funktion zur Modellierung
def train_models(X_train, y_train, X_test, y_test, best_rf_params, best_xgb_params, best_rf_model, best_xgb_model):
    # Random Forest Regressor (Vanilla)
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print_model_metrics("Random Forest (Vanilla)", y_test, y_pred_rf)
    # Random Forest Modell (Hyperparameter-Tuning) 
    best_rf_model.fit(X_train, y_train)
    y_pred_rf = best_rf_model.predict(X_test)
    print_model_metrics("Random Forest (Hyperparam-tuning)", y_test, y_pred_rf)
    # Speichere das beste Modell und seine besten Hyperparameter
    best_rf_model_info = {
        'model': best_rf_model,
        'best_params': best_rf_params
    }

    # XGBoost Regressor (Vanilla)
    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    print_model_metrics("XGBoost (Vanilla)", y_test, y_pred_xgb)
    # XGBoost Modell (Hyperparameter-Tuning) 
    best_xgb_model.fit(X_train, y_train)
    y_pred_xgb = best_xgb_model.predict(X_test)
    print_model_metrics("XGBoost (Hyperparam-tuning)", y_test, y_pred_xgb)
    # Speichere das beste Modell und seine besten Hyperparameter
    best_xgb_model_info = {
        'model': best_xgb_model,
        'best_params': best_xgb_params
    }

    # SVR
    svm_models = {}
    column = y_train.columns
    for column in y_train.columns:
        svr_model = SVR()
        svr_model.fit(X_train, y_train[column])
        y_pred_svr = svr_model.predict(X_test)
        print_model_metrics(f"Support Vector Regressor for Column {column}", y_test[column], y_pred_svr)
        svm_models[column] = svr_model

    # Neuronales Netz
    nn_model = Sequential()
    nn_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    nn_model.add(Dense(32, activation='relu'))
    nn_model.add(Dense(y_train.shape[1]))  # Anzahl der Ausgabeneuronen anpassen
    nn_model.compile(loss='mean_squared_error', optimizer='adam')
    nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    y_pred_nn = nn_model.predict(X_test)
    print_model_metrics("Neural Network", y_test, y_pred_nn)

    # Gib die besten Modelle und deren Hyperparameter zurück
    return best_rf_model_info, best_xgb_model_info, svm_models, nn_model

# Funktion für das Hyperparameter-Tuning
def hyperparam_tuning_RSCV(X_train, y_train, model, param_dist, n_iter, scoring):
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=5,
        n_jobs=-1,
        scoring=scoring
    )
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    return best_params, best_model

# Funktion für das Hyperparameter-Tuning mit HalvingGridSearchCV
def hyperparam_tuning_HGSCV(X_train, y_train, model, param_grid, scoring):
    grid_search = HalvingGridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    return best_params, best_model

def define_hyperparameter_spaces():
    # Hyperparameter-Raum für den Random Forest definieren
    rf_param_dist = {
        'n_estimators': [5, 10, 50, 100, 200],
        'max_depth': [None, 5, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10, 12]
    }
    
    # Hyperparameter-Raum für XGBoost definieren
    xgb_param_dist = {
        'n_estimators': [2, 5, 10, 50, 100, 200],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    }
    return rf_param_dist, xgb_param_dist

def scale_datasets(x_train, x_test):
	
  """
  Standard Scale test and train data
  Z - Score normalization
  """
  standard_scaler = StandardScaler()
  x_train_scaled = pd.DataFrame(
      standard_scaler.fit_transform(x_train),
      columns=x_train.columns
  )
  x_test_scaled = pd.DataFrame(
      standard_scaler.transform(x_test),
      columns = x_test.columns
  )
  return x_train_scaled, x_test_scaled



#######################################################################################
# Neurales Netzwerk
#######################################################################################

DATA_BASE_FILE_PATH = '.\\Reduzierte_Datenbank'
ACCUM_EXCEL_FILE_PATH = DATA_BASE_FILE_PATH + '\\Kumulierte_res_Dateien.xlsx'
RES_FILE_PATH = DATA_BASE_FILE_PATH + '\\Einzelne_res_Dateien'
SDCF_FILE_PATH = f'{DATA_BASE_FILE_PATH}\\SDCF_Datenbank_1.xlsx'
# Spaltennamen für die Eingabedaten und Ausgabedaten
input_columns = ['Drehrichtung', 'Schnittgeschw.', 'Eingriffsbreite', 'Zahnvorschub', 'Versuch', 'Eingriffstiefe', 'Spannungskomp.']
output_columns = ['a', 'b', 'c', 'd', 'e', 'f']

# Lese die Daten aus der Excel-Datei
data = pd.read_excel(SDCF_FILE_PATH)

# Datenbereinigung
data = clean_data(data)

# Datenaufteilung
data_D16, data_D50 = split_data(data)

# Aufteilen der Daten in Eingabe (X) und Ausgabe (y)
X_16 = data_D16[input_columns]
y_16 = data_D16[output_columns]
X_50 = data_D50[input_columns]
y_50 = data_D50[output_columns]

X_train_16, X_test_16, y_train_16, y_test_16 = train_test_split(X_16, y_16, test_size=0.3, random_state=42)
X_train_50, X_test_50, y_train_50, y_test_50 = train_test_split(X_50, y_50, test_size=0.3, random_state=42)

x_train = X_train_16#.drop("Versuch", axis=1)
x_test = X_test_16#.drop("Versuch", axis=1)
y_train = y_train_16
y_test = y_test_16

x_train_scaled, x_test_scaled = scale_datasets(x_train, x_test)


#### Model
hidden_units1 = 32#160
hidden_units2 = 128#480
hidden_units3 = 64#256
learning_rate = 0.01
# Creating model using the Sequential in tensorflow
def build_model_using_sequential():
  model = Sequential([
    Dense(hidden_units1, kernel_initializer='normal', activation='relu'),
    Dropout(0.2),
    Dense(hidden_units2, kernel_initializer='normal', activation='relu'),
    Dropout(0.2),
    Dense(hidden_units3, kernel_initializer='normal', activation='relu'),
    Dense(y_train.shape[1], kernel_initializer='normal', activation='linear')
  ])
  return model
# build the model
model = build_model_using_sequential()

# loss function
msle = MeanSquaredLogarithmicError()
model.compile(
    loss=msle, 
    optimizer=Adam(learning_rate=learning_rate), 
    metrics=[msle]
)
# train the model
history = model.fit(
    x_train_scaled.values, 
    y_train.values, 
    epochs=120, 
    batch_size=64,
    validation_split=0.2,
    verbose=0
)


# Plot the history
plot_history(history, 'mean_squared_logarithmic_error')

