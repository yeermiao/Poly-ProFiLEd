import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import uniform
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import warnings
from sklearn.exceptions import ConvergenceWarning
import joblib

# region 1: Daten einlesen und verarbeiten
######################################################################################
# Lese die Daten aus der Excel-Datei
def read_and_split_dataset(data):
    # 根据第二列的数据划分成两个数据集 D16 和 D50
    d16_dataset = data[data.iloc[:, 1] == 'D16']
    d50_dataset = data[data.iloc[:, 1] == 'D50']

    return d16_dataset, d50_dataset

def train_test(input_columns, output_columns, data_filtered):
    X = data_filtered[input_columns]
    y = data_filtered[output_columns]
    X_train_func, X_test_func, y_train_func, y_test_func = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train_func, X_test_func, y_train_func, y_test_func
######################################################################################
# endregion

# region 2: Random Forest Model
######################################################################################
# Definiere die Pipeline
def random_forest_model(X_train, X_test, y_train, y_test):
    pipeline_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor())
    ])

    param_dist = {
        'rf__n_estimators': (50, 100, 150),
        'rf__max_depth': [None, 5, 10, 15],
        'rf__min_samples_split': randint(2, 4),
        'rf__min_samples_leaf': randint(1, 2),
        'rf__max_features': ['sqrt', 'log2', None]
    }

    random_search = RandomizedSearchCV(estimator=pipeline_rf, param_distributions=param_dist, n_iter=100,
                                       cv=5, scoring='neg_mean_squared_error', random_state=42)


    # param_dist = {
    #     'rf__n_estimators': (100, 150, 200),
    #     'rf__max_depth': [None, 10, 20, 30, 40, 50],
    #     'rf__min_samples_split': randint(2, 5),
    #     'rf__min_samples_leaf': randint(1, 3),
    #     'rf__max_features': ['sqrt', 'log2', None]
    # }
    #
    # random_search = RandomizedSearchCV(estimator=pipeline_rf, param_distributions=param_dist, n_iter=100,
    #                                    cv=5, scoring='neg_mean_squared_error', random_state=42)


    random_search.fit(X_train, y_train)
    best_rf_model = random_search.best_estimator_
    y_pred_rf = best_rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    return best_rf_model, mse_rf, r2_rf
######################################################################################
# endregion

# region 3: Support Vector Machine Model
######################################################################################
# Pipeline: Datenstandardisierung und SVM-Regression
def svm_model(X_train, X_test, y_train, y_test):
    pipeline_svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', MultiOutputRegressor(SVR()))
    ])

    param_dist_svm = {
        'svm__estimator__C': uniform(loc=0, scale=5),
        'svm__estimator__kernel': ['linear', 'rbf', 'poly'],
        'svm__estimator__degree': [2, 3, 4],
        'svm__estimator__gamma': ['scale', 'auto'] + [0.1, 0.01, 0.001],
    }

    random_search_svm = RandomizedSearchCV(estimator=pipeline_svm, param_distributions=param_dist_svm, n_iter=100,
                                           cv=5, scoring='neg_mean_squared_error', random_state=42)

    # param_dist_svm = {
    #     'svm__estimator__C': uniform(loc=0, scale=10),
    #     'svm__estimator__kernel': ['linear', 'rbf', 'poly'],
    #     'svm__estimator__degree': [2, 3, 4],
    #     'svm__estimator__gamma': ['scale', 'auto'] + [0.1, 0.01, 0.001, 0.0001],
    # }
    #
    # random_search_svm = RandomizedSearchCV(estimator=pipeline_svm, param_distributions=param_dist_svm, n_iter=100,
    #                                        cv=5, scoring='neg_mean_squared_error', random_state=42)




    random_search_svm.fit(X_train, y_train)
    best_svm_model = random_search_svm.best_estimator_
    y_pred_svm = best_svm_model.predict(X_test)
    mse_svm = mean_squared_error(y_test, y_pred_svm)
    r2_svm = r2_score(y_test, y_pred_svm)

    # Output feature importance
    # SVM does not have a direct feature_importances_ attribute, so we skip this part
    return best_svm_model, mse_svm, r2_svm

######################################################################################
# endregion

# region 4: Neural Network Model
#####################################################################################
# Definiere der Pipeline für die Datenverarbeitung und das Modell
def neural_network_model(X_train, X_test, y_train, y_test):
    pipeline_nn = Pipeline([
        ('scaler', StandardScaler()),
        ('nn', MLPRegressor(max_iter=3000))
    ])

    param_dist_nn = {
        'nn__hidden_layer_sizes': [(50,), (100, 50)],
        'nn__activation': ['relu', 'tanh', 'logistic'],
        'nn__alpha': uniform(loc=0, scale=0.5),
        'nn__learning_rate_init': [0.001, 0.01, 0.1, 0.0001],
        'nn__solver': ['adam', 'sgd']
    }

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    random_search_nn = RandomizedSearchCV(estimator=pipeline_nn, param_distributions=param_dist_nn,
                                           n_iter=100, cv=5, scoring='neg_mean_absolute_error')


    # param_dist_nn = {
    #     'nn__hidden_layer_sizes': [(50,), (100, 50), (100, 100), (50, 30, 20)],
    #     'nn__activation': ['relu', 'tanh', 'logistic'],
    #     'nn__alpha': uniform(loc=0, scale=0.5),
    #     'nn__learning_rate_init': [0.001, 0.01, 0.1, 0.0001],
    #     'nn__solver': ['adam', 'sgd']
    # }
    #
    # warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # random_search_nn = RandomizedSearchCV(estimator=pipeline_nn, param_distributions=param_dist_nn,
    #                                        n_iter=100, cv=5, scoring='neg_mean_absolute_error')


    random_search_nn.fit(X_train, y_train)
    best_nn_model = random_search_nn.best_estimator_
    y_pred_nn = best_nn_model.predict(X_test)
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    r2_nn = r2_score(y_test, y_pred_nn)

    # Output feature importance
    # Neural Network does not have a direct feature_importances_ attribute, so we skip this part

    return best_nn_model, mse_nn, r2_nn
#####################################################################################
# endregion

# region 5: Bayesian Neural Network Model
######################################################################################

######################################################################################
# endregion

# region 6: permutation_importance
######################################################################################
def calculate_and_print_permutation_importance(model, X_test, y_test, input_columns):
    result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
    importance = result.importances_mean
    sorted_features = sorted(zip(input_columns, importance), key=lambda x: x[1], reverse=True)
    # Print Feature-Importance
    for feature, imp in sorted_features:
        print(f'{feature}: {imp}')


def print_feature_importance(model, X_train):
    feature_importance = model.named_steps['rf'].feature_importances_
    sorted_features = sorted(zip(X_train.columns, feature_importance), key=lambda x: x[1], reverse=True)
    print('Random Forest Feature Importance:')
    for feature, importance in sorted_features:
        print(f'{feature}: {importance}')
######################################################################################
# endregion

# region 7: Execution code
######################################################################################
if __name__ == "__main__":
    Berechnete_Eigenspannungen_File_Path = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Zusammenfassung_der_berechneten_Restspannungen.xlsx'
    input_columns = ['Fraesrichtung', 'Schnittgeschwindigkeit', 'Eingriffsbreite', 'Zahnvorschub', 'Eingriffstiefe']
    output_columns = ['σxx_mean', 'σyy_mean', 'Txy_mean']
    filtering_columns = ['σxx_SD', 'σyy_SD', 'Txy_SD']
    threshold = 3
    data = pd.read_excel(Berechnete_Eigenspannungen_File_Path, na_values='NaN')
    data = data.dropna()
    data_filtered = data[(data[filtering_columns] <= threshold).all(axis=1)]
    data_16, data_50 = read_and_split_dataset(data_filtered)

    # Berechnete_Eigenspannungen_File_Path = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Zusammenfassung_σ_Verzug.xlsx'
    # input_columns = ['Fraesrichtung', 'Schnittgeschwindigkeit', 'Eingriffsbreite', 'Zahnvorschub', 'Eingriffstiefe']
    # output_columns = ['σxx', 'σyy', 'Txy']
    # data = pd.read_excel(Berechnete_Eigenspannungen_File_Path, na_values='NaN')
    # data = data.dropna()
    # data_16, data_50 = read_and_split_dataset(data)



    # D16 数据训练与评估
    X_train_16, X_test_16, y_train_16, y_test_16 = train_test(input_columns, output_columns, data_16)

    best_rf_model_16, mse_rf_16, r2_rf_16 = random_forest_model(X_train_16, X_test_16, y_train_16, y_test_16)
    print("Random Forest - D16 Dataset")
    print(f'MSE: {mse_rf_16}, R2: {r2_rf_16}')
    calculate_and_print_permutation_importance(best_rf_model_16, X_test_16, y_test_16, input_columns)
    joblib.dump(best_rf_model_16, '../best_rf_model_16.joblib')
    print("-" * 50)
    print_feature_importance(best_rf_model_16, X_train_16)
    print("=" * 100)
    best_svm_model_16, mse_svm_16, r2_svm_16 = svm_model(X_train_16, X_test_16, y_train_16, y_test_16)
    print("SVM - D16 Dataset")
    print(f'MSE: {mse_svm_16}, R2: {r2_svm_16}')
    calculate_and_print_permutation_importance(best_svm_model_16, X_test_16, y_test_16, input_columns)
    joblib.dump(best_svm_model_16, '../best_svm_model_16.joblib')
    print("=" * 100)
    best_nn_model_16, mse_nn_16, r2_nn_16 = neural_network_model(X_train_16, X_test_16, y_train_16, y_test_16)
    print("Neural Network - D16 Dataset")
    print(f'MSE: {mse_nn_16}, R2: {r2_nn_16}')
    calculate_and_print_permutation_importance(best_nn_model_16, X_test_16, y_test_16, input_columns)
    joblib.dump(best_nn_model_16, '../best_nn_model_16.joblib')
    print("#" * 100)
    print("#" * 100)

    # D50 数据训练与评估
    X_train_50, X_test_50, y_train_50, y_test_50 = train_test(input_columns, output_columns, data_50)

    best_rf_model_50, mse_rf_50, r2_rf_50 = random_forest_model(X_train_50, X_test_50, y_train_50, y_test_50)
    print("Random Forest - D50 Dataset")
    print(f'MSE: {mse_rf_50}, R2: {r2_rf_50}')
    calculate_and_print_permutation_importance(best_rf_model_50, X_test_50, y_test_50, input_columns)

    # 保存 Random Forest - D50 模型
    joblib.dump(best_rf_model_50, '../best_rf_model_50.joblib')

    print("-" * 50)
    print_feature_importance(best_rf_model_50, X_train_50)
    print("=" * 100)

    best_svm_model_50, mse_svm_50, r2_svm_50 = svm_model(X_train_50, X_test_50, y_train_50, y_test_50)
    print("SVM - D50 Dataset")
    print(f'MSE: {mse_svm_50}, R2: {r2_svm_50}')

    # 保存 SVM - D50 模型
    joblib.dump(best_svm_model_50, '../best_svm_model_50.joblib')

    calculate_and_print_permutation_importance(best_svm_model_50, X_test_50, y_test_50, input_columns)
    print("=" * 100)
    best_nn_model_50, mse_nn_50, r2_nn_50 = neural_network_model(X_train_50, X_test_50, y_train_50, y_test_50)
    print("Neural Network - D50 Dataset")
    print(f'MSE: {mse_nn_50}, R2: {r2_nn_50}')

    # 保存 Neural Network - D50 模型
    joblib.dump(best_nn_model_50, '../best_nn_model_50.joblib')

    calculate_and_print_permutation_importance(best_nn_model_50, X_test_50, y_test_50, input_columns)
    print("#" * 100)
    print("#" * 100)
######################################################################################
# endregion



