import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import uniform
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, HalvingGridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import warnings
from sklearn.exceptions import ConvergenceWarning
import joblib
from sklearn.model_selection import LeaveOneOut
import time

# region 1: Daten einlesen und verarbeiten
######################################################################################
# Lese die Daten aus der Excel-Datei
def read_and_split_dataset(data):
    # 根据第二列的数据划分成两个数据集 D16 和 D50
    d16_dataset = data[data.iloc[:, 1] == 'D16']
    d50_dataset = data[data.iloc[:, 1] == 'D50']

    return d16_dataset, d50_dataset


def split_gegen_gleich_dataset(data):
    # 根据第三列的数据划分成两个数据集 D16 和 D50
    gegenlauf_0_dataset = data[data.iloc[:, 2] == 0]
    gleichlauf_1_dataset = data[data.iloc[:, 2] == 1]

    return gegenlauf_0_dataset, gleichlauf_1_dataset


def split_high_low_Vc_dataset(data):
    High_Vc_dataset = data[data.iloc[:, 3] == 2670]
    Low_Vc_dataset = data[data.iloc[:, 3] != 2670]

    return High_Vc_dataset, Low_Vc_dataset


def split_Ap_dataset(data):
    Ap1_dataset = data[data.iloc[:, 6] == 1]
    Ap2_dataset = data[data.iloc[:, 6] == 2]

    return Ap1_dataset, Ap2_dataset



def train_test(input_columns, output_columns, data_fil):
    X = data_fil[input_columns]
    y = data_fil[output_columns].values.ravel()
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

    # region 针对单个输出的Grid Search超参数组合
    param_grid = {
        'rf__n_estimators': [5, 10, 50],
        'rf__max_depth': [None, 5, 10],
        'rf__min_samples_split': [2, 3, 4],
        'rf__min_samples_leaf': [1, 2],
        'rf__max_features': ['sqrt', 'log2', None]
    }

    # halving_grid_search = HalvingGridSearchCV(estimator=pipeline_rf, param_grid=param_grid,
    #                                           scoring='neg_mean_squared_error', random_state=42)

    grid_search = GridSearchCV(
        estimator=pipeline_rf,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=LeaveOneOut(),  # 设置为一个小一点的值
        n_jobs=-1
    )
    # endregion

    # region MIMO常用超参数组合
    # param_dist = {
    #     'rf__n_estimators': (50, 100, 150),
    #     'rf__max_depth': [None, 5, 10, 15],
    #     'rf__min_samples_split': randint(2, 4),
    #     'rf__min_samples_leaf': randint(1, 2),
    #     'rf__max_features': ['sqrt', 'log2', None]
    # }
    #
    # random_search = RandomizedSearchCV(estimator=pipeline_rf, param_distributions=param_dist, n_iter=100,
    #                                    cv=5, scoring='neg_mean_squared_error', random_state=42)
    # endregion

    # region MIMO 另一种超参数组合
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
    # endregion

    grid_search.fit(X_train, y_train)
    best_rf_model = grid_search.best_estimator_
    y_pred_rf = best_rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    return best_rf_model, mse_rf, r2_rf
######################################################################################
# endregion

# region 3: Support Vector Machine Model
######################################################################################
# Pipeline: Datenstandardisierung und SVM-Regression
def support_vector_machine_model(X_train, X_test, y_train, y_test):

    # region SVR针对单个输出的模型Grid search超参数组合
    pipeline_svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVR())  # 不再使用 MultiOutputRegressor
    ])

    param_grid_svm = {
        'svm__C': [0.1, 1, 5],
        'svm__kernel': ['linear', 'rbf', 'poly'],
        'svm__degree': [2, 3, 4],
        'svm__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    }

    # halving_grid_search_svm = HalvingGridSearchCV(estimator=pipeline_svm, param_grid=param_grid_svm,
    #                                             scoring='neg_mean_squared_error', random_state=42)
    grid_search_svm = GridSearchCV(
        estimator=pipeline_svm,
        param_grid=param_grid_svm,
        scoring='neg_mean_squared_error',
        cv=LeaveOneOut(),  # 设置为适当的折数
        n_jobs=-1
    )
    # endregion

    # region MIMO常用超参数组合
    # pipeline_svm = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('svm', MultiOutputRegressor(SVR()))
    # ])
    #
    # param_dist_svm = {
    #     'svm__estimator__C': uniform(loc=0, scale=5),
    #     'svm__estimator__kernel': ['linear', 'rbf', 'poly'],
    #     'svm__estimator__degree': [2, 3, 4],
    #     'svm__estimator__gamma': ['scale', 'auto'] + [0.1, 0.01, 0.001],
    # }
    #
    # random_search_svm = RandomizedSearchCV(estimator=pipeline_svm, param_distributions=param_dist_svm, n_iter=100,
    #                                        cv=5, scoring='neg_mean_squared_error', random_state=42)
    # endregion

    # region MIMO另一种超参数组合
    # param_dist_svm = {
    #     'svm__estimator__C': uniform(loc=0, scale=10),
    #     'svm__estimator__kernel': ['linear', 'rbf', 'poly'],
    #     'svm__estimator__degree': [2, 3, 4],
    #     'svm__estimator__gamma': ['scale', 'auto'] + [0.1, 0.01, 0.001, 0.0001],
    # }
    #
    # random_search_svm = RandomizedSearchCV(estimator=pipeline_svm, param_distributions=param_dist_svm, n_iter=100,
    #                                        cv=5, scoring='neg_mean_squared_error', random_state=42)
    # endregion

    grid_search_svm.fit(X_train, y_train)
    best_svm_model = grid_search_svm.best_estimator_
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

    param_grid_nn = {
        'nn__hidden_layer_sizes': [(5,), (10,), (20,)],
        'nn__activation': ['relu', 'tanh'],
        'nn__alpha': [0, 0.1, 0.3],
        'nn__learning_rate_init': [0.001, 0.01, 0.1],
        'nn__solver': ['adam', 'sgd'],
    }
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # halving_grid_search_nn = HalvingGridSearchCV(estimator=pipeline_nn, param_grid=param_grid_nn,
    #                                              scoring='neg_mean_absolute_error', random_state=42)
    grid_search_nn = GridSearchCV(
        estimator=pipeline_nn,
        param_grid=param_grid_nn,
        scoring='neg_mean_absolute_error',
        cv=LeaveOneOut(),
        n_jobs=-1
    )


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


    grid_search_nn.fit(X_train, y_train)
    best_nn_model = grid_search_nn.best_estimator_
    y_pred_nn = best_nn_model.predict(X_test)
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    r2_nn = r2_score(y_test, y_pred_nn)

    # Output feature importance
    # Neural Network does not have a direct feature_importances_ attribute, so we skip this part

    return best_nn_model, mse_nn, r2_nn
#####################################################################################
# endregion

# region 5: permutation_importance
######################################################################################
def calculate_permutation_importance(model, X_test, y_test, input_columns):
    result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
    importance = result.importances_mean
    sorted_features = sorted(zip(input_columns, importance), key=lambda x: x[1], reverse=True)

    # Create a dictionary to store the results
    importance_dict = {feature: imp for feature, imp in sorted_features}

    return importance_dict


# def print_feature_importance(model, X_train):
#     feature_importance = model.named_steps['rf'].feature_importances_
#     sorted_features = sorted(zip(X_train.columns, feature_importance), key=lambda x: x[1], reverse=True)
#     print('Random Forest Feature Importance:')
#     for feature, importance in sorted_features:
#         print(f'{feature}: {importance}')
######################################################################################
# endregion

# region 6: Execution code
######################################################################################
if __name__ == "__main__":
    start_time = time.time()
    Berechnete_Eigenspannungen_File_Path = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Zusammenfassung_der_berechneten_Restspannungen.xlsx'
    # input_columns_all = ['Fraesrichtung', 'Schnittgeschwindigkeit', 'Eingriffsbreite', 'Zahnvorschub', 'Eingriffstiefe']
    # input_columns_wihout_Fraesrichtung = ['Schnittgeschwindigkeit', 'Eingriffsbreite', 'Zahnvorschub', 'Eingriffstiefe']
    input_columns_wihout_Richtung_und_Ap = ['Schnittgeschwindigkeit', 'Eingriffsbreite', 'Zahnvorschub']
    # output_columns_spannungen = ['σxx_mean', 'σyy_mean', 'Txy_mean']
    output_columns_sigemaxx = ['σxx_mean']
    output_columns_sigemayy = ['σyy_mean']
    output_columns_schubxy = ['Txy_mean']
    # filtering_columns_SD = ['σxx_SD', 'σyy_SD', 'Txy_SD']
    # threshold = 3
    data = pd.read_excel(Berechnete_Eigenspannungen_File_Path, na_values='NaN')
    data = data.dropna()
    # data_filtered = data[(data[filtering_columns_SD] <= threshold).all(axis=1)]
    # D16 und D50
    D16, D50 = read_and_split_dataset(data)
    # D16_gegenlauf und D16_gleichlauf
    D16_gegenlauf_0, D16_gleichlauf_1 = split_gegen_gleich_dataset(D16)

    D16_gegenlauf_0_Ap1, D16_gegenlauf_0_Ap2 = split_Ap_dataset(D16_gegenlauf_0)
    D16_gleichlauf_1_Ap1, D16_gleichlauf_1_Ap2 = split_Ap_dataset(D16_gleichlauf_1)
    # D50_2670 und D50_500,700,900
    # Zahnvorschub hier nur mit 0.16
    D50_high_Vc, D50_low_Vc = split_high_low_Vc_dataset(D50)
    # D50_2670_gegenlauf und D50_2670_gleichlauf
    # data_50_high_Vc_gegenlauf_0, data_50_high_Vc_gleichlauf_1 = split_gegen_gleich_dataset(data_50_high_Vc)
    D50_low_Vc_gegenlauf_0, D50_low_Vc_gleichlauf_1 = split_gegen_gleich_dataset(D50_low_Vc)
    D50_low_Vc_gegenlauf_0_Ap1, D50_low_Vc_gegenlauf_0_Ap2 = split_Ap_dataset(D50_low_Vc_gegenlauf_0)
    D50_low_Vc_gleichlauf_1_Ap1, D50_low_Vc_gleichlauf_1_Ap2 = split_Ap_dataset(D50_low_Vc_gleichlauf_1)

    # 输出的DataFrame，用于存储结果
    results_df = pd.DataFrame(columns=['Output', 'Best Model', 'Best Model R2', 'Feature', 'Importance'])

    # 循环处理四个数据库
    # for database_name, database in [('data_16_gegenlauf_0', D16_gegenlauf_0),
    #                                 ('data_16_gleichlauf_1', D16_gleichlauf_1),
    #                                 ('data_50_low_Vc_gegenlauf_0', D50_low_Vc_gegenlauf_0),
    #                                 ('data_50_low_Vc_gleichlauf_1', D50_low_Vc_gleichlauf_1)]:
    for database_name, database in [('D16_gegenlauf_0_Ap1', D16_gegenlauf_0_Ap1),
                                    ('D16_gegenlauf_0_Ap2', D16_gegenlauf_0_Ap2),
                                    ('D16_gleichlauf_1_Ap1', D16_gleichlauf_1_Ap1),
                                    ('D16_gleichlauf_1_Ap2', D16_gleichlauf_1_Ap2),
                                    ('D50_low_Vc_gegenlauf_0_Ap1', D50_low_Vc_gegenlauf_0_Ap1),
                                    ('D50_low_Vc_gegenlauf_0_Ap2', D50_low_Vc_gegenlauf_0_Ap2),
                                    ('D50_low_Vc_gleichlauf_1_Ap1', D50_low_Vc_gleichlauf_1_Ap1),
                                    ('D50_low_Vc_gleichlauf_1_Ap2', D50_low_Vc_gleichlauf_1_Ap2)]:

        # 循环处理三个输出特征
        for output_column in ['σxx_mean', 'σyy_mean', 'Txy_mean']:
            # 选择输入和输出
            X = database[input_columns_wihout_Richtung_und_Ap]
            y = database[output_column]

            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(),
                                                                test_size=0.2, random_state=42)

            rf_model, mse_rf, r2_rf = random_forest_model(X_train, X_test, y_train, y_test)
            importance_rf = calculate_permutation_importance(rf_model, X_test, y_test,
                                                             input_columns_wihout_Richtung_und_Ap)
            svm_model, mse_svm, r2_svm = support_vector_machine_model(X_train, X_test, y_train, y_test)
            importance_svm = calculate_permutation_importance(svm_model, X_test, y_test,
                                                              input_columns_wihout_Richtung_und_Ap)
            nn_model, mse_nn, r2_nn = neural_network_model(X_train, X_test, y_train, y_test)
            importance_nn = calculate_permutation_importance(nn_model, X_test, y_test,
                                                             input_columns_wihout_Richtung_und_Ap)

            best_model = max([('Random Forest', r2_rf, importance_rf, rf_model),
                              ('SVM', r2_svm, importance_svm, svm_model),
                              ('Neural Network', r2_nn, importance_nn, nn_model)],
                             key=lambda x: x[1])

            desktop_path = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten'  # Replace with your desktop path
            best_models_folder = os.path.join(desktop_path, "best_models")
            best_model_name = f"{database_name}_{output_column}_{best_model[0]}"
            best_model_path = os.path.join(best_models_folder, f"{best_model_name}_model.joblib")
            if not os.path.exists(best_models_folder):
                os.makedirs(best_models_folder)
            joblib.dump(best_model[3], best_model_path)

            for feature, imp in best_model[2].items():
                results_df = pd.concat([results_df, pd.DataFrame([{'Output': output_column,
                                                                   'Best Model': best_model[0],
                                                                   'Best Model R2': best_model[1],
                                                                   'Feature': feature,
                                                                   'Importance': imp}])])



    output_path = r'F:\Users\yeerm\Desktop\Masterarbeit\Messdaten\Signifikanzanalyse_split_Ap_Richtung.xlsx'
    # 将结果保存到Excel文件
    results_df.to_excel(output_path, index=False, engine='openpyxl')
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")






    # # D16 数据训练与评估
    # X_train_16, X_test_16, y_train_16, y_test_16 = train_test(input_columns_all,
    #                                                           output_columns_sigemaxx, data_50_high_Vc)
    #
    # best_rf_model_16, mse_rf_16, r2_rf_16 = random_forest_model(X_train_16, X_test_16, y_train_16, y_test_16)
    # print("Random Forest - D16 Dataset")
    # print(f'MSE: {mse_rf_16}, R2: {r2_rf_16}')
    # calculate_and_print_permutation_importance(best_rf_model_16, X_test_16, y_test_16, input_columns_all)
    # joblib.dump(best_rf_model_16, '../best_rf_model_16.joblib')
    # # print("-" * 50)
    # # print_feature_importance(best_rf_model_16, X_train_16)
    # print("=" * 100)
    # best_svm_model_16, mse_svm_16, r2_svm_16 = svm_model(X_train_16, X_test_16, y_train_16, y_test_16)
    # print("SVM - D16 Dataset")
    # print(f'MSE: {mse_svm_16}, R2: {r2_svm_16}')
    # calculate_and_print_permutation_importance(best_svm_model_16, X_test_16, y_test_16, input_columns_all)
    # joblib.dump(best_svm_model_16, '../best_svm_model_16.joblib')
    # print("=" * 100)
    # best_nn_model_16, mse_nn_16, r2_nn_16 = neural_network_model(X_train_16, X_test_16, y_train_16, y_test_16)
    # print("Neural Network - D16 Dataset")
    # print(f'MSE: {mse_nn_16}, R2: {r2_nn_16}')
    # calculate_and_print_permutation_importance(best_nn_model_16, X_test_16, y_test_16, input_columns_all)
    # joblib.dump(best_nn_model_16, '../best_nn_model_16.joblib')
    # print("#" * 100)
    # print("#" * 100)

    # D50 数据训练与评估
    # X_train_50, X_test_50, y_train_50, y_test_50 = train_test(input_columns_wihout_Fraesrichtung,
    #                                                           output_columns_sigemaxx, data_50)
    #
    # best_rf_model_50, mse_rf_50, r2_rf_50 = random_forest_model(X_train_50, X_test_50, y_train_50, y_test_50)
    # print("Random Forest - D50 Dataset")
    # print(f'MSE: {mse_rf_50}, R2: {r2_rf_50}')
    # calculate_and_print_permutation_importance(best_rf_model_50, X_test_50, y_test_50, input_columns_all)
    #
    # # 保存 Random Forest - D50 模型
    # joblib.dump(best_rf_model_50, '../best_rf_model_50.joblib')
    #
    # print("-" * 50)
    # print_feature_importance(best_rf_model_50, X_train_50)
    # print("=" * 100)
    #
    # best_svm_model_50, mse_svm_50, r2_svm_50 = svm_model(X_train_50, X_test_50, y_train_50, y_test_50)
    # print("SVM - D50 Dataset")
    # print(f'MSE: {mse_svm_50}, R2: {r2_svm_50}')
    #
    # # 保存 SVM - D50 模型
    # joblib.dump(best_svm_model_50, '../best_svm_model_50.joblib')
    #
    # calculate_and_print_permutation_importance(best_svm_model_50, X_test_50, y_test_50, input_columns_all)
    # print("=" * 100)
    # best_nn_model_50, mse_nn_50, r2_nn_50 = neural_network_model(X_train_50, X_test_50, y_train_50, y_test_50)
    # print("Neural Network - D50 Dataset")
    # print(f'MSE: {mse_nn_50}, R2: {r2_nn_50}')
    #
    # # 保存 Neural Network - D50 模型
    # joblib.dump(best_nn_model_50, '../best_nn_model_50.joblib')
    #
    # calculate_and_print_permutation_importance(best_nn_model_50, X_test_50, y_test_50, input_columns_all)
    # print("#" * 100)
    # print("#" * 100)

######################################################################################
# endregion
