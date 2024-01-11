import pandas as pd
import numpy as np
import os
# import inspect
from scipy import optimize
from scipy.optimize import minimize_scalar
from sklearn.metrics import r2_score
from tqdm import tqdm

# DATA_BASE_FILE_PATH = 'Y:\\B5-Daten\\01-Forschung\\Poly-ProFiLEd\\02_Forschungsergebnisse\\Messung\\Reduzierte_Datenbank'
# ACCUM_EXCEL_FILE_PATH = DATA_BASE_FILE_PATH + '\\Kumulierte_res_Dateien.xlsx'
# RES_FILE_PATH = DATA_BASE_FILE_PATH + '\\Einzelne_res_Dateien'

DATA_BASE_FILE_PATH = '.\\Reduzierte_Datenbank'
ACCUM_EXCEL_FILE_PATH = DATA_BASE_FILE_PATH + '\\Kumulierte_res_Dateien.xlsx'
RES_FILE_PATH = DATA_BASE_FILE_PATH + '\\Einzelne_res_Dateien'

def gen_accumulated_database(res_data_file_path):
    # Liste für das akkumulierte DataFrame
    accumulated_data = []

    # Spaltennamen für das DataFrame
    column_names = ["Index", "Bohrungstiefe z", "σx", "σy", "τxy",
                    "filename", "Durchmesser", "Drehrichtung", "Schnittgeschw.",
                    "Eingriffsbreite", "Zahnvorschub", "Versuch", "Eingriffstiefe"]

    # Loop durch alle Excel-Dateien im angegebenen Verzeichnis
    for filename in os.listdir(res_data_file_path):
        if filename.endswith(".xlsx"):
            excel_path = os.path.join(res_data_file_path, filename)

            # Lese die gewünschten Daten aus dem Excel-Blatt ein
            df = pd.read_excel(excel_path, sheet_name="ASTM DMS Typ A", usecols="A:D", skiprows=361, nrows=20)

            # Erstelle einen DataFrame für die aktuellen Daten
            
            current_data = pd.DataFrame(columns=column_names[1:5], data=df.values)
            current_data["Index"] = np.arange(1, 21)
            current_data["filename"] = filename[:-5]
            current_data["Durchmesser"] = np.where(current_data["filename"].str.contains("_25_"), "50",
                                                    np.where(current_data["filename"].str.contains("_40_"), "50",
                                                        np.where(current_data["filename"].str.contains("_8_"), "16", "NaN")))
            current_data["Drehrichtung"] = np.where(current_data["filename"].str.contains("KSS_0"), "0", "1")       
            current_data["Schnittgeschw."] = np.where(current_data["filename"].str.contains("500"), "500",
                                                    np.where(current_data["filename"].str.contains("700"), "700",
                                                                np.where(current_data["filename"].str.contains("800"), "800",
                                                                        np.where(current_data["filename"].str.contains("900"), "900",
                                                                                np.where(current_data["filename"].str.contains("1000"), "1000",
                                                                                        np.where(current_data["filename"].str.contains("2670"), "2670", "NaN"))))))
            current_data["Eingriffsbreite"] = np.where(current_data["filename"].str.contains("_25_"), "25",
                                                       np.where(current_data["filename"].str.contains("_40_"), "25",
                                                            np.where(current_data["filename"].str.contains("_8_"), "8", "NaN")))
            current_data["Zahnvorschub"] = np.where(current_data["filename"].str.contains("_0,1_"), "0,1",
                                                    np.where(current_data["filename"].str.contains("_0,16_"), "0,16",
                                                                np.where(current_data["filename"].str.contains("_0,2_"), "0,2", "NaN")))
            
            current_data["Versuch"] = np.where(current_data["filename"].str.contains("_a_"), "a",
                                            np.where(current_data["filename"].str.contains("_b_"), "b",
                                                        np.where(current_data["filename"].str.contains("_c_"), "c", "NaN")))
            current_data["Eingriffstiefe"] = np.where(current_data["filename"].str.contains("ap1"), "1",
                                                      np.where(current_data["filename"].str.contains("ap2"), "2", "NaN"))


            # Füge die aktuellen Daten dem akkumulierten DataFrame hinzu
            accumulated_data.append(current_data)

    # Erstelle das endgültige akkumulierte DataFrame
    accumulated_df = pd.concat(accumulated_data, ignore_index=True, sort=False)
    accumulated_df = accumulated_df.reindex(columns=column_names)  # Reihenfolge der Spalten anpassen

    return accumulated_df

def edit_measurement(measurement: pd.DataFrame) -> pd.DataFrame:
        
    # Iteriere über die Spalten "σx", "σy" und "τxy"
    for col_name in ["σx", "σy", "τxy"]:
        # Wähle die jeweilige Spalte aus dem DataFrame aus
        col = measurement[col_name]

        # Setze die letzen 4 Werte der Spalte auf None
        col[-4:] = None
        # Werte größer als 300 und kleiner als -300 auf None setzen
        col[(col > 300) | (col < -300)] = None

        # Berechne die Ableitung (Gradient) der Spalte abzüglich der 4 letzen Werte
        gradient = np.diff(col[:-4])

        # Iteriere über die umgedrehte Gradientenliste
        for i, i_value in enumerate(gradient[::-1]):
            # i bezieht sich auf die umgedrehte Gradientenliste -> invertiere den Index
            normal_idx = len(gradient) - i

            # Überprüfe, ob der Wert in der Spalte positiv ist
            if (col[normal_idx] >= 0):
                # Wenn die Ableitung positiv ist, setze den Wert in der Spalte auf None
                if i_value > 0:
                    col[normal_idx] = None
                else: 
                    break  # Breche die Schleife ab, wenn die Bedingung nicht mehr erfüllt ist
            else:
                # Wenn der Wert in der Spalte negativ ist, setze den Wert in der Spalte auf None
                if i_value < 0:
                    col[normal_idx] = None
                else:
                    break  # Breche die Schleife ab, wenn die Bedingung nicht mehr erfüllt ist

        # Aktualisiere die Spalte im DataFrame mit den geänderten Werten
        measurement[col_name] = col

    # Gebe das aktualisierte DataFrame zurück
    return measurement

# sinusförmige Abklingfunktion definieren
def sinus_decay(t, a, b, c, d, e):
    return a * np.exp(-b * t) * np.cos(c * t + d) + e

def sinus_decay_2(t, a, b, c, d, e, f):
    return a * np.exp(-b * t) * np.cos(c * t**f + d) + e

# Finden der lokalen Extremstellen
def extremum_derivative(t, a, b, c, d):
    # Ableitung der Sinus-Decay-Funktion
    derivative = -a * b * np.exp(-b * t) * np.sin(c * t + d)
    return derivative

# Funktion zur Suche nach der zweiten Nullstelle
def find_second_zero(params):
    
    # Suche nach der ersten Nullstelle mit NumPy und SciPy
    result1 = minimize_scalar(lambda t: abs(sinus_decay(t, *params)), bounds=(0, 0.8), method='bounded')
    if result1.success:
        zero1 = result1.x

        # Suche nach der zweiten Nullstelle in der Nähe der ersten Nullstelle
        result2 = minimize_scalar(lambda t: abs(sinus_decay(t, *params)), bounds=(zero1 + 0.01, 1), method='bounded')
        if result2.success:
            zero2 = result2.x
            return zero2
        else:
            return "no second 0"
    else:
        return "no first 0"

# Funktion zur Suche nach dem Extremum oder Konvergenzpunkt
def find_extremum_or_convergence(params):
    
    # Suche nach dem ersten lokalen Extremum mit NumPy und SciPy
    result = minimize_scalar(lambda t: -extremum_derivative(t, *params[:-1]), bounds=(0, 1))
    if result.success:
        # Wenn ein Extremum gefunden wird, geben Sie die Stelle und den Wert des Extremums zurück
        depth_extr = result.x
        stress_extr = sinus_decay(depth_extr, *params)
        return depth_extr, stress_extr
    else:
        # Wenn kein Extremum gefunden wird, suchen wir nach Konvergenzpunkten ohne Scikit-learn
        t_values = np.linspace(0, 10, 1000)  # Beispielwerte für t
        y_values = sinus_decay(t_values, *params)
        
        # Suchen nach Konvergenzpunkten (Punkte, an denen sich die Werte nicht signifikant ändern)
        tol = 1e-6  # Toleranz für Konvergenz
        converged_points = []
        for i in range(1, len(y_values)):
            if abs(y_values[i] - y_values[i-1]) < tol:
                converged_points.append((y_values[i], t_values[i]))
        
        if converged_points:
            # Konvergenzpunkt gefunden, geben Sie den ersten gefundenen Punkt zurück
            stress_extr, depth_extr = converged_points[0]
            return depth_extr, stress_extr
        else:
            return None, None

def calculate_r_squared(measurement: pd.DataFrame, col_name: str, params):
    # Berechne R^2 zwischen y und y_pred
    y = measurement[col_name]
    y= y.dropna()
    n = len(y)
    y_pred = sinus_decay(measurement["Bohrungstiefe z"], *params)
    y_pred = y_pred[:n]
    r_squared = r2_score(y, y_pred)
    measurement["R²"] = round (r_squared, 2)
    return r_squared

def find_characteristic_points(params):
    # Surface values bestimmen
    stress_surf = sinus_decay(0, *params) #funktioniert
    depth_surf = 0 #funktioniert
    stress_extr, depth_extr = find_extremum_or_convergence(params) #funktioniert nicht
    stress_null = 0 #funktioniert
    depth_null = find_second_zero(params) #funktioniert nicht
    
    return stress_surf, depth_surf, stress_extr, depth_extr, stress_null, depth_null

# Erstellen der Sinus-Decay-Funktions-Datenbank
def create_SDCF_database(dataframe: pd.DataFrame) -> pd.DataFrame:
    
    # Liste für das akkumulierte DataFrame
    accumulated_df = []
    # Spaltennamen für das DataFrame
    new_column_names = ["filename", "Durchmesser", "Drehrichtung", "Schnittgeschw.", "Eingriffsbreite", "Zahnvorschub", "Versuch", 
                    "Eingriffstiefe", "Spannungskomp.", "a", "b", "c", "d", "e", "R²", "σ_surf", "SDCF"] 

    groups = dataframe.groupby("filename")
    # Loop durch jede Messung im DataFrame
    for idx, measurement in tqdm(groups, "Fitting measurements"):
        
        # Berechne die Parameter der Abklingfunktion
        for col_name in ["σx", "σy", "τxy"]:
            current_SDCF_data = pd.DataFrame(columns=new_column_names)
            col = measurement[col_name]
            nn_mask = col.notna()
            if nn_mask.sum() >= 6 and measurement[col_name].all != 0 :
                try:
                    params, params_covariance = optimize.curve_fit(sinus_decay, measurement["Bohrungstiefe z"][nn_mask], col[nn_mask],
                                                                  p0=[-400, 0.7921, 0.7300, 0, 0.9595])
                    
                    # Berechne den R²-Wert
                    r_squared = calculate_r_squared(measurement, col_name, params)

                    # Erstelle Liste mit den Return values von find_characteristic_points
                    characteristic_points= find_characteristic_points(params)
                    
                    # Übertragen die Parameter in die neue Datenbank
                    current_SDCF_data["Spannungskomp."] = [col_name]
                    current_SDCF_data["a"] = params[0]
                    current_SDCF_data["b"] = params[1]
                    current_SDCF_data["c"] = params[2]
                    current_SDCF_data["d"] = params[3]
                    current_SDCF_data["e"] = params[4]
                    current_SDCF_data["filename"] = [idx]
                    current_SDCF_data["Durchmesser"] = measurement["Durchmesser"]
                    current_SDCF_data["Drehrichtung"] = measurement["Drehrichtung"]
                    current_SDCF_data["Schnittgeschw."] = measurement["Schnittgeschw."]
                    current_SDCF_data["Eingriffsbreite"] = measurement["Eingriffsbreite"]
                    current_SDCF_data["Zahnvorschub"] = measurement["Zahnvorschub"]
                    current_SDCF_data["Versuch"] = measurement["Versuch"]
                    current_SDCF_data["Eingriffstiefe"] = measurement["Eingriffstiefe"]
                    current_SDCF_data["Spannungskomp."] = col_name
                    current_SDCF_data["R²"] = r_squared
                    current_SDCF_data["σ_surf"] = characteristic_points[0]
                    # current_SDCF_data["z_surf"] = characteristic_points[1]
                    # current_SDCF_data["σ_extr"] = characteristic_points[2]
                    # current_SDCF_data["z_extr"] = characteristic_points[3]
                    # current_SDCF_data["σ_null"] = characteristic_points[4]
                    # current_SDCF_data["z_null"] = characteristic_points[5]
                    current_SDCF_data["SDCF"] = f"{params[0]} * e^(-{params[1]} * t) * cos({params[2]} * t + {params[3]}) + {params[4]}"

                    #measurement.loc[nn_mask, ["Spannungskomp.", "a", "b", "c", "d", "e"]] = params
                except RuntimeError:
                    pass
            accumulated_df.append(current_SDCF_data)

    # Erstelle das endgültige akkumulierte DataFrame
    accumulated_df = pd.concat(accumulated_df, ignore_index=True, sort=False)
    accumulated_df = accumulated_df.reindex(columns=new_column_names)  # Reihenfolge der Spalten anpassen
    return accumulated_df
    '''
    # Iteriere über die Spalten "σx", "σy" und "τxy"
    for col_name in ["σx", "σy", "τxy"]:
        # Wähle die jeweilige Spalte aus dem DataFrame aus
        col = SDCF_data[col_name]
    # Berechne die Parameter der Abklingfunktion
            # Überprüfe, ob die Spalte NaN-Werte enthält
        nn_mask = col.notna()
        if nn_mask.sum() >=6: # Überprüfe, ob mindest 6 Datenpunkte vorhanden sind
            try:
                    params, params_covariance = optimize.curve_fit(sinus_decay, SDCF_data["Bohrungstiefe z"][nn_mask], col[nn_mask],
                                                                p0=[-400, 0.7921, 0.7300, 0, 0.9595]) 
                    SDCF_data["fit"] = [*params, *([None]*15)]
            except RuntimeError:
                    SDCF_data["fit"] = ["NaN"] + [None] * 19
                
        else:
                SDCF_data["fit"] = ["NaN"] + [None] * 19
    '''
    '''
    #R^2 berechnen
    y_mean = y.mean()
    y_pred = function_to_fit(t, *popt)
    SS_tot = ((y - y_mean)**2).sum()
    SS_res = ((y - y_pred)**2).sum()
    R_squared = 1 - SS_res/SS_tot
    R_squared = abs(R_squared)
    print(function_name, R_squared)
    
    return dataframe
    '''

# Hauptfunktion
def main():
    
    # Generiere die kumulierte Datenbank
    kumulierte_df = gen_accumulated_database(RES_FILE_PATH)
    # Speichere die kumulierte Datenbank in eine Excel-Datei
    kumulierte_df.to_excel(ACCUM_EXCEL_FILE_PATH, sheet_name='Kumulierte_Datenbank', index=False)
    
    # Lese die Excel-Datei ein und wähle das Blatt 'Kumulierte_Datenbank' aus
    df = pd.read_excel(ACCUM_EXCEL_FILE_PATH, sheet_name='Kumulierte_Datenbank')
    

    # Überprüfe, ob die Anzahl der Zeilen durch 20 teilbar ist
    assert df.shape[0] % 20 == 0
    # Berechne die Anzahl der Messungen
    n_measurements = df.shape[0] // 20
    # Teile den DataFrame in einzelne Messungen auf
    measurements = []
    for m in range(n_measurements):
        start = m * 20
        end = start + 20
        measurements.append(df.iloc[start:end].copy(
            deep=True).reset_index(drop=True))
    
    # Liste zur Aufbewahrung der verarbeiteten Messungen
    edited_measurements: list[pd.DataFrame] = []
    # Verarbeite jede Messung und füge sie zur Liste hinzu
    for m in measurements:
        edited_measurements.append(edit_measurement(m))
    # Kombiniere die verarbeiteten Messungen zu einem neuen DataFrame
    combined_edited_df = pd.concat(edited_measurements)
    # Speichere den kombinierten DataFrame in eine neue Excel-Datei
    combined_edited_df.to_excel(DATA_BASE_FILE_PATH + "\\Gefilterte_Datenbank.xlsx", sheet_name='Gefilterte_DB', index=False)
    
    
        
    # Rufe die create_SDCF_database-Funktion auf
    combined_SDCF_df = create_SDCF_database(combined_edited_df)
    combined_SDCF_df.to_excel(DATA_BASE_FILE_PATH + "\\SDCF_Datenbank.xlsx", sheet_name='SDCF_DB', index=False)

if __name__ == "__main__":
    main()
