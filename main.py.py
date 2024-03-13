import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.api import qqplot

def load_data(file_path):
    """Cargar datos desde un archivo CSV."""
    return pd.read_csv(file_path)

def print_data_info(data):
    """Imprimir información básica sobre los datos."""
    print("Información de los datos:")
    print(data.head(4))
    print("\nForma de los datos:")
    print(data.shape)
    print("\nColumnas de los datos:")
    print(data.columns)
    print("\nInformación adicional:")
    print(data.info())
    print()

def fit_linear_model(data, formula):
    """Ajustar un modelo lineal a los datos."""
    return ols(formula, data=data).fit()

def make_predictions(model, explanatory_data):
    """Hacer predicciones utilizando el modelo."""
    return model.predict(explanatory_data)

def visualize_data_and_model(data, prediction_data=None):
    """Visualizar los datos y, opcionalmente, las predicciones."""
    sns.set_style("whitegrid")
    plt.title("Gráfico de regresion n_convenience y price_twd_msq")
    sns.regplot(x="n_convenience", y="price_twd_msq", data=data, ci=None)
    if prediction_data is not None:
        sns.scatterplot(x="n_convenience", y="price_twd_msq", data=prediction_data, color="black", marker="s")
    plt.savefig("model_ncon_pricetwd.png")
    plt.show()

def analyze_residuals(model):
    """Realizar análisis de residuos."""
    sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True)
    plt.xlabel("Valores ajustados")
    plt.ylabel("Residuos")
    plt.title("Análsis de residuos")
    plt.savefig("residuals_plot1.png")
    plt.show()
    
    qqplot(data=model.resid, fit=True, line="45")
    plt.show()
    model_norm_residuals = model.get_influence().resid_studentized_internal
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    sns.regplot(x=model.fittedvalues, y=model_norm_residuals_abs_sqrt, ci=None, lowess=True)
    plt.title("Valores ajustados y transformación de los residuos")
    plt.xlabel("Valores ajustados")
    plt.ylabel("Raíz cuadrada del valor absoluto de los residuos estandarizados")
    plt.savefig("residuals_plot2.png")
    plt.show()

def main():
    # Cargar datos
    data = load_data("taiwan_real_estate2.csv")
    
    # Imprimir información de los datos
    print_data_info(data)
    
    # Ajustar modelo lineal
    model = fit_linear_model(data, "price_twd_msq ~ n_convenience")
    
    # Hacer predicciones
    explanatory_data = pd.DataFrame({'n_convenience': np.arange(0, 11)})
    predictions = make_predictions(model, explanatory_data)
    prediction_data = explanatory_data.assign(price_twd_msq=predictions)
    
    # Visualizar datos y modelo
    visualize_data_and_model(data, prediction_data)
    
    # Realizar análisis de residuos
    analyze_residuals(model)

if __name__ == "__main__":
    main()
