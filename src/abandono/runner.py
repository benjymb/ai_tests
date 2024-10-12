from abandono.agent import ChurnPredictionModel
from helpers.file_helper import get_filepath

# Uso del modelo
if __name__ == '__main__':
    # 1. Crear una instancia de la clase con el dataset
    churn_model = ChurnPredictionModel(data_path=get_filepath('interacciones.csv'))
    
    # 2. Realizar ingeniería de características
    churn_model.feature_engineering()
    
    # 3. Preparar los datos
    churn_model.prepare_data()
    
    # 4. Entrenar el modelo
    churn_model.train_model()
    
    # 5. Evaluar el modelo
    churn_model.evaluate_model()
    
    # 6. Mostrar la importancia de las características
    churn_model.feature_importance()