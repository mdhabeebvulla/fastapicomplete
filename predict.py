# predict.py
import pickle
import pandas as pd

from config.core.py import config

def predict(new_data):
    model_path = config.get('model_save_path')
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    predictions = model.predict(new_data)
    return predictions

# Example usage
if __name__ == "__main__":
    new_data = pd.DataFrame({
        'sepal_length': [5.1, 4.9],
        'sepal_width': [3.5, 3.0],
        'petal_length': [1.4, 1.4],
        'petal_width': [0.2, 0.2]
    })
    print(predict(new_data))
