import pickle
import pandas as pd

# Load the model
with open('render-demo/best_model.pkl', 'rb') as file:
    model = pickle.load(file)

input_data = pd.DataFrame([[
    1,    # fbs
    1,    # sex
    1,    # restecg
    0,    # dataset
    2,    # cp
    3,    # thal
    1,    # exang
    1,    # slope
    160,  # trestbps
    586,  # chol
    150,  # thalch
    78,   # age
    1.7,  # oldpeak
    3.0    # ca
]], columns=[
    'fbs', 'sex', 'restecg', 'dataset', 'cp', 'thal', 'exang', 'slope', 
    'trestbps', 'chol', 'thalch', 'age', 'oldpeak', 'ca'
])
# Predict
prediction = model.predict(input_data)
print(f"Prediction: {prediction}")