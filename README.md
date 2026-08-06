# Customer-Churn-Prediction

![License](https://img.shields.io/badge/License-GNU%20GPL%20v3-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange?logo=streamlit)

## Description

**Customer-Churn-Prediction** is a machine learning project designed to predict whether a customer is likely to churn based on various features such as credit score, geography, gender, age, balance, and more. The project uses a neural network model built with TensorFlow/Keras and includes a Streamlit web application for interactive predictions.

Key components of the project include:
- A trained deep learning model (`model.h5`) for churn classification.
- Preprocessing pipelines using Label Encoding, One-Hot Encoding, and Standard Scaling.
- A Streamlit app (`app.py`) that allows users to input customer data and receive real-time churn predictions.
- Jupyter Notebooks (`experiments.ipynb`, `prediction.ipynb`) documenting the model development and evaluation process.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/duttabikram/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure all model and encoder files are present in the project directory:
   - `model.h5`
   - `label_encoder_gender.pkl`
   - `onehot_encoder_geo.pkl`
   - `scaler.pkl`

4. (Optional) Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

### Running the Streamlit App

After installation, start the app using:

```bash
streamlit run app.py
```

This will open a web interface where you can:
- Select **Geography**, **Gender**, and other demographic/behavioral inputs.
- Adjust sliders for **Age**, **Tenure**, and **Number of Products**.
- Enter numerical values for **Balance**, **Credit Score**, and **Estimated Salary**.
- Get a real-time prediction on whether the customer is likely to churn.

### Predicting Churn via Script

You can also load the model and make predictions programmatically:

```python
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# Load model and preprocessing tools
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    le_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    ohe_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Example input (replace with actual data)
input_df = pd.DataFrame({
    'CreditScore': [650],
    'Gender': [le_gender.transform(['Male'])[0]],
    'Age': [45],
    'Tenure': [5],
    'Balance': [50000],
    'NumOfProducts': [2],
    'HasCrCard': [1],
    'IsActiveMember': [1],
    'EstimatedSalary': [75000]
})

# Apply one-hot encoding for Geography
geo_encoded = ohe_geo.transform(input_df[['Geography']])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

# Combine and scale
final_input = pd.concat([input_df.drop('Geography', axis=1), geo_encoded_df], axis=1)
scaled_input = scaler.transform(final_input)

# Predict
prediction = model.predict(scaled_input)
print("Churn Probability:", prediction[0][0])
```

## Tech Stack

| Technology       | Purpose                          |
|------------------|----------------------------------|
| Python           | Core programming language        |
| TensorFlow       | Deep learning framework          |
| Keras            | Neural network API               |
| Scikit-learn     | Data preprocessing               |
| Streamlit        | Web application framework        |
| Pandas           | Data manipulation                |
| NumPy            | Numerical computing              |
| Matplotlib       | Visualization                    |
| TensorBoard      | Model visualization              |
| Pickle           | Saving/loading models & encoders |

## Features

- **Neural Network Model**: Trained deep learning model for accurate churn prediction.
- **Preprocessing Pipeline**: Handles categorical encoding and feature scaling.
- **Interactive Web App**: Streamlit UI for easy input and prediction.
- **Reproducible Notebooks**: Jupyter notebooks for model experimentation and analysis.
- **Pre-trained Artifacts**: Includes saved model weights and preprocessing objects.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some feature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a pull request.

## License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.