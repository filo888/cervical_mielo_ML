# README: Cervical Mielopathy Prediction Model


## Overview
This repository contains three pre-trained machine learning models for analyzing predictors of improvement in clinical outcomes. These models focus on 3 key measures:
- **Neck VAS (Visual Analog Scale)**
- **Arm VAS (Visual Analog Scale)**
- **mJOA Score**


The models were trained using PyCaret Library (Version 3.3.2) to predict improvement (Class 1) or no improvement (Class 0) in the corresponding target using a diverse set of clinical and demographic features. They are designed for researchers and clinicians as a starting point for a possible future multi-center study with external validation.

---

## Features Used by the Models
The following features are included in the models:


**Features with "###" are encoded using our specific convention**


**NB: Please refer to "Variables Encoding" to see how each # feature is encoded for model use**

   **Features List**
   - ASA Score
   - BMI (Body Mass Index) ###
   - Preoperative Grade of Myelopathy (mJOA)
   - Levels of Cervical Pathology ###
   - Levels of Radiological Myelopathy ###
   - Extent of Compression ###
   - Sex ###
   - Type of Approach ###
   - Smoking Status ###
   - Previous Cervical Surgery ###
   - Predominance of Site of Compression ###
   - Cervical Alignment ###
   - Type of Compression ###
   - Age
   - Symptoms Duration (days) 
   - Neck VAS
   - Arm VAS
   - NDI (Neck Disability Index) ###
   - Charlson Index

---

## Model Performance
Key performance highlights:
each model is a tuned RF Classifier with a custom classification threshold 

1. **Neck VAS Model**
   - Top Features: Neck VAS, Arm VAS, NDI
   - High accuracy for predicting improvement with balanced feature importance.

2. **Arm VAS Model**
   - Top Features: Arm VAS, Cervical Alignment, Compression Predominance
   - Worse than the Neck VAs in predicting the 0 class for this target.

3. **mJOA Score Model**
   - Top Features: BMI, Smoking Status, Type of Approach
   - Suffers the high imbalanced dataset used for training with poor perfomance on 0 class.

---

## Repository Contents
- **Models:**
  - `Neck_VAS_model.pkl`
  - `Arm_VAS_model.pkl`
  - `mJOA_model.pkl`

- **Utility Files:**
  - `requirements.txt`: Lists all necessary Python packages.

---

## Setup and Requirements
To use these models, install the required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyCaret 3.3.2
- pandas
- numpy
- scikit-learn
- matplotlib
- pickle

---

## Using the Models
The models are compatible with PyCaret, a low-code machine learning library. Below is a step-by-step tutorial on how to load and test the models:

### 1. Import Required Libraries
```python
from pycaret.classification import load_model, predict_model
import pandas as pd
```

### 2. Load a Pre-Trained Model
Choose the model you want to use (e.g., Neck VAS):
```python
# Load the model
model = load_model('Neck_VAS_model')
```

### 3. Prepare Input Data
Create a DataFrame with the required features. 

Ensure that the column names match exactly (case and space sesnitive) the column names in column_names.txt (a raw model input example with a few features is shown below):


```python
data = pd.DataFrame({
    'Neck VAS': [7],
    'Arm VAS': [6],
    'Age': [45],
    'BMI': [1.0],
    'Type of Approach': [1.0],
    'Extent of Compression': [3],
    'Cervical Alignment': [1.0],
    'Predominance of Compression Site': [2.0]
})
```

### 4. Make Predictions
Use the `predict_model` function to test the model:
```python
predictions = predict_model(model, data=data)
print(predictions)
```

### 5. Explore Additional Parameters
Refer to the original PyCaret documentation for detailed insights into the functions, their parameters, and customization options:
- [PyCaret Documentation](https://pycaret.gitbook.io/docs)

---

## Additional Notes
1. These models are not intended for direct clinical decision-making. Always consult clinical guidelines and experts.
2. For any issues or suggestions, please create a new issue in the repository.

---

## Authors
Filippo Colella

---

## License
This repository is licensed. See `LICENSE` for details.


   ```
