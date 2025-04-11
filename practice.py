# Try this in a Python shell to test
import pickle
import numpy as np

with open('pcos_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Normal input sample (manually selected or based on training data)
sample_input = np.array([[21, 56, 165, 20.6, 3, 76, 18, 13.5, 1, 30, 0, 0, 0,
                          7.5, 6.5, 1.15, 36, 27, 0.75, 2.5, 3.0, 18.0,
                          28.0, 7.0, 90, 0, 0, 0, 0, 0, 0, 1,
                          115, 75, 5, 5, 10, 10, 8]])
print("Predicted:", model.predict(sample_input))
