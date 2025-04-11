from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the trained models
with open('pcos_model.pkl', 'rb') as f:
    pcos_model = joblib.load(f)

with open('catboost_uti_model.pkl', 'rb') as f:
    uti_model = joblib.load(f)

# Load the scaler and selector for PCOS
with open('pcos_scaler.pkl', 'rb') as f:
    pcos_scaler = joblib.load(f)

with open('pcos_selector.pkl', 'rb') as f:
    pcos_selector = joblib.load(f)

# Home page
@app.route('/')
def home():
    return render_template('homepage2.html')

# About Us page
@app.route('/about')
def about():
    return render_template('about.html')

# Contact page
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        return render_template('contact.html', submitted=True, name=name, message=message)
    return render_template('contact.html', submitted=False)

# GDDES system dropdown routes
@app.route('/pcos', methods=['GET', 'POST'])
def pcos():
    result = None
    if request.method == 'POST':
        try:
            input_features = [
                float(request.form['Age (yrs)']),
                float(request.form['Weight (Kg)']),
                float(request.form['Height(Cm)']),
                float(request.form['BMI']),
                float(request.form['Blood Group']),
                float(request.form['Pulse rate(bpm)']),
                float(request.form['RR (breaths/min)']),
                float(request.form['Hb(g/dl)']),
                1 if request.form['Cycle(R/I)'] == 'R' else 0,
                float(request.form['Cycle length(days)']),
                float(request.form['Marraige Status (Yrs)']),
                1 if request.form['Pregnant(Y/N)'].lower() == 'yes' else 0,
                float(request.form['No. of aborptions']),
                float(request.form['FSH(mIU/mL)']),
                float(request.form['LH(mIU/mL)']),
                float(request.form['FSH/LH']),
                float(request.form['Hip(inch)']),
                float(request.form['Waist(inch)']),
                float(request.form['Waist:Hip Ratio']),
                float(request.form['TSH (mIU/L)']),
                float(request.form['AMH(ng/mL)']),
                float(request.form['PRL(ng/mL)']),
                float(request.form['Vit D3 (ng/mL)']),
                float(request.form['PRG(ng/mL)']),
                float(request.form['RBS(mg/dl)']),
                1 if request.form['Weight gain(Y/N)'].lower() == 'yes' else 0,
                1 if request.form['hair growth(Y/N)'].lower() == 'yes' else 0,
                1 if request.form['Skin darkening (Y/N)'].lower() == 'yes' else 0,
                1 if request.form['Hair loss(Y/N)'].lower() == 'yes' else 0,
                1 if request.form['Pimples(Y/N)'].lower() == 'yes' else 0,
                1 if request.form['Fast food (Y/N)'].lower() == 'yes' else 0,
                1 if request.form['Reg.Exercise(Y/N)'].lower() == 'yes' else 0,
                float(request.form['BP _Systolic (mmHg)']),
                float(request.form['BP _Diastolic (mmHg)']),
                float(request.form['Follicle No. (L)']),
                float(request.form['Follicle No. (R)']),
                float(request.form['Avg. F size (L) (mm)']),
                float(request.form['Avg. F size (R) (mm)']),
                float(request.form['Endometrium (mm)'])
            ]

            input_array = np.array([input_features])
            print("Scaler type:", type(pcos_scaler))
            print("Selector type:", type(pcos_selector))
            scaled_input = pcos_scaler.transform(input_array)
            selected_input = pcos_selector.transform(scaled_input)
            prediction = pcos_model.predict(selected_input)
            result = 'Oh nooo!YOU have PCOS' if prediction[0] == 1 else 'Hurrayy!Your condition is normal,No PCOS'

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('pcos.html', result=result)

@app.route('/uti', methods=['GET', 'POST'])
def uti():
    result = None
    if request.method == 'POST':
        data = request.form
        feature_names = [
            'Temperature of patient',
            'Occurrence of nausea',
            'Lumbar pain',
            'Urine pushing (continuous need for urination)',
            'Micturition pains',
            'Burning of urethra, itch, swelling of urethra outlet',
            'Inflammation of urinary bladder'
        ]
        values = []
        for feature in feature_names:
            if feature == 'Temperature of patient':
                try:
                    values.append(float(data.get(feature)))
                except:
                    values.append(0.0)
            else:
                values.append(1 if data.get(feature) == 'yes' else 0)

        input_df = pd.DataFrame([values], columns=feature_names)
        pred = uti_model.predict(input_df)[0]
        result = "You have UTI" if pred == 1 else "Your Condition is normal,no UTI"
    return render_template('uti.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
