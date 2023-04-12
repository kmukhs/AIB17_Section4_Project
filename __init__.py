from flask import Flask, request, jsonify
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv('C:/Users/admin/Desktop/df_cleaning_2021.csv')

X = ['AAE', 'EHW', 'Industry']
y = 'Injuries'

train, test = train_test_split(df, test_size=0.2, stratify=df[y], random_state=42)
train, val = train_test_split(train, test_size=0.2, stratify=train[y], random_state=42)

X_train, y_train = train[X], train[y]
X_val, y_val = val[X], val[y]
X_test, y_test = test[X], test[y]

cat_cols = ['Industry']
ordinal_encoder = OrdinalEncoder()
X_train[cat_cols] = ordinal_encoder.fit_transform(X_train[cat_cols])
X_val[cat_cols] = ordinal_encoder.transform(X_val[cat_cols])
X_test[cat_cols] = ordinal_encoder.transform(X_test[cat_cols])

num_cols = ['AAE', 'EHW']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

pp2 = make_pipeline(
    SimpleImputer(),
    RandomForestClassifier(random_state = 42)
)

ps2 = {
    "simpleimputer__strategy": ["median", "mean"],
    "randomforestclassifier__max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "randomforestclassifier__n_estimators" : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
    "randomforestclassifier__max_samples" : [0.2, 0.4, 0.6, 0.8, 1.0],
    "randomforestclassifier__max_features" : [0.2, 0.4, 0.6, 0.8, 1.0]
}

clf1 = RandomizedSearchCV(
    pp2,
    param_distributions = ps2,
    n_iter = 10,
    cv = 5,
    n_jobs = -1,
    random_state = 42
)

clf1.fit(X_train, y_train)

val_score = clf1.best_estimator_.score(X_val, y_val)
print("Validation score: {:.2f}%".format(val_score*100))

test_score = clf1.best_estimator_.score(X_test, y_test)
print("Test score: {:.2f}%".format(test_score*100))

joblib.dump(clf1.best_estimator_, 'model.pkl')

from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__, template_folder='templates')

model = joblib.load('model.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        AAE = request.form['AAE']
        EHW = request.form['EHW']
        Industry = request.form['Industry']

        Industry_encoded = ordinal_encoder.transform([[Industry]])[0][0]
        
        input_data = np.array([[AAE, EHW, Industry_encoded]])
        prediction_proba = model.predict_proba(input_data)[0][1]
        
        if prediction_proba < 0.5:
            result = "산업재해 발생 확률이 낮아 안전합니다"
        elif prediction_proba < 0.8:
            result = "산업재해 발생 확률이 비교적 높습니다. 주의하세요!"
        else:
            result = "산업재해 발생 확률이 매우 높습니다. 직원 수에 따라 적절한 근무환경을 조성하거나 근로시간을 조정해주세요." 
        
        return render_template('index.html', prediction_text=result)
    
    else:
        return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)
