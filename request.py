import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Pregnancies':8, 'Glucose':150, 'BloodPressure':60, 'SkinThickness':50, 'Insulin':20, 'BMI': 50, 'DiabetesPedigreeFunction':2, 'Age':30})

print(r.json())

