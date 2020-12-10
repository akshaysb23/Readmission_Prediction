import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'race':1, 'age':4, 'admission_type_id':0, 'discharge_disposition_id':0,'admission_source_id':0, 'time_in_hospital':0, 'num_lab_procedures':0, 'num_procedures':0, 'num_medications':0, 'number_outpatient':0,'number_emergency':0, 'number_inpatient':0, 'diag_1':0, 'diag_2':0, 'diag_3':0, 'number_diagnoses':0, 'max_glu_serum':0, 'A1Cresult':0, 'metformin':0, 'repaglinide':0, 'glimepiride':0, 'glipizide':0, 'glyburide':0, 'pioglitazone':0, 'rosiglitazone':0, 'miglitol':0, 'insulin':0, 'glyburide-metformin':0, 'Up_medicine':0, 'Down_medicine':0, 'Steady_medicine':0, 'change':0, 'diabetesMed':0})

print(r.json())