endpoint = 'http://3e238058-4ff1-4f25-ad86-7c3a72a0f682.southeastasia.azurecontainer.io/score' #Replace with your endpoint
key = '65RJs9zoLwrJL9fjZ5U0tZr2us7Ud5yb' #Replace with your key

import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
data =  {
  "Inputs": {
    "input1": [
      {
        "itching": 0,
        "skin_rash": 0,
        "nodal_skin_eruptions": 0,
        "continuous_sneezing": 0,
        "shivering": 0,
        "chills": 0,
        "joint_pain": 0,
        "stomach_pain": 0,
        "acidity": 0,
        "ulcers_on_tongue": 0,
        "muscle_wasting": 0,
        "vomiting": 0,
        "burning_micturition": 0,
        "spotting_ urination": 0,
        "fatigue": 0,
        "weight_gain": 0,
        "anxiety": 0,
        "cold_hands_and_feets": 0,
        "mood_swings": 0,
        "weight_loss": 0,
        "restlessness": 0,
        "lethargy": 0,
        "patches_in_throat": 0,
        "irregular_sugar_level": 0,
        "cough": 0,
        "high_fever": 0,
        "sunken_eyes": 0,
        "breathlessness": 0,
        "sweating": 0,
        "dehydration": 0,
        "indigestion": 0,
        "headache": 0,
        "yellowish_skin": 0,
        "dark_urine": 0,
        "nausea": 0,
        "loss_of_appetite": 0,
        "pain_behind_the_eyes": 0,
        "back_pain": 0,
        "constipation": 0,
        "abdominal_pain": 0,
        "diarrhoea": 0,
        "mild_fever": 0,
        "yellow_urine": 0,
        "yellowing_of_eyes": 0,
        "acute_liver_failure": 0,
        "fluid_overload": 0,
        "swelling_of_stomach": 0,
        "swelled_lymph_nodes": 0,
        "malaise": 0,
        "blurred_and_distorted_vision": 0,
        "phlegm": 0,
        "throat_irritation": 0,
        "redness_of_eyes": 0,
        "sinus_pressure": 0,
        "runny_nose": 0,
        "congestion": 0,
        "chest_pain": 0,
        "weakness_in_limbs": 0,
        "fast_heart_rate": 0,
        "pain_during_bowel_movements": 0,
        "pain_in_anal_region": 0,
        "bloody_stool": 0,
        "irritation_in_anus": 0,
        "neck_pain": 0,
        "dizziness": 0,
        "cramps": 0,
        "bruising": 0,
        "obesity": 0,
        "swollen_legs": 0,
        "swollen_blood_vessels": 0,
        "puffy_face_and_eyes": 0,
        "enlarged_thyroid": 0,
        "brittle_nails": 0,
        "swollen_extremeties": 0,
        "excessive_hunger": 0,
        "extra_marital_contacts": 0,
        "drying_and_tingling_lips": 0,
        "slurred_speech": 0,
        "knee_pain": 0,
        "hip_joint_pain": 0,
        "muscle_weakness": 0,
        "stiff_neck": 0,
        "swelling_joints": 0,
        "movement_stiffness": 0,
        "spinning_movements": 0,
        "loss_of_balance": 0,
        "unsteadiness": 0,
        "weakness_of_one_body_side": 0,
        "loss_of_smell": 0,
        "bladder_discomfort": 0,
        "foul_smell_of urine": 0,
        "continuous_feel_of_urine": 0,
        "passage_of_gases": 0,
        "internal_itching": 0,
        "toxic_look_(typhos)": 0,
        "depression": 0,
        "irritability": 0,
        "muscle_pain": 0,
        "altered_sensorium": 0,
        "red_spots_over_body": 0,
        "belly_pain": 0,
        "abnormal_menstruation": 1,
        "dischromic _patches": 1,
        "watering_from_eyes": 1,
        "increased_appetite": 0,
        "polyuria": 1,
        "family_history": 0,
        "mucoid_sputum":0,
        "rusty_sputum": 1,
        "lack_of_concentration": 0,
        "visual_disturbances": 0,
        "receiving_blood_transfusion": 0,
        "receiving_unsterile_injections": 0,
        "coma": 0,
        "stomach_bleeding": 0,
        "distention_of_abdomen": 0,
        "history_of_alcohol_consumption": 0,
        "fluid_overload_1": 0,
        "blood_in_sputum": 0,
        "prominent_veins_on_calf": 0,
        "palpitations": 0,
        "painful_walking": 1,
        "pus_filled_pimples": 1,
        "blackheads": 1,
        "scurring": 1,
        "skin_peeling": 1,
        "silver_like_dusting": 1,
        "small_dents_in_nails": 1,
        "inflammatory_nails": 1,
        "blister": 1,
        "red_sore_around_nose": 1,
        "yellow_crust_ooze": 1
      }
    ]
  },
  "GlobalParameters": {
    "Random number seed": 12345
  }
}

body = str.encode(json.dumps(data))

url = 'http://3e238058-4ff1-4f25-ad86-7c3a72a0f682.southeastasia.azurecontainer.io/score'
api_key = '65RJs9zoLwrJL9fjZ5U0tZr2us7Ud5yb' # Replace this with the API key for the web service

# The azureml-model-deployment header will force the request to go to a specific deployment.
# Remove this header to have the request observe the endpoint traffic rules
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()

    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))
