endpoint = 'http://420bab90-504b-4843-9ece-c5c4d5a349a1.southeastasia.azurecontainer.io/score' #Replace with your endpoint
key = 'MlFoNu6Sa4LV1Xg9j3C0RcniJUdVdLOb' #Replace with your key

from tracemalloc import StatisticDiff
import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.
itching = 0
skin_rash = 0
nodal_skin_eruption = 0
continuous_sneezing = 0
shivering = 0
chills = 0
joint_pain = 0
stomach_pain = 0
acidity = 0 
ulcers_on_tongue = 0
vomiting = 0
burning_micturition = 0
spotting_urination = 0
fatique = 0
weight_gain = 0
anxiety = 0
cold_hands_and_feets = 0
mood_swings = 0
weight_loss = 0
restlessness = 0
lethargy = 0
patches_in_throat = 0
irregular_sugar_level = 0
cough = 0
high_fever = 0
sunken_eyes = 0
breathlessness = 0
sweating = 0
dehydration = 0
indigestion =0
headache = 0
yellowish_skin = 0
dark_urine = 0
nausea = 0
loss_of_appetite = 0
pain_behind_the_eyes = 0
back_pain = 0
constipation = 0
abdominal_pain = 0
diarrhoea = 0
mild_fever = 0
yellow_urine = 0
yellowing_of_eyes = 0
acute_liver_failure = 0
fluid_overload = 0
swelling_of_stomach = 0
swelled_lymph_nodes = 0
muscle_wasting = 0
malise = 0
blurred_and_distored_vision = 0
phlegm = 0
throat_irritaion = 0
redness_of_eyes = 0
sinus_pressure = 0
runny_nose = 0
congestion = 0
chest_pain = 0
weakness_in_limbs = 0
fast_heart_rate = 0
pain_during_bowel_movement = 0
pain_in_anal_region = 0
bloody_stool = 0
irritation_in_anus = 0
neck_pain = 0
dizziness = 0
cramps = 0
bruising = 0
obesity = 0
swollen_legs = 0
swollen_blood_vessels = 0
puffy_face_and_eyes = 0
enlarged_thyroid = 0
brittles_nails = 0
swollen_extremeties = 0
excessive_hunger = 1
extra_marital_contacts = 1
drying_and_tingling_lips = 1
slurred_speech = 1
knee_pain = 1
hip_joint_pain = 1
muscle_weakness = 1
stif_neck = 1
swelling_joints = 1
movement_stiffness = 0
spinning_movements = 1
loss_of_balance = 1
unsteadiness = 1
weakness_of_one_body_side = 0
loss_of_smell = 0
bladder_discomfort = 0
foul_smell_of_urine = 0
continuous_feel_of_urine = 0
passage_of_gases = 0
internal_itching = 0
toxic_look = 1
depression = 1
irritability = 1
muscle_pain = 1
altered_sensorium = 1
red_spots_over_body = 1
belly_pain = 1
abnormal_menstruation = 1
dischromic_patches = 1
watering_from_eyes = 1
increased_appetite = 1
polyuria = 0
family_history = 0
mucoid_sputum = 0
rusty_sputum = 0
lack_of_concentration = 0
visual_disturbances = 0
receiving_blood_transfusion = 0
receiving_unsterile_injections = 0
coma = 1
stomach_bleeding = 1
distention_of_abdomen = 1
history_of_alcohal_consumption = 1
fluid_overload_1 = 1
blood_in_sputum = 0
prominent_veins_on_calf = 0
palpitations = 0
painful_walking = 0
pus_filled_pimples = 0
blackheads = 0
scurring = 0
skin_peeling = 0
silver_like_dusting = 0
small_dents_in_nails = 0
inflammatory_nails = 0
blister = 0
red_sore_around_nose = 0
yellow_crust_ooze = 0





# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
data =  {
  "Inputs": {
    "input1": [
      {
        "itching": itching,
        "skin_rash": skin_rash,
        "nodal_skin_eruptions": nodal_skin_eruption,
        "continuous_sneezing": continuous_sneezing,
        "shivering": shivering,
        "chills": chills,
        "joint_pain": joint_pain,
        "stomach_pain": stomach_pain,
        "acidity": acidity,
        "ulcers_on_tongue": ulcers_on_tongue,
        "muscle_wasting": muscle_wasting,
        "vomiting": vomiting,
        "burning_micturition": burning_micturition,
        "spotting_ urination": spotting_urination,
        "fatigue": fatique,
        "weight_gain": weight_gain,
        "anxiety": anxiety,
        "cold_hands_and_feets": cold_hands_and_feets,
        "mood_swings": mood_swings,
        "weight_loss": weight_loss,
        "restlessness": restlessness,
        "lethargy": lethargy,
        "patches_in_throat": patches_in_throat,
        "irregular_sugar_level": irregular_sugar_level,
        "cough": cough,
        "high_fever": high_fever,
        "sunken_eyes": sunken_eyes,
        "breathlessness": breathlessness,
        "sweating": sweating,
        "dehydration": dehydration,
        "indigestion": indigestion,
        "headache": headache,
        "yellowish_skin": yellowish_skin,
        "dark_urine": dark_urine,
        "nausea": nausea,
        "loss_of_appetite": loss_of_appetite,
        "pain_behind_the_eyes": pain_behind_the_eyes,
        "back_pain": back_pain,
        "constipation": constipation,
        "abdominal_pain": abdominal_pain,
        "diarrhoea": diarrhoea,
        "mild_fever": mild_fever,
        "yellow_urine": yellow_urine,
        "yellowing_of_eyes": yellowing_of_eyes,
        "acute_liver_failure": acute_liver_failure,
        "fluid_overload": fluid_overload,
        "swelling_of_stomach": swelling_of_stomach,
        "swelled_lymph_nodes": swelled_lymph_nodes,
        "malaise": malise,
        "blurred_and_distorted_vision": blurred_and_distored_vision,
        "phlegm": phlegm,
        "throat_irritation": throat_irritaion,
        "redness_of_eyes": redness_of_eyes,
        "sinus_pressure": sinus_pressure,
        "runny_nose": runny_nose,
        "congestion": congestion,
        "chest_pain": chest_pain,
        "weakness_in_limbs": weakness_in_limbs,
        "fast_heart_rate": fast_heart_rate,
        "pain_during_bowel_movements": pain_during_bowel_movement,
        "pain_in_anal_region": pain_in_anal_region,
        "bloody_stool": bloody_stool,
        "irritation_in_anus": irritation_in_anus,
        "neck_pain": neck_pain,
        "dizziness": dizziness,
        "cramps": cramps,
        "bruising": bruising,
        "obesity": obesity,
        "swollen_legs": swollen_legs,
        "swollen_blood_vessels": swollen_blood_vessels,
        "puffy_face_and_eyes": puffy_face_and_eyes,
        "enlarged_thyroid": enlarged_thyroid,
        "brittle_nails": brittles_nails,
        "swollen_extremeties": swollen_extremeties,
        "excessive_hunger": excessive_hunger,
        "extra_marital_contacts": extra_marital_contacts,
        "drying_and_tingling_lips": drying_and_tingling_lips,
        "slurred_speech": slurred_speech,
        "knee_pain": knee_pain,
        "hip_joint_pain": hip_joint_pain,
        "muscle_weakness": muscle_weakness,
        "stiff_neck": stif_neck,
        "swelling_joints": swelling_joints,
        "movement_stiffness": movement_stiffness,
        "spinning_movements": spinning_movements,
        "loss_of_balance": loss_of_balance,
        "unsteadiness": unsteadiness,
        "weakness_of_one_body_side":weakness_of_one_body_side,
        "loss_of_smell": loss_of_smell,
        "bladder_discomfort": bladder_discomfort,
        "foul_smell_of urine": foul_smell_of_urine,
        "continuous_feel_of_urine": continuous_feel_of_urine,
        "passage_of_gases": passage_of_gases,
        "internal_itching": internal_itching,
        "toxic_look_(typhos)": toxic_look,
        "depression": depression,
        "irritability": irritability,
        "muscle_pain": muscle_pain,
        "altered_sensorium": altered_sensorium,
        "red_spots_over_body": red_spots_over_body,
        "belly_pain": belly_pain,
        "abnormal_menstruation": abnormal_menstruation,
        "dischromic _patches": dischromic_patches,
        "watering_from_eyes": watering_from_eyes,
        "increased_appetite": increased_appetite,
        "polyuria": polyuria,
        "family_history": family_history,
        "mucoid_sputum": mucoid_sputum,
        "rusty_sputum": rusty_sputum,
        "lack_of_concentration": lack_of_concentration,
        "visual_disturbances": visual_disturbances,
        "receiving_blood_transfusion": receiving_blood_transfusion,
        "receiving_unsterile_injections": receiving_unsterile_injections,
        "coma": coma,
        "stomach_bleeding": stomach_bleeding,
        "distention_of_abdomen": distention_of_abdomen,
        "history_of_alcohol_consumption": history_of_alcohal_consumption,
        "fluid_overload_1": fluid_overload_1,
        "blood_in_sputum": blood_in_sputum,
        "prominent_veins_on_calf": prominent_veins_on_calf,
        "palpitations": palpitations,
        "painful_walking": painful_walking,
        "pus_filled_pimples": pus_filled_pimples,
        "blackheads": blackheads,
        "scurring": scurring,
        "skin_peeling": skin_peeling,
        "silver_like_dusting": silver_like_dusting,
        "small_dents_in_nails": small_dents_in_nails,
        "inflammatory_nails": inflammatory_nails,
        "blister": blister,
        "red_sore_around_nose": red_sore_around_nose,
        "yellow_crust_ooze": yellow_crust_ooze
      }
    ]
  },
  "GlobalParameters": {
    "Random number seed": 12345
  }
}

body = str.encode(json.dumps(data))

url = 'http://420bab90-504b-4843-9ece-c5c4d5a349a1.southeastasia.azurecontainer.io/score'
api_key = 'MlFoNu6Sa4LV1Xg9j3C0RcniJUdVdLOb' # Replace this with the API key for the web service

# The azureml-model-deployment header will force the request to go to a specific deployment.
# Remove this header to have the request observe the endpoint traffic rules
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    json_result = json.loads(result)
    output = json_result["Results"]["output1"][0]
    prediction = str(output)
    finalPrediction = prediction[19:-2]
    print(finalPrediction)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))