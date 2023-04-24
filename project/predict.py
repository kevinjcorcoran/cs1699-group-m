import numpy as np
import tensorflow as tf
import os

# Primary Directories
basePath = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(basePath, 'templates')

static_dir = os.path.join(basePath, 'static/')

model_file = os.path.join(basePath, 'model/eightyfiveepoch_512batch_breed.h5')
model = tf.keras.models.load_model(model_file, compile=True)

class_names = ['Afghan_hound', 'African_hunting_dog', 'Airedale',
    'American_Staffordshire_terrier', 'Bernese_mountain_dog', 'Blenheim_spaniel',
    'Boston_bull', 'Bouvier_des_Flandres', 'Brittany_spaniel', 'Cardigan',
    'Chihuahua', 'Doberman', 'English_foxhound', 'English_setter',
    'English_springer', 'French_bulldog', 'German_shepherd',
    'German_short-haired_pointer', 'Gordon_setter', 'Great_Dane',
    'Great_Pyrenees', 'Greater_Swiss_Mountain_dog', 'Irish_setter',
    'Irish_water_spaniel', 'Irish_wolfhound', 'Italian_greyhound',
    'Japanese_spaniel', 'Labrador_retriever', 'Lakeland_terrier', 'Leonberg',
    'Lhasa', 'Maltese_dog', 'Mexican_hairless', 'Newfoundland',
    'Norwegian_elkhound', 'Old_English_sheepdog', 'Pekinese', 'Pembroke',
    'Pomeranian', 'Rhodesian_ridgeback', 'Rottweiler', 'Saint_Bernard', 'Samoyed',
    'Scotch_terrier', 'Scottish_deerhound', 'Shetland_sheepdog', 'Shih-Tzu',
    'Siberian_husky', 'Staffordshire_bullterrier', 'Sussex_spaniel',
    'Tibetan_mastiff', 'Weimaraner', 'Welsh_springer_spaniel',
    'Yorkshire_terrier', 'affenpinscher', 'american_pit_bull_terrier', 'basenji',
    'basset', 'beagle', 'bloodhound', 'bluetick', 'borzoi', 'boxer', 'briard',
    'bull_mastiff', 'cairn', 'chow', 'clumber', 'cocker_spaniel', 'collie',
    'dhole', 'dingo', 'golden_retriever', 'groenendael', 'keeshond', 'kelpie',
    'komondor', 'kuvasz', 'malamute', 'malinois', 'miniature_pinscher',
    'papillon', 'pug', 'schipperke', 'shiba_inu', 'standard_poodle',
    'standard_schnauzer', 'toy_poodle', 'toy_terrier', 'vizsla',
    'wire-haired_fox_terrier']


def predict(file_path):
    img_height = 180
    img_width = 180

    img = tf.keras.utils.load_img(file_path, target_size=(img_height, img_width)) 
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    breed = class_names[np.argmax(score)]
    precision = np.max(score)

    results = {'breed': breed, 'precision': precision}
    
    return results

