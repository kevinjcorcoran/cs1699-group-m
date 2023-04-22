import numpy as np
import tensorflow as tf
import os

# Primary Directories
basePath = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(basePath, 'templates')

static_dir = os.path.join(basePath, 'static/')

model_file = os.path.join(basePath, 'model/eightyfiveepoch_512batch_breed.h5')
model = tf.keras.models.load_model(model_file, compile=False)
model.compile()

class_names = ['affenpinscher', 'Afghan_hound', 'African_hunting_dog', 'Airedale', 'american_pit_bull_terrier', 'American_Staffordshire_terrier', 'basenji', 'basset', 'beagle', 'Bernese_mountain_dog', 'Blenheim_spanial', 'bloodhound', 'bluetick', 'borzoi', 'Boston_bull', 'Bouvier_des_Flandres', 'boxer', 'briard', 'Brittany_spaniel', 'bull_mastiff', 'cairn', 'Cardigan', 'Chihuahua', 'chow', 'clumber', 'cocker_spaniel', 'collie', 'dhole', 'dingo', 'Doberman', 'English_foxhound', 'English_setter', 'English_springer', 'French_bulldog', 'German_shepherd', 'German_short-haired_pointer', 'golden_retriever', 'Gordon_setter', 'Great_Dane', 'Great_Pyrenees', 'Greater_Swiss_Mountain_dog', 'groenendael', 'Irish_setter', 'Irish_water_spaniel', 'Irish_wolfhound', 'Italian_greyhound', 'Japanese_spaniel', 'keeshond', 'kelpie', 'komondor', 'kuvasz', 'Labrador_retriever', 'Lakeland_terrier', 'Leonberg', 'Lhasa', 'malamute', 'malinois', 'Maltese_dog', 'Mexican_hairless', 'miniature_pinscher', 'Newfoundland', 'Norwegian_elkhound', 'Old_English_sheepdog', 'papillon', 'Pekinese', 'Pembroke', 'Pomeranian', 'pug', 'Rhodesian_ridgeback', 'Rottweiler', 'Saint_Bernard', 'Samoyed', 'schipperke', 'Scotch_terrier', 'Scottish_deerhound', 'Shetland_sheepdog', 'shiba_inu', 'Shih-Tzu', 'Siberian_husky', 'Staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer', 'Sussex_spanial', 'Tibetan_mastiff', 'toy_poodle', 'toy_terrier', 'vizsla', 'Weimaraner', 'Welsh_springer_spaniel', 'wire-haired_fox_terrier', 'Yorkshire_terrier']


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

