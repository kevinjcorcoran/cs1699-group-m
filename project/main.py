from flask import Blueprint, render_template, request
import os
from .predict import predict
from googlesearch import search

# Primary Directories
basePath = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(basePath, 'templates')

main = Blueprint('main', __name__)


@main.route('/')
@main.route('/index.html') 
def root():
    return render_template('index.html', page_title='Home')


@main.route('/about.html')
def about():
    return render_template('about.html', page_title='About')


@main.route('/upload_and_predict', methods=['GET', 'POST'])
def upload_and_predict():
    target = os.path.join(basePath, 'temp/')
    if request.method == 'POST':
        user_image = request.files['user-image']
        file_name = user_image.filename
        file_path = ''.join([target, file_name])
        user_image.save(file_path)

        results = predict(file_path)

        breed = (' '.join(word.capitalize() for word in results['breed'].split('_')))
        plural_breed = breed + 's' if not breed.endswith('s') else breed + 'es'

        precision = round((results['precision']*100), 2)
        precision_descriptor = get_precision_descriptor(results['precision'])

        os.remove(file_path)

        google_results = google_search_breed(breed)
        return render_template("results.html", page_title="Results", breed=breed, plural_breed=plural_breed, descriptor=precision_descriptor, search_results=google_results, precision=precision)
    

def get_precision_descriptor(precision):
    if precision >= 0.9:
        return 'is definitely'
    elif precision >= 0.8:
        return 'is almost definitely'
    elif precision >= 0.7:
        return 'probably'
    elif precision >= 0.6:
        return 'somewhat resembles'
    elif precision >= 0.5:
        return 'might be'
    elif precision >= 0.4:
        return 'could maybe be'
    elif precision >= 0.3:
        return 'kinda looks like'
    elif precision >= 0.2:
        return 'looks looks a little bit like'
    elif precision >= 0.1:
        return 'barely looks like'
    else:
        return 'doesn\'t look like anything, but I guess it\'s'


def google_search_breed(breed):
    query = f'information on the {breed} dog breed'
    return search(query, tld="co.in", num=3, stop=3, pause=2)
