from flask import Blueprint, render_template, request
import os
from .predict import predict

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

        os.remove(file_path)

        return render_template("results.html", page_title="Results", breed=results['breed'], precision=results['precision'])
