from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
@app.route('/index.html') 
def root():
    return render_template('index.html', page_title='Home')


@app.route('/about.html')
def about():
    return render_template('about.html', page_title="About")


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
