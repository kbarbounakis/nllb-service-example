from flask import Flask, render_template, request, Response
from os import path
from .translate import TranslateRequest, translate, TranslateResponse

# set static folder
static_folder = path.join(path.abspath(path.dirname(__file__)), 'templates/assets')
app = Flask(__name__, static_folder=static_folder)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def handle_translate():
    try:
        # get body
        body: dict = request.json
        # validate request
        TranslateRequest.validate(body)
        # get question and table schema
        req = TranslateRequest(**body)
        # get result
        result = translate(req.text)
        # send response
        return TranslateResponse(result=result)._asdict()
    except Exception as err:
        return {
            'message': err.message,
            'statusCode': 500,
            'type': type(err).__name__
        }, 500
    
