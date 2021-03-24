from flask import Flask, request, jsonify, render_template
import pickle



app = Flask(__name__)
model = pickle.load(open('nlp.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    from spacy import displacy
    text_features =  [x for x in request.form.values() ]
    example = ''.join(text_features)
    doc = model(example)
    
    ENTITY = []
    for ent in doc.ents:
        ENTITY.append((ent.text, ent.label_))
    
#     output = displacy.render(doc, style='ent')

    return render_template('index.html', prediction_text = 'NER for the given text is {}'.format(ENTITY))
    


if __name__ == "__main__":
    app.run(debug=True)