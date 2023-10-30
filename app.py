from sklearn.feature_extraction.text import CountVectorizer


from flask import Flask, render_template, request
import pickle

tokenizer = pickle.load(open("templates/models/cv.pkl", "rb"))
model = pickle.load(open("templates/models/clf.pkl", "rb"))

app = Flask(__name__)
#says about home page
@app.route("/")
def home():
    return render_template("index.html")
    #home we don't need these things
    # text = ""
    # if request.method == 'POST':
    #     text = request.form.get('email-content')
    # return render_template('index.html' ,text = text)

@app.route("/predict", methods = ['POST']) # only get when click on button
def predict():
    
    message_text = request.form.get('email-content')
    cv = CountVectorizer()
    tokenized_message = cv.transform([message_text]).toarray()

    # tokenized_message = tokenizer.transform([message_text])


    prediction = model.predict(tokenized_message)
    prediction = int(prediction)
    prediction = 1 if prediction == 1 else 0
    return render_template("index.html", prediction = prediction, message_text = message_text)

if __name__ == "__main__":
    app.run( host = "0.0.0.0" , port = "8080" , debug = True)   #debug mode

