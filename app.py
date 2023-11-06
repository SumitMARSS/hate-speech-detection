from flask import Flask, render_template, request
import pickle

cv = pickle.load(open("templates/models/new_cv.pkl", "rb"))
model = pickle.load(open("templates/models/new_clf.pkl", "rb"))
#create an object of flask name
app = Flask(__name__)
#says about home page
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ['POST']) # only get when click on button
def predict():

    if request.method == 'POST':
        # get message from form

        message_text = request.form.get('content')
        data = cv.transform([message_text]).toarray()
        #pridicting by using our model
        prediction = model.predict(data)[0]

        print(prediction)
        return render_template("index.html", prediction = prediction, message_text = message_text)

if __name__ == "__main__":
    app.run( host = "0.0.0.0" , port = "8080" , debug = True)   #debug mode
