from flask import Flask, render_template, request
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re, string

model=pickle.load(open("model.pkl", "rb"))
vectorizer=pickle.load(open("vectorizer.pkl","rb"))

lemmatizer=WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text=text.lower()  
    text=re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  
    text=re.sub(r'\d+', ' ', text) 
    text=re.sub(r'\s+', ' ', text)  
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]) 
    return text

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
# if u dont want to use the templates folder then you have to directly return the HTML code in the function which is hard for large HTML files
# so we need to create a templates folder and put the HTML files in it 

@app.route('/analyze', methods=['POST'])
def predict():
    message = request.form['text']
    cleaned = clean_text(message)
    vect_text = vectorizer.transform([cleaned])
    prediction = model.predict(vect_text)[0]
    confidence = model.predict_proba(vect_text)[0]  # Add this
    result = "Fake" if prediction == 0 else "Not Fake"
    confidence_pct = round(confidence[prediction] * 100, 2)
    result += f" (Confidence: {confidence_pct}%)"
    # Return the result to the template 
    return render_template('home.html', result=result, confidence=confidence_pct, email=message)
if __name__ == '__main__':
    app.run(debug=True)