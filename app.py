#import statements
from flask import Flask, request, render_template
import tensorflow as tf
from keras.models import load_model

app = Flask(__name__)
#loading the trained model
model = tf.keras.models.load_model(r"model_10.h5")

#setting up routes
@app.route('/') 
def home():
    return render_template('index.html')

#image preprocessing -> converting it to grayscale and resizing the input image
def prepare(image):
    img_size=100
    img=tf.keras.preprocessing.image.load_img(image,color_mode="grayscale",target_size=(img_size,img_size))
    new_array = tf.keras.preprocessing.image.img_to_array(img)
    return new_array.reshape(-1,img_size,img_size,1)


@app.route('/predict',methods=['POST']) #post is used as the filename is posted from the webpage to the backend
def predict():
    image = request.files.get('filename','') #fetches the filename from the test dataset
    image = request.form["filename"]
    CATEGORIES = ["normal","opacity"]
    print("filename", image)
    image = "ChestXrays/test/"+image
    prediction = model.predict([prepare(image)/255.0]) #prediction is made using the model
    print(prediction)
    print(CATEGORIES[int(round(prediction[0][0]))])
    if CATEGORIES[int(round(prediction[0][0]))] == 'normal':
        output = "The patient does not have pneumonia"
    elif CATEGORIES[int(round(prediction[0][0]))] == 'opacity':
        output = "The patient does have pneumonia"
    #index.html is rendered and the prediction_text is displayed on the webpage
    return render_template('index.html', prediction_text = format(output)) 


if __name__ == "__main__":
    app.run(port=3000, debug=True)
