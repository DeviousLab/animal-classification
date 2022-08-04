from flask import Flask, render_template, request
import os

import numpy as np
import scipy
import pickle

# skimage
import skimage
import skimage.color
import skimage.transform
import skimage.feature
import skimage.io

app = Flask(__name__)
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/uploads/')
MODEL_PATH = os.path.join(BASE_PATH, 'static/models/')

model_sgd = pickle.load(
    open(os.path.join(MODEL_PATH, 'dsa_image_classification_sgd.pickle'), 'rb'))
scaler_transform = pickle.load(
    open(os.path.join(MODEL_PATH, 'dsa_scaler.pickle'), 'rb'))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['animal_image']
        print('File name is', uploaded_file.filename)

        extensions = uploaded_file.filename.split('.')[-1]
        print('File extension is', extensions)

        if extensions.lower() in ['jpg', 'jpeg', 'png']:
            uploaded_file.save(os.path.join(
                UPLOAD_PATH, uploaded_file.filename))
            print('File uploaded successfully')
            res = pipeline_model(os.path.join(
                UPLOAD_PATH, uploaded_file.filename), scaler_transform, model_sgd)
            confirmed_animal = (list(res.keys())[0]).capitalize()
            return render_template('upload.html', file_upload=True, data=res, image_uploaded=uploaded_file.filename, animal=confirmed_animal)
        else:
            print('File extension not supported')
            return render_template('upload.html', file_upload=False)
    else:
        return render_template('upload.html', file_upload=False)


def pipeline_model(path, scaler_transform, model_sgd):
    image = skimage.io.imread(path)
    image_resize = skimage.transform.resize(image, (80, 80))
    image_scale = 255*image_resize
    image_transform = image_scale.astype(np.uint8)
    gray = skimage.color.rgb2gray(image_transform)
    feature_vector = skimage.feature.hog(
        gray, orientations=10, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

    scalex = scaler_transform.transform(feature_vector.reshape(1, -1))
    result = model_sgd.predict(scalex)
    decision_value = model_sgd.decision_function(scalex).flatten()
    labels = model_sgd.classes_
    z = scipy.stats.zscore(decision_value)
    prob_value = scipy.special.softmax(z)

    top_5_prob_ind = prob_value.argsort()[::-1][:5]
    top_labels = labels[top_5_prob_ind]
    top_prob = prob_value[top_5_prob_ind]
    top_dict = dict()
    for key, val in zip(top_labels, top_prob):
        top_dict.update({key: np.round(val, 3)})

    return top_dict


if __name__ == '__main__':
    app.run(debug=True)
