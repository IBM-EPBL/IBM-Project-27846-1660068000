# import the necessary packages
from flask import Flask, render_template
import cv2
from tensorflow.keras.models import load_model  # to load our trained model
import numpy as np
from skimage.transform import resize

app = Flask(__name__, template_folder="templates")

model = load_model(
    'Project Development Phase\Sprint 2\Model Building\model.h5')
print("Loaded model from disk")
vals = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/home', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def predict():

    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(0)
    (W, H) = (None, None)

    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        output = frame.copy()
        img = resize(frame, (64, 64, 1))
        img = np.expand_dims(img, axis=0)
        if (np.max(img) > 1):
            img = img/255.0
        result = np.argmax(model.predict(img), axis=-1)
        index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        result = str(index[result[0]])

        cv2.putText(output, "It indicates: {}".format(result), (10, 120), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 0), 2, )

        cv2.imshow("Output", output)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    print("[INFO] cleaning up...")
    vs.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
