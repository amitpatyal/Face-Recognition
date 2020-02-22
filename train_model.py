import numpy as np
import os
from PIL import Image
from helper_function import faceCascade, modelFaceRecognizer

def getImagesAndLabels():
    path = 'dataset'
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePathF in imagePaths:
        imagePathF = [os.path.join(imagePathF, f) for f in os.listdir(imagePathF)]
        for imagePath in imagePathF:
            PIL_image = Image.open(imagePath).convert('L')  # convert it to GRAYSCALE image
            img_numpy = np.array(PIL_image, 'uint8')
            fid = int(os.path.split(imagePath)[-1].split('_')[1])
            faces = faceCascade.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(fid)

    return faceSamples, ids

def train_data_model():
    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels()
    modelFaceRecognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    modelFaceRecognizer.write('trainee_data/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))