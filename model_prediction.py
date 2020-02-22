import numpy as np
import cv2
from helper_function import cvtColorgray_image, faceCascadeClassifierDetectMultiScale, modelFaceRecognizer\
    , videoCaptureConnection, videoCaptureReleaseConnection, getCSVFileData

def face_detector(image, capture_image):
    gray_image = cvtColorgray_image(image)
    face_image = faceCascadeClassifierDetectMultiScale(gray_image, capture_image)

    if face_image is None():
        return image ,[]

    for(x, y, w, h) in face_image:
        cv2.rectangle(image, (x, y) ,(x + w, y + h) ,(0 ,255 ,255) ,2)
        region_of_interest = image[y:y + h, x:x + w]
        # region_of_interest = cv2.resize(region_of_interest, (200,200))

    return image ,region_of_interest

def getModelPrediction():
    modelFaceRecognizer.read('trainee_data/trainer.yml')
    font = cv2.FONT_ITALIC

    image_capture = videoCaptureConnection()
    while (True):
        ret, frame = image_capture.read()
        grayImage = cvtColorgray_image(frame)
        faces = faceCascadeClassifierDetectMultiScale(frame, image_capture)
        if faces is ():
            cv2.putText(frame, 'Face is not recognized please try again !!', (10 ,60), font, 1, (255, 255, 0), 1)
            pass
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                id, confidence = modelFaceRecognizer.predict(cv2.resize(grayImage[y:y + h, x:x + w] ,(200, 200)))
                names = np.array(getCSVFileData())
                confidence = round(int(100 * (1 - (confidence / 300))))

                if (confidence > 75):
                    name = names[id]
                    confidence = '{0}%'.format(confidence)
                else:
                    name = names[0]
                    # confidence = '{0}%'.format(confidence)
                    confidence = 'Face not found !!'
                cv2.putText(frame, *name.flatten(), (x + 5, y - 5), font, 1, (255, 255, 0), 2)  # * + flatten() is used to removed the [] from np.array function
                cv2.putText(frame, confidence, (x + 5, y + h - 5), font, 1, (255, 255, 0), 2)

        cv2.imshow('Face Recognition', frame)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
    videoCaptureReleaseConnection(image_capture)