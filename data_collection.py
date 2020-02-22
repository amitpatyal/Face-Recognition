import cv2
import os
import numpy as np
from helper_function import videoCaptureConnection, videoCaptureReleaseConnection, cvtColorgray_image, \
    faceCascadeClassifierDetectMultiScale, getCSVFileData


def face_extractor(image, capture_image):
    gray_image = cvtColorgray_image(image)
    face_image = faceCascadeClassifierDetectMultiScale(gray_image, capture_image)

    if face_image is not None:
        for (x_coordinate, y_coordinate, width, height) in face_image:
            cropped_face = image[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]
            return cropped_face
    else:
        return None


def data_collection():
    image_capture = videoCaptureConnection()
    names = np.array(getCSVFileData())
    face_id = input('\n enter user id end press enter ==>  ')
    # ​print("\n [INFO] Initializing face capture. Look the camera and wait ...")

    face_count = 0

    while True:
        ret, frame = image_capture.read()
        face_image = face_extractor(frame, image_capture)
        if face_image is not None:
            face_count += 1
            face_image = cv2.resize(face_image, (200, 200))
            face_image = cvtColorgray_image(face_image)
            face_dr = os.path.join('dataset', *names[int(face_id)].flatten() + face_id)
            if not os.path.isdir(face_dr):
                os.mkdir(face_dr)
            cv2.imwrite(face_dr + "/user_" + str(face_id) + '_' + str(face_count) + ".jpg", face_image)

            cv2.putText(img=face_image, text=str(face_count), org=(50, 50), fontFace=cv2.FONT_ITALIC, fontScale=1,
                        color=(255, 0, 0), lineType=2)
            cv2.imshow('Face Images', face_image)
        else:
            # print('Face not found !' + str(face_count))
            pass

        if cv2.waitKey(1) == 13 or face_count == 100:
            break

    videoCaptureReleaseConnection(image_capture)
    print('Colleting Samples Complete!!!')
