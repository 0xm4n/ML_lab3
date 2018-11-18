import cv2

# if you get error when running the following code, maybe you should change the relative paths to absolute paths.
if __name__ == '__main__':

    # load cascade classifier training file for haarcascade
    face_cascade = cv2.CascadeClassifier('/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    
    # load test image
    img = cv2.imread('/Users/Dave/Desktop/机器学习实验/实验3/ML2017-lab-03-master/face_detection/IMG_9687.JPG')

    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # draw the detected faces in the test image
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)

    # show and save the result
    cv2.imshow('img', img)
    cv2.imwrite('./result.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()