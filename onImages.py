import cv2
import numpy as np
import pytesseract
from imutils.object_detection import non_max_suppression

pytesseract.pytesseract.tesseract_cmd = 'C://Program Files//Tesseract-OCR//tesseract.exe'
net = cv2.dnn.readNet("frozen_east_text_detection.pb")


def text_detector(image1):
    # hasFrame, image = cap.read()
    orig1 = image1
    (H, W) = image1.shape[:2]
    (newW, newH) = (640, 320)
    rW = W / float(newW)
    rH = H / float(newH)
    image1 = cv2.resize(image1, (newW, newH))
    (H, W) = image1.shape[:2]
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    blob = cv2.dnn.blobFromImage(image1, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.5:
                continue
            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        r = orig1[startY:endY, startX:endX]
        # configuration setting to convert image to string.
        configuration = "-l eng --oem 1 --psm 8"
        # This will recognize the text from the image of bounding box
        text = pytesseract.image_to_string(r, config=configuration)
        print(text)
        # draw the bounding box on the image
        cv2.rectangle(orig1, (startX, startY), (endX, endY), (0, 255, 0), 3)
    return orig1


# cv2.imshow("Text Detection", orig)
# k = cv2.waitKey(30) & 0xff
# if k == 27:
# 	break


image = cv2.imread('love.jpg')
image2 = cv2.imread('one.jpg')
image3 = cv2.imread('crime.jpg')
image4 = cv2.imread('hands.jpg')
image5 = cv2.imread('beautiful.jpg')
image6 = cv2.imread('everyday.jpg')

array = [image]  # ,image2,image3,image4,image5,image6]

for i in range(0, 1):
    for img in array:
        imageO = cv2.resize(img, (640, 320), interpolation=cv2.INTER_AREA)
        imageX = imageO
        orig = text_detector(imageO)
        cv2.imshow("Text Detection", orig)
        cv2.imwrite("lovetext.jpg", orig)
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break
# cv2.destroyAllWindows()
