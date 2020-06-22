import cv2
import numpy as np


def in_roi(ex1, ey1, ew1, eh1, roi_list1):
    for (fx, fy, fw, fh) in roi_list1:
        if fx < ex1 < (fx + fw) and fy < ey1 < (fy + fh) and \
                fx < (ex1 + ew1) < (fx + fw) and fy < (ey1 + eh1) < (fy + fh):
            return True

cascade1 = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cascade2 = cv2.CascadeClassifier('haarcascade_upperbody.xml')
cascade3 = cv2.CascadeClassifier('haarcascade_lowerbody.xml')

cap = cv2.VideoCapture('View_001.avi')
# cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('outputSD.avi', fourcc, 20.0, (1200, 800))

ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)
while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    SD = 0
    PSDV = 0
    SDV = 0

    boxes = []

    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            (x, y, w, h) = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

    boxes1 = []

    for (bx, by, bw, bh) in boxes:
        if not in_roi(bx, by, bw, bh, boxes):
            boxes1.append((bx, by, bw, bh))

    for (x, y, w, h) in boxes1:
        rx = int(x - w)
        ry = int(y - h)
        rx2 = int(x + w*2)
        ry2 = int(y + h*2)
        # cv2.rectangle(frame1, (rx, ry), (rx2, ry2), (255, 255, 0), 2)
        # cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 255), 2)
        # ROIs.append((x, y, w, h))
        roi_gray = gray1[ry:ry2, rx:rx2]
        roi_color = frame1[ry:ry2, rx:rx2]

        person_count = 0
        person1 = []
        body = []
        body1 = cascade1.detectMultiScale(roi_gray, 1.1, 5)
        body2 = cascade2.detectMultiScale(roi_gray, 1.1, 5)
        body3 = cascade3.detectMultiScale(roi_gray, 1.1, 5)

        for (x1, y1, w1, h1) in body1:
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            body.append((x1, y1, w1, h1))

        for (x2, y2, w2, h2) in body2:
            tx = int(x2 + 0.2 * w2)
            ty = int(y2 + 0.2 * h2)
            tw = int(0.6 * w2)
            th = int(0.6 * h2)
            if not in_roi(tx, ty, tw, th, body):
                body.append((x2, y2, w2, 3 * h2))
                # cv2.rectangle(img, (tx, ty), (tx + tw, ty + th), (255, 255, 0), 2)

        for (x3, y3, w3, h3) in body3:
            tx = int(x3 + 0.2 * w3)
            ty = int(y3 + 0.2 * h3)
            tw = int(0.6 * w3)
            th = int(0.6 * h3)
            if not in_roi(tx, ty, tw, th, body):
                body.append((x3, y3 - h3, w3, 2 * h3))
                # cv2.rectangle(img, (tx, ty), (tx + tw, ty + th), (255, 255, 255), 2)

        for (px, py, pw, ph) in body:
            if not in_roi(px, py, pw, ph, body):
                person1.append((px, py, pw, ph))
                SD += 1

        for (px1, py1, pw1, ph1) in person1:
            cv2.rectangle(roi_color, (px, py), (px + pw, py + ph), (255, 0, 0), 2)
            person_count += 1

        if person_count > 1:
            #cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)
            SD = SD - person_count
            SDV += person_count
        elif 0.75 < w / h < 1.25:
            #cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 255), 2)
            #SD = SD - 2
            PSDV += 2
        #else:
            #cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame1, "SD followed: {}".format(SD), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 3)
    cv2.putText(frame1, "Potential SD Violation: {}".format(PSDV), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 255), 3)
    cv2.putText(frame1, "SD Violation: {}".format(SDV), (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 255), 3)


    image = cv2.resize(frame1, (1200, 800))
    out.write(image)
    cv2.imshow("View", image)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
