# -*- coding: utf-8 -*-
"""
Created on Thu May 25 04:07:14 2020
CSE 30 Spring 2020 Program 4 starter code
@author: Fahim
"""

import cv2


def in_roi(ex1, ey1, ew1, eh1, roi_list1):
    for (fx, fy, fw, fh) in roi_list1:
        if fx < ex1 < (fx + fw) and fy < ey1 < (fy + fh) and \
                fx < (ex1 + ew1) < (fx + fw) and fy < (ey1 + eh1) < (fy + fh):
            return True


# Download the required files/sample videos here : https://users.soe.ucsc.edu/~pang/30/s20/prog4/data/
cascade1 = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cascade2 = cv2.CascadeClassifier('haarcascade_upperbody.xml')
cascade3 = cv2.CascadeClassifier('haarcascade_lowerbody.xml')

cap = cv2.VideoCapture("Sample.webm")
# cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output3.avi', fourcc, 20.0, (640, 480))

cf = 0
cq = [0]*40

while cap.isOpened():
    cf += 1
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    c1 = 0
    c2 = 0
    c3 = 0
    c = 0

    body = []
    body1 = cascade3.detectMultiScale(gray, 1.3, 5)
    #body2 = cascade2.detectMultiScale(gray, 1.3, 5)
    #body3 = cascade3.detectMultiScale(gray, 1.3, 5)

    for (x1, y1, w1, h1) in body1:
        c1 += 1
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        body.append((x1, y1, w1, h1))

    '''for (x2, y2, w2, h2) in body2:
        c2 += 1
        tx = int(x2+0.2*w2)
        ty = int(y2+0.2*h2)
        tw = int(0.6*w2)
        th = int(0.6*h2)
        if not in_roi(tx, ty, tw, th, body):
            body.append((x2, y2, w2, 3 * h2))
            #cv2.rectangle(img, (tx, ty), (tx + tw, ty + th), (255, 255, 0), 2)

    for (x3, y3, w3, h3) in body3:
        c3 += 1
        tx = int(x3 + 0.2 * w3)
        ty = int(y3 + 0.2 * h3)
        tw = int(0.6 * w3)
        th = int(0.6 * h3)
        if not in_roi(tx, ty, tw, th, body):
            body.append((x3, y3 - h3, w3, 2 * h3))
            #cv2.rectangle(img, (tx, ty), (tx + tw, ty + th), (255, 255, 255), 2)'''

    people = []

    for (bx, by, bw, bh) in body:
        if not in_roi(bx, by, bw, bh, body):
            people.append((bx, by, bw, bh))

    for (x, y, w, h) in people:
        c += 1
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cq.pop(0)
    cq.append(c)
    pc = max(cq)
    print("frame, fullbody, upperbody, lowerbody, body: ", cf, c1, c2, c3, c)

    cv2.putText(img, "Lowerbody Counted: {}".format(c), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 3)
    image = cv2.resize(img, (640, 480))
    out.write(image)
    cv2.imshow('View', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
