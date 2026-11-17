#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""extract_frame.py <clip> <frame_idx> <out.jpg>"""
import sys
import cv2
cap = cv2.VideoCapture(sys.argv[1])
cap.set(cv2.CAP_PROP_POS_FRAMES, int(sys.argv[2]))
ok, f = cap.read()
cap.release()
if ok:
    cv2.imwrite(sys.argv[3], f)
    print("saved", sys.argv[3], f.shape)
else:
    print("failed")
