#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np

class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 2
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture("/home/fizzer/Downloads/Leopard.mp4")
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)
        
        self.sift = cv2.SIFT_create()


    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        pixmap = QtGui.QPixmap(self.template_path)
        pixmap = pixmap.scaled(320, 240, QtCore.Qt.KeepAspectRatio)

        self.template_label.setPixmap(pixmap)
        self.grayframe =cv2.cvtColor(cv2.imread(self.template_path), cv2.COLOR_BGR2GRAY)
        self.template_h, self.template_w = self.grayframe.shape
        self.index_params=dict(algorithm=0, trees=5)
        self.search_params=dict()
        self.flann=cv2.FlannBasedMatcher(self.index_params, self.search_params)

        self.kp_image, self.desc_image = self.sift.detectAndCompute(self.grayframe, None)
        print("Loaded template image file: " + self.template_path)

    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height,
                             bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        ret, frame = self._camera_device.read()
        # TODO run SIFT on the captured frame
        if not ret:
            print("Error: cannot read frame from camera")
            return
        grayframe_v = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_grayframe, desc_grayframe = self.sift.detectAndCompute(grayframe_v, None)
        

        self.sift.detectAndCompute(frame, None)
        matches = self.flann.knnMatch(self.desc_image, desc_grayframe, k=2)
        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        query_pts = np.float32([self.kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        
        if len(good_points) > 35:
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()
            h, w = grayframe_v.shape
            pts = np.float32([[0, 0], [0, self.template_h - 1], [self.template_w - 1, self.template_h - 1], [self.template_w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            frame = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
        else:
            print("Not enough matches are found - {}/10".format(len(good_points)))
            matches_mask = None

            match_vis = cv2.drawMatches(self.grayframe, self.kp_image, grayframe_v, kp_grayframe, good_points, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            frame = match_vis

        

        pixmap = self.convert_cv_to_pixmap(frame)
        pixmap = pixmap.scaled(320, 240, QtCore.Qt.KeepAspectRatio)
        self.live_image_label.setPixmap(pixmap)

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")

            

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())

