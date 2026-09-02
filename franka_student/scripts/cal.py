#!/usr/bin/env python3
"""
calibration_node.py
"""

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from cv_bridge import CvBridge, CvBridgeError
import cv2
import cv2.aruco as aruco
import numpy as np


class VisionNode:
    def __init__(self):
        rospy.init_node('vision_node', anonymous=True)
        self.bridge = CvBridge()

        self.latest_frame = None
        self.current_header = None

        # Stato iniziale
        self.state = "ROI"

        # ---------------------------------------------------------------
        # Parametri fissi
        # ---------------------------------------------------------------
        self.roi_top = 28
        self.roi_bottom = 78
        self.roi_left = 15
        self.roi_right = 72

        self.color_list = ['red', 'orange', 'blue', 'green']
        self.hsv_bounds = {
            'red': (np.array([0, 120, 40]), np.array([10, 255, 255])),
            'orange': (np.array([7, 90, 80]), np.array([24, 255, 255])),
            'blue': (np.array([100, 100, 40]), np.array([130, 255, 255])),
            'green': (np.array([40, 100, 40]), np.array([80, 255, 255]))
        }
        # ---------------------------------------------------------------

        self.last_color_idx = -1

        self.pub_mask_red = rospy.Publisher('/perception/HSV_red', Image, queue_size=10)
        self.pub_mask_orange = rospy.Publisher('/perception/HSV_orange', Image, queue_size=10)
        self.pub_mask_blue = rospy.Publisher('/perception/HSV_blue', Image, queue_size=10)
        self.pub_mask_green = rospy.Publisher('/perception/HSV_green', Image, queue_size=10)

        self.pub_aruco_img = rospy.Publisher('/perception/aruco', Image, queue_size=10)
        self.pub_aruco_id = rospy.Publisher('/perception/aruco_id', Int16, queue_size=10)
        self.pub_mask_aruco = rospy.Publisher('/perception/mask_aruco', Image, queue_size=10)

        self.sub_cam = rospy.Subscriber("/camera/color/image_raw", Image, self.camera_callback)
        rospy.loginfo("[VISION] Nodo avviato.")

    def camera_callback(self, data):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.current_header = data.header
        except CvBridgeError:
            return

    def stampare_parametri_fissi(self):
        print("\n" + "=" * 65)
        print("  PARAMETRI CALIBRATI COMPLETI - COPIA E INCOLLA NEL CODICE")
        print("=" * 65)
        print(f"self.roi_top = {self.roi_top}")
        print(f"self.roi_bottom = {self.roi_bottom}")
        print(f"self.roi_left = {self.roi_left}")
        print(f"self.roi_right = {self.roi_right}")
        print("\nself.hsv_bounds = {")
        for col in self.color_list:
            low, up = self.hsv_bounds[col]
            print(f"    '{col}': (np.array([{low[0]}, {low[1]}, {low[2]}]), np.array([{up[0]}, {up[1]}, {up[2]}])),")
        print("}")
        print("=" * 65 + "\n")

    def process_color_mask(self, hsv_image, lower_bound, upper_bound, min_area=400):
        mask_raw = cv2.inRange(hsv_image, lower_bound, upper_bound)
        kernel = np.ones((9, 9), np.uint8)
        mask_closed = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_clean = np.zeros_like(mask_opened)
        for cnt in contours:
            if min_area < cv2.contourArea(cnt) < 14000:
                cv2.drawContours(mask_clean, [cnt], -1, 255, thickness=cv2.FILLED)
        return mask_clean

    def publish_mask(self, mask_img, publisher, header):
        try:
            ros_img = self.bridge.cv2_to_imgmsg(mask_img, "mono8")
            ros_img.header = header
            publisher.publish(ros_img)
        except Exception:
            pass

    def spin_processing_loop(self):


        cv2.namedWindow("Calibrazione ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibrazione ROI", 750, 550)
        cv2.createTrackbar("ROI Alto (%)", "Calibrazione ROI", self.roi_top, 100, lambda x: None)
        cv2.createTrackbar("ROI Basso (%)", "Calibrazione ROI", self.roi_bottom, 100, lambda x: None)
        cv2.createTrackbar("ROI Sinistro (%)", "Calibrazione ROI", self.roi_left, 100, lambda x: None)
        cv2.createTrackbar("ROI Destro (%)", "Calibrazione ROI", self.roi_right, 100, lambda x: None)
        initialized_colors_win = False
        initialized_monitor_win = False

        while not rospy.is_shutdown():
            if self.latest_frame is None:
                rospy.logwarn_throttle(2, "[ATTESA] In attesa dei frame dalla telecamera...")
                rospy.sleep(0.1)
                continue

            cv_image = self.latest_frame.copy()
            header = self.current_header
            h, w, _ = cv_image.shape


            if self.state in ["ROI", "ASK_COLORS"]:
                self.roi_top = cv2.getTrackbarPos("ROI Alto (%)", "Calibrazione ROI")
                self.roi_bottom = cv2.getTrackbarPos("ROI Basso (%)", "Calibrazione ROI")
                self.roi_left = cv2.getTrackbarPos("ROI Sinistro (%)", "Calibrazione ROI")
                self.roi_right = cv2.getTrackbarPos("ROI Destro (%)", "Calibrazione ROI")


            # ---------------------------------------------------------------
            # Core del nodo
            # ---------------------------------------------------------------
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            punti_tavolo = np.array([
                [int(w * (self.roi_left / 100.0)), int(h * (self.roi_top / 100.0))],
                [int(w * (self.roi_right / 100.0)), int(h * (self.roi_top / 100.0))],
                [int(w * (self.roi_right / 100.0)), int(h * (self.roi_bottom / 100.0))],
                [int(w * (self.roi_left / 100.0)), int(h * (self.roi_bottom / 100.0))]
            ], dtype=np.int32)
            cv2.fillPoly(roi_mask, [punti_tavolo], 255)
            cv_image_roi = cv2.bitwise_and(cv_image, cv_image, mask=roi_mask)

            aruco_display = cv_image_roi.copy()

            gray = cv2.cvtColor(cv_image_roi, cv2.COLOR_BGR2GRAY)
            mask_aruco = np.zeros((h, w), dtype=np.uint8)
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
            detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None:
                aruco.drawDetectedMarkers(aruco_display, corners, ids)
                for i, marker_id in enumerate(ids):
                    self.pub_aruco_id.publish(int(marker_id[0]))
                    cv2.fillConvexPoly(mask_aruco, np.int32(corners[i][0]), 255)

            try:
                ros_aruco = self.bridge.cv2_to_imgmsg(aruco_display, "bgr8")
                ros_aruco.header = header
                self.pub_aruco_img.publish(ros_aruco)
            except Exception:
                pass

            self.publish_mask(mask_aruco, self.pub_mask_aruco, header)

            hsv = cv2.cvtColor(cv_image_roi, cv2.COLOR_BGR2HSV)
            mask_red_final = self.process_color_mask(hsv, self.hsv_bounds['red'][0], self.hsv_bounds['red'][1])
            mask_orange_raw = self.process_color_mask(hsv, self.hsv_bounds['orange'][0], self.hsv_bounds['orange'][1])
            pixel_ambigui = cv2.bitwise_and(mask_red_final, mask_orange_raw)
            mask_red_final = cv2.bitwise_and(mask_red_final, cv2.bitwise_not(pixel_ambigui))
            mask_orange_final = mask_orange_raw
            mask_blue_final = self.process_color_mask(hsv, self.hsv_bounds['blue'][0], self.hsv_bounds['blue'][1])
            mask_green_final = self.process_color_mask(hsv, self.hsv_bounds['green'][0], self.hsv_bounds['green'][1])

            self.publish_mask(mask_red_final, self.pub_mask_red, header)
            self.publish_mask(mask_orange_final, self.pub_mask_orange, header)
            self.publish_mask(mask_blue_final, self.pub_mask_blue, header)
            self.publish_mask(mask_green_final, self.pub_mask_green, header)
            # ---------------------------------------------------------------
            # Fine core del nodo
            # ---------------------------------------------------------------

            # Logica di visualizzazione a schermo e lettura tasti
            screen_img = cv_image.copy()
            cv2.polylines(screen_img, [punti_tavolo], True, (0, 0, 255), 2)

            if self.state == "ROI":
                cv2.putText(screen_img, "FASE 1: REGOLA LA ROI", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255),
                            2)
                cv2.imshow("Calibrazione ROI", screen_img)

            elif self.state == "ASK_COLORS":
                cv2.putText(screen_img, "FASE 2: I COLORI VANNO BENE? (Y/N)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 100, 0), 2)
                y_offset = 80
                for col in self.color_list:
                    low, up = self.hsv_bounds[col]
                    cv2.putText(screen_img,
                                f"- {col.upper()}: H:[{low[0]}-{up[0]}] S:[{low[1]}-{up[1]}] V:[{low[2]}-{up[2]}]",
                                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    y_offset += 22
                cv2.imshow("Calibrazione ROI", screen_img)

            elif self.state == "CALIBRATE_COLORS":
                if not initialized_colors_win:
                    cv2.namedWindow("Calibrazione Colori", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Calibrazione Colori", 750, 550)
                    cv2.createTrackbar("Colore (0:R, 1:O, 2:B, 3:G)", "Calibrazione Colori", 0, 3, lambda x: None)
                    cv2.createTrackbar("H Min", "Calibrazione Colori", 0, 180, lambda x: None)
                    cv2.createTrackbar("H Max", "Calibrazione Colori", 180, 180, lambda x: None)
                    cv2.createTrackbar("S Min", "Calibrazione Colori", 0, 255, lambda x: None)
                    cv2.createTrackbar("S Max", "Calibrazione Colori", 255, 255, lambda x: None)
                    cv2.createTrackbar("V Min", "Calibrazione Colori", 0, 255, lambda x: None)
                    cv2.createTrackbar("V Max", "Calibrazione Colori", 255, 255, lambda x: None)
                    initialized_colors_win = True

                color_idx = cv2.getTrackbarPos("Colore (0:R, 1:O, 2:B, 3:G)", "Calibrazione Colori")
                color_name = self.color_list[color_idx]

                if color_idx != self.last_color_idx:
                    self.last_color_idx = color_idx
                    low, up = self.hsv_bounds[color_name]
                    cv2.setTrackbarPos("H Min", "Calibrazione Colori", low[0])
                    cv2.setTrackbarPos("H Max", "Calibrazione Colori", up[0])
                    cv2.setTrackbarPos("S Min", "Calibrazione Colori", low[1])
                    cv2.setTrackbarPos("S Max", "Calibrazione Colori", up[1])
                    cv2.setTrackbarPos("V Min", "Calibrazione Colori", low[2])
                    cv2.setTrackbarPos("V Max", "Calibrazione Colori", up[2])
                else:
                    h_min = cv2.getTrackbarPos("H Min", "Calibrazione Colori")
                    h_max = cv2.getTrackbarPos("H Max", "Calibrazione Colori")
                    s_min = cv2.getTrackbarPos("S Min", "Calibrazione Colori")
                    s_max = cv2.getTrackbarPos("S Max", "Calibrazione Colori")
                    v_min = cv2.getTrackbarPos("V Min", "Calibrazione Colori")
                    v_max = cv2.getTrackbarPos("V Max", "Calibrazione Colori")
                    self.hsv_bounds[color_name] = (np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))

                current_mask = mask_red_final
                if color_name == 'orange':
                    current_mask = mask_orange_final
                elif color_name == 'blue':
                    current_mask = mask_blue_final
                elif color_name == 'green':
                    current_mask = mask_green_final

                monitor_colori = cv2.bitwise_and(cv_image_roi, cv_image_roi, mask=current_mask)
                if ids is not None:
                    aruco.drawDetectedMarkers(monitor_colori, corners, ids)

                cv2.putText(monitor_colori, f"CALIBRAZIONE: {color_name.upper()}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)
                cv2.imshow("Calibrazione Colori", monitor_colori)

            elif self.state == "RUNNING":
                if not initialized_monitor_win:
                    cv2.namedWindow("Monitor Principale", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Monitor Principale", 700, 550)
                    initialized_monitor_win = True
                cv2.imshow("Monitor Principale", screen_img)

            raw_key = cv2.waitKey(30)
            key = raw_key & 0xFF
            tasti_conferma = [32, 13, ord('c'), ord('C')]
            tasti_si = [ord('y'), ord('Y'), 121, 89]
            tasti_no = [ord('n'), ord('N'), 110, 78]

            if self.state == "ROI" and key in tasti_conferma:
                self.state = "ASK_COLORS"
            elif self.state == "ASK_COLORS":
                if key in tasti_si:
                    cv2.destroyWindow("Calibrazione ROI")
                    cv2.waitKey(10)
                    self.stampare_parametri_fissi()
                    self.state = "RUNNING"
                elif key in tasti_no:
                    cv2.destroyWindow("Calibrazione ROI")
                    cv2.waitKey(10)
                    self.state = "CALIBRATE_COLORS"
            elif self.state == "CALIBRATE_COLORS" and key in tasti_conferma:
                cv2.destroyWindow("Calibrazione Colori")
                cv2.waitKey(10)
                self.stampare_parametri_fissi()
                self.state = "RUNNING"

        cv2.destroyAllWindows()



if __name__ == '__main__':
    try:
        node = VisionNode()
        node.spin_processing_loop()
    except rospy.ROSInterruptException:
        pass