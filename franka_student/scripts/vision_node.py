#!/usr/bin/env python3
"""
vision_node.py - Franka FR3 Vision Node
Elabora il flusso video della camera RGB per produrre maschere binarie
degli oggetti target (colori HSV e marker ArUco).

Le maschere vengono pubblicate come immagini mono8 su topic dedicati
e consumate dal pick_node per la localizzazione dei cubi.

Note sui parametri HSV calibrati:
  - Rosso:     hue [0-10], S >= 100, V >= 50
  - Arancione: hue [7-24], S >= 90,  V >= 80
               Gap hue 0-6 lasciato vuoto per ridurre ambiguità con il rosso.
               Filtro XOR finale: pixel ambigui assegnati all'arancione.
  - Blu:       hue [100-130], S >= 100, V >= 40
  - Verde:     hue [40-80],   S >= 100, V >= 40

ROI fissa (percentuale sull'immagine):
  top=30%, bottom=89%, left=28%, right=90%
"""

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from cv_bridge import CvBridge, CvBridgeError
import cv2
import cv2.aruco as aruco
import numpy as np


class VisionNode:
    """
    Nodo ROS per la percezione visiva.
    Pubblica maschere binarie per quattro colori (rosso, arancione, blu, verde)
    e una maschera sintetica per la presa dei marker ArUco.
    """

    def __init__(self):
        rospy.init_node('vision_node', anonymous=True)
        self.bridge = CvBridge()

        self.latest_frame = None
        self.current_header = None

        # ------------------------------------------------------------------
        # Parametri ROI fissi (percentuale sulla dimensione dell'immagine)
        # ------------------------------------------------------------------
        self.roi_top = 21
        self.roi_bottom = 81
        self.roi_left = 21
        self.roi_right = 81

        self.hsv_bounds = {
            'red': (np.array([0, 113, 116]), np.array([20, 255, 255])),
            'orange': (np.array([7, 90, 80]), np.array([24, 255, 255])),
            'blue': (np.array([100, 100, 40]), np.array([130, 255, 255])),
            'green': (np.array([40, 100, 40]), np.array([80, 255, 255])),
        }

        # ------------------------------------------------------------------
        # Publisher delle maschere colore (mono8)
        # ------------------------------------------------------------------
        self.pub_mask_red    = rospy.Publisher('/perception/HSV_red',    Image, queue_size=10)
        self.pub_mask_orange = rospy.Publisher('/perception/HSV_orange', Image, queue_size=10)
        self.pub_mask_blue   = rospy.Publisher('/perception/HSV_blue',   Image, queue_size=10)
        self.pub_mask_green  = rospy.Publisher('/perception/HSV_green',  Image, queue_size=10)

        # Publisher delle maschere sintetiche per la presa dei marker ArUco,
        # separate per parita' dell'ID: il control_node decide a runtime
        # se in gara vanno presi i pari o i dispari, e pick_node si iscrive
        # al topic corrispondente (mask_aruco_even / mask_aruco_odd).
        # Contengono un poligono bianco riempito sull'area del marker rilevato,
        # così il pick_node può trattarlo come un qualunque oggetto.
        self.pub_mask_aruco_even = rospy.Publisher('/perception/mask_aruco_even', Image, queue_size=10)
        self.pub_mask_aruco_odd  = rospy.Publisher('/perception/mask_aruco_odd',  Image, queue_size=10)

        # Publisher di debug: immagine con overlay dei marker + ID numerico
        self.pub_aruco_img = rospy.Publisher('/perception/aruco',    Image, queue_size=10)
        self.pub_aruco_id  = rospy.Publisher('/perception/aruco_id', Int16, queue_size=10)

        # Publisher di debug: immagine raw con rettangolo ROI sovrapposto
        self.pub_debug_roi = rospy.Publisher('/perception/debug_roi', Image, queue_size=10)

        self.sub_cam = rospy.Subscriber("/camera/color/image_raw", Image, self.camera_callback)
        rospy.loginfo("[VISION] Nodo avviato. Rilevamento colori e ArUco attivo.")

    # ------------------------------------------------------------------
    # Callback camera
    # ------------------------------------------------------------------

    def camera_callback(self, data):
        """
        Callback principale. Elabora ogni frame della camera e pubblica
        le maschere aggiornate per tutti i target.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError:
            return

        h, w, _ = cv_image.shape

        # ----------------------------------------------------------
        # Applicazione ROI fissa
        # ----------------------------------------------------------
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        punti_tavolo = np.array([
            [int(w * (self.roi_left   / 100.0)), int(h * (self.roi_top    / 100.0))],
            [int(w * (self.roi_right  / 100.0)), int(h * (self.roi_top    / 100.0))],
            [int(w * (self.roi_right  / 100.0)), int(h * (self.roi_bottom / 100.0))],
            [int(w * (self.roi_left   / 100.0)), int(h * (self.roi_bottom / 100.0))],
        ], dtype=np.int32)
        cv2.fillPoly(roi_mask, [punti_tavolo], 255)
        cv_image_roi = cv2.bitwise_and(cv_image, cv_image, mask=roi_mask)

        # Pubblica immagine di debug con il rettangolo ROI sovrapposto
        debug_roi = cv_image.copy()
        cv2.polylines(debug_roi, [punti_tavolo], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(debug_roi, "ROI", (punti_tavolo[0][0] + 5, punti_tavolo[0][1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        try:
            ros_debug = self.bridge.cv2_to_imgmsg(debug_roi, "bgr8")
            ros_debug.header = data.header
            self.pub_debug_roi.publish(ros_debug)
        except Exception:
            pass

        # ----------------------------------------------------------
        # Rilevamento ArUco e generazione maschera sintetica
        # ----------------------------------------------------------
        gray          = cv2.cvtColor(cv_image_roi, cv2.COLOR_BGR2GRAY)
        aruco_display = cv_image_roi.copy()
        mask_aruco_even = np.zeros((h, w), dtype=np.uint8)
        mask_aruco_odd  = np.zeros((h, w), dtype=np.uint8)

        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
        detector   = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(aruco_display, corners, ids)
            for i, marker_id in enumerate(ids):
                mid = int(marker_id[0])
                self.pub_aruco_id.publish(mid)
                if mid % 2 == 0:
                    cv2.fillConvexPoly(mask_aruco_even, np.int32(corners[i][0]), 255)
                else:
                    cv2.fillConvexPoly(mask_aruco_odd, np.int32(corners[i][0]), 255)

        try:
            ros_aruco_img = self.bridge.cv2_to_imgmsg(aruco_display, "bgr8")
            ros_aruco_img.header = data.header
            self.pub_aruco_img.publish(ros_aruco_img)
        except Exception:
            pass

        self.publish_mask(mask_aruco_even, self.pub_mask_aruco_even, data.header)
        self.publish_mask(mask_aruco_odd,  self.pub_mask_aruco_odd,  data.header)

        # ----------------------------------------------------------
        # Elaborazione maschere colore
        # ----------------------------------------------------------
        MAX_AREA_CUBE = 14000

        hsv = cv2.cvtColor(cv_image_roi, cv2.COLOR_BGR2HSV)

        mask_red_final    = self.process_color_mask(
            hsv,
            self.hsv_bounds['red'][0], self.hsv_bounds['red'][1],
            min_area=400, max_area=MAX_AREA_CUBE
        )
        mask_orange_raw   = self.process_color_mask(
            hsv,
            self.hsv_bounds['orange'][0], self.hsv_bounds['orange'][1],
            min_area=400, max_area=MAX_AREA_CUBE
        )

        # Risoluzione ambiguità rosso/arancione:
        # i pixel presenti in entrambe le maschere vengono assegnati all'arancione.
        pixel_ambigui  = cv2.bitwise_and(mask_red_final, mask_orange_raw)
        mask_red_final = cv2.bitwise_and(mask_red_final, cv2.bitwise_not(pixel_ambigui))
        mask_orange_final = mask_orange_raw

        mask_blue_final  = self.process_color_mask(
            hsv,
            self.hsv_bounds['blue'][0], self.hsv_bounds['blue'][1],
            max_area=MAX_AREA_CUBE
        )
        mask_green_final = self.process_color_mask(
            hsv,
            self.hsv_bounds['green'][0], self.hsv_bounds['green'][1]
        )

        self.publish_mask(mask_red_final,    self.pub_mask_red,    data.header)
        self.publish_mask(mask_orange_final, self.pub_mask_orange, data.header)
        self.publish_mask(mask_blue_final,   self.pub_mask_blue,   data.header)
        self.publish_mask(mask_green_final,  self.pub_mask_green,  data.header)

    # ------------------------------------------------------------------
    # Elaborazione maschere colore
    # ------------------------------------------------------------------

    def process_color_mask(self, hsv_image, lower_bound, upper_bound,
                           min_area=400, max_area=14000):
        """
        Genera una maschera binaria pulita per un dato range HSV.

        Pipeline:
          1. Threshold HSV
          2. Chiusura morfologica (riempie piccoli buchi interni)
          3. Apertura morfologica (rimuove piccoli artefatti di bordo)
          4. Filtraggio per area: mantiene solo i contorni compresi
             tra min_area e max_area

        Args:
            hsv_image:   immagine sorgente nello spazio HSV
            lower_bound: limite inferiore del range HSV (np.array)
            upper_bound: limite superiore del range HSV (np.array)
            min_area:    area minima in pixel per mantenere un contorno
            max_area:    area massima in pixel; contorni più grandi vengono
                         scartati (appartengono a elementi fissi della scena)

        Returns:
            maschera binaria (uint8, 0/255)
        """
        mask_raw    = cv2.inRange(hsv_image, lower_bound, upper_bound)
        kernel      = np.ones((5, 5), np.uint8)
        mask_closed = cv2.morphologyEx(mask_raw,    cv2.MORPH_CLOSE, kernel)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_clean  = np.zeros_like(mask_opened)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                cv2.drawContours(mask_clean, [cnt], -1, 255, thickness=cv2.FILLED)

        return mask_clean

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def publish_mask(self, cv_mask, publisher, header):
        """Converte una maschera OpenCV in un messaggio ROS mono8 e la pubblica."""
        try:
            ros_mask = self.bridge.cv2_to_imgmsg(cv_mask, "mono8")
            ros_mask.header = header
            publisher.publish(ros_mask)
        except Exception:
            pass


if __name__ == '__main__':
    try:
        VisionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass