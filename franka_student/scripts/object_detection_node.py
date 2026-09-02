#!/usr/bin/env python3
"""
object_detection_node.py - Riconoscimento Oggetti COCO Dataset con Filtro Temporale (Specifica Esame 6 CFU)

Nodo di object detection generico basato su YOLOv8 (pesi pre-addestrati sul
dataset COCO), usato come nodo indipendente dalla pipeline di pick-and-place
principale (vision_node / pick_node / place_node / control_node).

Applica un filtro temporale a finestra mobile sul tracker di YOLO: invece di
fidarsi della singola inferenza per frame (soggetta a "flickering" di classe
e confidenza), accumula lo storico delle ultime N predizioni per ciascun
oggetto tracciato (stesso box.id) e pubblica solo quando la classe piu'
votata supera una soglia di confidenza media.

Publisher:
  - /perception/object_detection  (Image)  - frame annotato con bounding box
  - /perception/object_class      (String) - stringa "classe (confidenza)",
                                              una per ogni oggetto stabile nel frame

Subscriber:
  - /camera/color/image_raw (Image) - stream RGB della camera
"""

import sys
import os
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
from collections import deque

# Importiamo YOLOv8 (pre-addestrato su COCO dataset)
try:
    from ultralytics import YOLO
except ImportError:
    rospy.logerr("Libreria 'ultralytics' non trovata! Installa con: pip install ultralytics")
    sys.exit(1)


class ObjectDetectionNode:
    """
    Nodo ROS per il riconoscimento di oggetti generici (classi COCO) tramite
    YOLOv8 con tracking persistente e filtro temporale a finestra mobile,
    per stabilizzare classe e confidenza pubblicate nel tempo.
    """

    def __init__(self):
        rospy.init_node('object_detection_node', anonymous=True)
        self.bridge = CvBridge()

        # Ottiene il percorso della cartella in cui si trova questo script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Unisce il percorso della cartella al nome del file del modello
        model_path = os.path.join(script_dir, 'yolov8l.pt')

        rospy.loginfo(f"Caricamento modello YOLO da: {model_path}")
        self.model = YOLO(model_path)
        # --------------------

        self.pub_detection_img = rospy.Publisher('/perception/object_detection', Image, queue_size=10)
        self.pub_object_class = rospy.Publisher('/perception/object_class', String, queue_size=10)

        # --- PARAMETRI DI MEMORIA TEMPORALE ---
        # object_history: mappa obj_id (assegnato dal tracker di YOLO) -> deque
        # delle ultime `memory_window` coppie (classe, confidenza) osservate.
        # Serve a smorzare il flickering di classe/confidenza tra un frame e
        # l'altro tramite votazione + media, invece di fidarsi del singolo frame.
        self.memory_window = 10  # Numero di fotogrammi consecutivi da ricordare
        self.object_history = {}  # Memoria per ogni oggetto tracciato
        self.final_threshold = 0.70  # Soglia di sicurezza sulla media per pubblicare l'oggetto

        # Sottoscrizione allo stream video della telecamera
        self.sub_cam = rospy.Subscriber("/camera/color/image_raw", Image, self.camera_callback)
        rospy.loginfo("NODO OBJECT DETECTION (CON FILTRO TEMPORALE) PRONTO! In ascolto sulla camera...")

    def camera_callback(self, data):
        """
        Esegue l'inferenza YOLO con tracking persistente sul frame ricevuto,
        applica il filtro temporale per ogni oggetto tracciato e pubblica:
          - il frame annotato con i bounding box degli oggetti stabili;
          - la stringa "classe (confidenza)" per ciascun oggetto che ha
            superato la soglia `final_threshold` sulla confidenza mediata.
        """
        try:
            # Convertiamo il messaggio ROS Image in un frame OpenCV BGR
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("Errore CvBridge: %s", str(e))
            return

        # Eseguiamo l'inferenza usando il TRACKER invece del modello base
        # Abbassiamo conf a 0.25 per registrare i frame incerti e mediarli
        results = self.model.track(cv_image, persist=True, conf=0.25, verbose=False)

        annotated_frame = cv_image.copy()
        detected_objects_strings = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Se l'oggetto non ha ancora un ID stabile, lo ignoriamo momentaneamente
                if box.id is None:
                    continue

                obj_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                raw_class_name = self.model.names[class_id]

                # --- 1. AGGIORNO LA MEMORIA ---
                if obj_id not in self.object_history:
                    self.object_history[obj_id] = deque(maxlen=self.memory_window)
                self.object_history[obj_id].append((raw_class_name, confidence))

                # --- 2. CALCOLO DELLA MEDIA ---
                history = self.object_history[obj_id]

                # Sistema a votazione per la classe (per evitare "flickering" dei nomi)
                class_votes = {}
                for c, conf in history:
                    class_votes[c] = class_votes.get(c, 0) + 1
                smoothed_class = max(class_votes, key=class_votes.get)

                # Calcolo della confidenza media solo per la classe più votata
                confidences_for_class = [conf for c, conf in history if c == smoothed_class]
                smoothed_conf = sum(confidences_for_class) / len(confidences_for_class)

                # --- 3. APPLICAZIONE SOGLIA FINALE ---
                if smoothed_conf < self.final_threshold:
                    continue  # Se la media non supera il 70%, non mostro l'oggetto

                # DISEGNA BOUNDING BOX (Rettangolo Verde)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # CONFIGURA L'ETICHETTA DI CLASSE E CONFIDENZA MEDIATA
                img_label = f"{smoothed_class} {smoothed_conf:.2f}"
                cv2.putText(annotated_frame, img_label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                exam_string = f"{smoothed_class} ({smoothed_conf:.2f})"
                detected_objects_strings.append(exam_string)

        # --- PUBBLICAZIONE SUL TOPIC STRINGA ---
        if detected_objects_strings:
            msg_string = ", ".join(detected_objects_strings)
            self.pub_object_class.publish(msg_string)

        # --- PUBBLICAZIONE SUL TOPIC IMMAGINE ---
        try:
            ros_img = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            ros_img.header = data.header  # Manteniamo intatto il timestamp originario
            self.pub_detection_img.publish(ros_img)
        except CvBridgeError as e:
            rospy.logerr("Errore pubblicazione immagine annotata: %s", str(e))


if __name__ == '__main__':
    try:
        ObjectDetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass