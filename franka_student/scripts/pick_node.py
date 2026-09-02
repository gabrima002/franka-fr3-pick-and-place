#!/usr/bin/env python3

import sys
import copy
import numpy as np
import rospy
import moveit_commander
import tf2_ros
import tf2_geometry_msgs
import tf.transformations as tf_trans
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import actionlib
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from image_geometry import PinholeCameraModel
import cv2
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal

# ------------------------------------------------------------
# Parametri geometrici della scena (in metri)
# ------------------------------------------------------------
Z_TABLE_WORLD      = -0.14   # Altezza del piano del tavolo rispetto a fr3_link0 (-0.14 nel lab)
FINGER_TIP_OFFSET  = 0.100   # Offset fingertip rispetto al frame EEF
CONTACT_MARGIN     = 0.00    # Margine aggiuntivo di contatto (0 = nessun offset)
APPROACH_DIST      = 0.10    # Distanza di avvicinamento verticale pre-presa

# Stima conservativa dell'altezza dell'oggetto per il calcolo della Z di presa
# e per il bounding box nella scena MoveIt.
# Non dipende dalla larghezza: il gripper si adatta tramite la GraspAction.
# Abbassare per oggetti piatti (es. cilindri bassi); alzare per oggetti alti.
OBJECT_HEIGHT_ESTIMATE = 0.05   # [m] - valore nominale per oggetti ~5 cm

# ------------------------------------------------------------
# Parametri gripper
# ------------------------------------------------------------
GRIPPER_OPEN          = 0.04   # Posizione di apertura per dito (simulazione)
GRIPPER_ACTION_SERVER = "/franka_gripper/gripper_action"
GRIPPER_TIMEOUT       = 5.0    # Timeout per le action del gripper (s)
GRIPPER_FORCE         = 30.0   # Forza di presa in Newton (robot reale)

# Parametri GraspAction geometry-agnostic.
#
# Strategia: width quasi-zero fa si' che le dita cerchino di chiudersi
# completamente. Si fermano sull'oggetto indipendentemente dalla sua
# larghezza (cubo, parallelepipedo, cilindro). La forza costante GRIPPER_FORCE
# mantiene la presa durante il trasporto.
#
# epsilon.inner stretto: se le dita arrivano davvero a zero e' presa a vuoto.
# epsilon.outer largo (80 mm): accetta qualunque oggetto entro l'apertura
# massima del gripper FR3 (85 mm) senza harcodare dimensioni geometriche.
#
# Il controllo sulla presa a vuoto e' delegato interamente al check
# post-chiusura sul valore del giunto (GRASP_EMPTY_THRESHOLD).
GRASP_WIDTH           = 0.005   # Target di chiusura quasi-zero [m]
GRASP_EPSILON_INNER   = 0.005   # Tolleranza interna: presa a vuoto se raggiunta
GRASP_EPSILON_OUTER   = 0.080   # Tolleranza esterna: accetta oggetti fino a ~80 mm
GRASP_SPEED           = 0.03    # Velocita' di chiusura [m/s] - lenta per stabilita'
GRASP_EMPTY_THRESHOLD = 0.010   # Soglia minima giunto post-presa [m]: sotto = a vuoto

# ------------------------------------------------------------
# Parametri ROS / TF
# ------------------------------------------------------------
OBJECT_ID            = "target_object"   # Nome generico dell'oggetto nella scena MoveIt
CAMERA_INFO_TOPIC    = "/camera/color/camera_info"
CAMERA_FRAME         = "camera_color_optical_frame"
ROBOT_BASE_FRAME     = "fr3_link0"
Z_RAYCAST_PLANE      = Z_TABLE_WORLD
MAX_CAMERA_TILT_FOR_ANGLE = np.radians(25.0)  # Oltre questa inclinazione non si stima l'angolo

# ------------------------------------------------------------
# Posizione HOME "finale" di fine missione
# ------------------------------------------------------------
# NOTA: e' volutamente diversa dalla HOME operativa usata in go_to_ready_pose()
# (quella su cui e' calibrata la ROI di vision_node). Va raggiunta SOLO dopo
# che control_node ha esaurito l'intera task list (comando sentinella
# "HOME_FINAL" su /task/command), mai durante il ciclo di pick-and-place,
# altrimenti la ROI calibrata su vision_node smetterebbe di essere valida.
FINAL_HOME_JOINTS = [0.000, -0.785, 0.000, -2.356, 0.000, 1.571, 0.785]
HOME_FINAL_CMD    = "HOME_FINAL"


class FrankaPicker:
    """Nodo ROS per la presa di oggetti con il robot Franka FR3."""

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('franka_pick_node', anonymous=True)

        # Offset di calibrazione XY recuperabili come parametri ROS
        self.xy_offset_x = rospy.get_param('~xy_offset_x', 0.0)
        self.xy_offset_y = rospy.get_param('~xy_offset_y', 0.0)

        # ------------------------------------------------------------
        # Inizializzazione MoveIt
        # ------------------------------------------------------------
        self.arm   = moveit_commander.MoveGroupCommander("fr3_arm")
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm.set_pose_reference_frame(ROBOT_BASE_FRAME)
        self.arm.set_max_velocity_scaling_factor(0.3)
        self.arm.set_max_acceleration_scaling_factor(0.3)
        self.arm.set_planning_time(10.0)
        self.arm.set_num_planning_attempts(5)
        self.arm.set_goal_position_tolerance(0.005)
        self.arm.set_goal_orientation_tolerance(0.01)
        self.arm.allow_replanning(True)

        self.gripper_mg = moveit_commander.MoveGroupCommander("fr3_hand")

        # ------------------------------------------------------------
        # Rilevamento automatico dell'ambiente (robot reale vs simulazione)
        # Il nodo verifica la presenza dei server di azione nativi Franka.
        # ------------------------------------------------------------
        self.grasp_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
        self.move_client  = actionlib.SimpleActionClient('/franka_gripper/move',  MoveAction)

        self.use_franka_native = (
            self.grasp_client.wait_for_server(rospy.Duration(1.0)) and
            self.move_client.wait_for_server(rospy.Duration(1.0))
        )

        # Client di fallback per la simulazione Gazebo
        self.gripper_client     = actionlib.SimpleActionClient(GRIPPER_ACTION_SERVER, GripperCommandAction)
        self.use_gripper_action = self.gripper_client.wait_for_server(rospy.Duration(1.0))

        if self.use_franka_native:
            rospy.loginfo("[PICK] Ambiente rilevato: robot reale. GraspAction nativa attiva.")
        else:
            rospy.loginfo("[PICK] Ambiente rilevato: simulazione. Utilizzo gripper_action standard.")

        # ------------------------------------------------------------
        # Vision pipeline
        # ------------------------------------------------------------
        self.bridge      = CvBridge()
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pin_cam              = PinholeCameraModel()
        self.camera_info_received = False
        self.sub_info = rospy.Subscriber(CAMERA_INFO_TOPIC, CameraInfo, self.camera_info_callback)

        # ------------------------------------------------------------
        # Stato interno della macchina a stati
        # ------------------------------------------------------------
        self.state         = "IDLE"
        self.target_found  = False
        self.allow_vision  = False
        self.target_pose   = None
        self.sub_mask      = None
        self.stable_frames = 0

        self.current_target  = None
        self.current_basket  = None
        self.place_completed = False

        # ------------------------------------------------------------
        # Publisher / Subscriber ROS
        # ------------------------------------------------------------
        self.pub_status     = rospy.Publisher('/task/status',        String, queue_size=10)
        self.sub_command    = rospy.Subscriber('/task/command',       String, self.command_callback)
        self.pub_place_cmd  = rospy.Publisher('/task/place_command',  String, queue_size=10)
        self.sub_place_done = rospy.Subscriber('/task/place_done',    String, self.place_done_callback)

        # Porta il robot in posizione HOME e attende la camera
        self.go_to_ready_pose()

        timeout = rospy.Time.now() + rospy.Duration(5.0)
        while not self.camera_info_received and rospy.Time.now() < timeout:
            rospy.sleep(0.1)

        rospy.loginfo("[PICK] Nodo pronto.")
        self.pub_status.publish("IDLE")

    # ------------------------------------------------------------------
    # Callback ROS
    # ------------------------------------------------------------------

    def camera_info_callback(self, msg):
        """Inizializza il modello pinhole della camera alla prima ricezione."""
        if not self.camera_info_received:
            self.pin_cam.fromCameraInfo(msg)
            self.camera_info_received = True

    def command_callback(self, msg):
        """
        Riceve un comando dal control_node nel formato:
            "<target_perception_suffix>,<cestino>"
        Ad esempio: "HSV_red,b2" oppure "mask_aruco_even,b1"
        """
        if self.state != "IDLE":
            return

        command_str = msg.data.strip()

        # Comando sentinella: fine missione, nessun target di percezione coinvolto.
        # Va gestito qui, PRIMA del parsing "<target>,<cestino>", perche' non ha
        # ne' un topic di percezione ne' un cestino associati.
        if command_str == HOME_FINAL_CMD:
            rospy.loginfo("[PICK] Comando ricevuto: %s (fine missione).", HOME_FINAL_CMD)
            self.state = "GOING_HOME_FINAL"
            return

        data_parts = command_str.split(',')
        self.current_target = data_parts[0]
        self.current_basket = data_parts[1] if len(data_parts) > 1 else "b1"

        topic_name = "/perception/" + self.current_target
        rospy.loginfo("[PICK] Comando ricevuto: target=%s, cestino=%s",
                      self.current_target, self.current_basket)

        self.state         = "WORKING"
        self.target_found  = False
        self.allow_vision  = True
        self.stable_frames = 0
        self.sub_mask = rospy.Subscriber(topic_name, Image, self.vision_callback)

    def place_done_callback(self, msg):
        """Riceve la conferma di completamento dal place_node."""
        if msg.data == "DONE":
            self.place_completed = True

    # ------------------------------------------------------------------
    # Controllo gripper
    # ------------------------------------------------------------------

    def gripper_open(self):
        """
        Apre il gripper al massimo.
        Sul robot reale usa MoveAction nativa; in simulazione usa GripperCommandAction
        o il MoveGroupCommander come fallback.
        """
        if self.use_franka_native:
            goal       = MoveGoal()
            goal.width = 0.08   # Apertura massima totale in metri
            goal.speed = 0.1
            self.move_client.send_goal(goal)
            self.move_client.wait_for_result(rospy.Duration(GRIPPER_TIMEOUT))
        elif self.use_gripper_action:
            goal                    = GripperCommandGoal()
            goal.command.position   = GRIPPER_OPEN
            goal.command.max_effort = 0.0
            self.gripper_client.send_goal(goal)
            self.gripper_client.wait_for_result(rospy.Duration(GRIPPER_TIMEOUT))
        else:
            self.gripper_mg.set_joint_value_target([GRIPPER_OPEN, GRIPPER_OPEN])
            self.gripper_mg.go(wait=True)
            self.gripper_mg.stop()

    def gripper_close(self):
        """
        Chiude il gripper sull'oggetto in modalita' geometry-agnostic.

        Sul robot reale usa GraspAction nativa con i parametri definiti
        nelle costanti GRASP_*:
          - GRASP_WIDTH quasi-zero: le dita cercano di chiudersi completamente
            e si fermano sull'oggetto indipendentemente dalla sua larghezza.
            Funziona con cubi, parallelepipedi e cilindri senza modifiche.
          - GRASP_EPSILON_OUTER largo: accetta qualunque posizione di arresto
            entro l'apertura massima del gripper (85 mm).
          - GRASP_FORCE costante: mantiene la presa durante il trasporto.
          - Il controllo sulla presa a vuoto e' delegato al check post-chiusura
            sul valore del giunto (soglia GRASP_EMPTY_THRESHOLD).

        In simulazione usa GripperCommandAction o MoveGroupCommander come fallback.
        """
        if self.use_franka_native:
            goal                = GraspGoal()
            goal.width          = GRASP_WIDTH
            goal.epsilon.inner  = GRASP_EPSILON_INNER
            goal.epsilon.outer  = GRASP_EPSILON_OUTER
            goal.speed          = GRASP_SPEED
            goal.force          = GRIPPER_FORCE

            rospy.loginfo("[PICK] GraspAction: width=%.3f m, epsilon_outer=%.3f m, force=%.1f N",
                          goal.width, goal.epsilon.outer, goal.force)
            self.grasp_client.send_goal(goal)
            result = self.grasp_client.wait_for_result(rospy.Duration(GRIPPER_TIMEOUT))
            if not result:
                rospy.logwarn("[PICK] GraspAction: timeout scaduto.")
        else:
            # Fallback simulazione: usa meta' dell'apertura nominale come target
            width_sim = 0.02
            if self.use_gripper_action:
                goal                    = GripperCommandGoal()
                goal.command.position   = width_sim
                goal.command.max_effort = GRIPPER_FORCE
                self.gripper_client.send_goal(goal)
                self.gripper_client.wait_for_result(rospy.Duration(GRIPPER_TIMEOUT))
            else:
                self.gripper_mg.set_joint_value_target([width_sim, width_sim])
                self.gripper_mg.go(wait=True)
                self.gripper_mg.stop()

    # ------------------------------------------------------------------
    # Movimenti del braccio
    # ------------------------------------------------------------------

    def go_to_ready_pose(self):
        """
        Porta il braccio nella posizione HOME in joint-space.
        Usa un planner deterministico e velocita' ridotta per garantire
        un percorso sicuro e riproducibile in ambiente reale.
        Ritenta fino a 3 volte in caso di fallimento del planner.
        """
        self.arm.set_planner_id("RRTConnectkConfigDefault")
        self.arm.set_max_velocity_scaling_factor(0.15)
        self.arm.set_max_acceleration_scaling_factor(0.1)

        joints_home = [0.000, -0.513, 0.000, -2.187, -0.000, 1.727, 0.785]
        self.arm.set_joint_value_target(joints_home)

        for attempt in range(3):
            if self.arm.go(wait=True):
                break
            rospy.logwarn("[PICK] go_to_ready_pose: tentativo %d/3 fallito, riprovo...", attempt + 1)
            rospy.sleep(0.5)

        self.arm.stop()
        self.arm.clear_pose_targets()

        # Ripristina i parametri di velocita' nominali
        self.arm.set_max_velocity_scaling_factor(0.3)
        self.arm.set_max_acceleration_scaling_factor(0.3)

    def go_to_final_home(self):
        """
        Porta il braccio nella posizione HOME "finale" di fine missione
        (FINAL_HOME_JOINTS), distinta dalla HOME operativa di go_to_ready_pose().
        Stessa strategia di pianificazione/velocita' ridotta e retry, per
        coerenza e sicurezza del movimento.
        """
        self.arm.set_planner_id("RRTConnectkConfigDefault")
        self.arm.set_max_velocity_scaling_factor(0.15)
        self.arm.set_max_acceleration_scaling_factor(0.1)

        self.arm.set_joint_value_target(FINAL_HOME_JOINTS)

        for attempt in range(3):
            if self.arm.go(wait=True):
                break
            rospy.logwarn("[PICK] go_to_final_home: tentativo %d/3 fallito, riprovo...", attempt + 1)
            rospy.sleep(0.5)

        self.arm.stop()
        self.arm.clear_pose_targets()

        # Ripristina i parametri di velocita' nominali
        self.arm.set_max_velocity_scaling_factor(0.3)
        self.arm.set_max_acceleration_scaling_factor(0.3)

    def reset_to_home(self):
        """Riporta il braccio in HOME con parametri nominali."""
        self.arm.clear_pose_targets()
        self.arm.set_max_velocity_scaling_factor(0.3)
        self.arm.set_max_acceleration_scaling_factor(0.3)
        self.go_to_ready_pose()
        rospy.sleep(0.5)

    # ------------------------------------------------------------------
    # Esecuzione della presa
    # ------------------------------------------------------------------

    def execute_pick(self):
        """
        Esegue la sequenza completa di presa:
          1. Aggiunge l'oggetto alla scena di collisione MoveIt
          2. Apertura gripper
          3. Movimento al punto di approccio (APPROACH_DIST sopra il target)
          4. Discesa cartesiana sul target
          5. Chiusura gripper geometry-agnostic con verifica della presa
          6. Sollevamento cartesiano con oggetto agganciato

        Restituisce True se la presa e' riuscita, False altrimenti.
        """
        if not self.target_found or self.target_pose is None:
            return False

        self.add_object_to_scene(
            self.target_pose.pose.position.x,
            self.target_pose.pose.position.y
        )
        self.gripper_open()

        # --- Fase 1: avvicinamento dall'alto ---
        approach_pose = copy.deepcopy(self.target_pose.pose)
        approach_pose.position.z += APPROACH_DIST

        self.arm.set_pose_target(approach_pose)
        if not self.arm.go(wait=True):
            rospy.logwarn("[PICK] Pianificazione approccio fallita.")
            self.reset_to_home()
            return False
        self.arm.clear_pose_targets()

        # --- Fase 2: discesa cartesiana sul punto di presa ---
        waypoints = [copy.deepcopy(self.target_pose.pose)]
        plan, fraction = self.arm.compute_cartesian_path(
            waypoints, eef_step=0.005, avoid_collisions=False
        )

        if fraction < 0.8:
            rospy.logwarn("[PICK] Path cartesiano parziale (%.0f%%). Uso go() come fallback.",
                          fraction * 100)
            self.arm.set_pose_target(self.target_pose.pose)
            self.arm.go(wait=True)
        else:
            self.arm.execute(plan, wait=True)
        self.arm.clear_pose_targets()

        # --- Fase 3: chiusura gripper e verifica ---
        self.gripper_close()
        rospy.sleep(1.5)  # Attesa per stabilizzazione della presa

        # Controllo presa a vuoto: se le dita sono sotto la soglia minima
        # l'oggetto non e' stato afferrato (o e' scivolato via).
        if self.gripper_mg.get_current_joint_values()[0] < GRASP_EMPTY_THRESHOLD:
            rospy.logerr("[PICK] Presa a vuoto rilevata (giunto < %.3f m). Operazione annullata.",
                         GRASP_EMPTY_THRESHOLD)
            self.clear_scene()
            self.gripper_open()
            return False

        self.attach_object()

        # --- Fase 4: sollevamento cartesiano ---
        lift_pose = copy.deepcopy(self.target_pose.pose)
        lift_pose.position.z += APPROACH_DIST

        plan_lift, frac_lift = self.arm.compute_cartesian_path(
            [lift_pose], eef_step=0.005, avoid_collisions=False
        )
        if frac_lift < 0.5:
            self.arm.set_pose_target(lift_pose)
            self.arm.go(wait=True)
        else:
            self.arm.execute(plan_lift, wait=True)

        self.arm.clear_pose_targets()
        return True

    # ------------------------------------------------------------------
    # Gestione scena MoveIt
    # ------------------------------------------------------------------

    def clear_scene(self):
        """Rimuove l'oggetto dalla scena di pianificazione (agganciato o libero)."""
        ee = self.arm.get_end_effector_link()
        self.scene.remove_attached_object(ee, name=OBJECT_ID)
        self.scene.remove_world_object(OBJECT_ID)
        rospy.sleep(0.5)

    def add_object_to_scene(self, x, y):
        """
        Aggiunge l'oggetto come bounding box di collisione nella scena MoveIt.
        Le dimensioni usano OBJECT_HEIGHT_ESTIMATE per tutti e tre gli assi:
        e' una approssimazione conservativa che funziona per cubi, parallelepipedi
        e cilindri senza richiedere la geometria esatta dell'oggetto.
        """
        self.clear_scene()
        rospy.sleep(0.1)
        obj_pose = PoseStamped()
        obj_pose.header.frame_id    = ROBOT_BASE_FRAME
        obj_pose.pose.position.x    = x
        obj_pose.pose.position.y    = y
        obj_pose.pose.position.z    = Z_TABLE_WORLD + OBJECT_HEIGHT_ESTIMATE / 2.0
        obj_pose.pose.orientation.w = 1.0
        self.scene.add_box(
            OBJECT_ID, obj_pose,
            size=(OBJECT_HEIGHT_ESTIMATE, OBJECT_HEIGHT_ESTIMATE, OBJECT_HEIGHT_ESTIMATE)
        )
        rospy.sleep(0.3)

    def attach_object(self):
        """Aggancia l'oggetto all'end-effector nella scena MoveIt."""
        ee = self.arm.get_end_effector_link()
        touch_links = [
            "fr3_hand", "fr3_leftfinger", "fr3_rightfinger",
            "fr3_finger_joint1", "fr3_finger_joint2"
        ]
        self.scene.attach_box(ee, OBJECT_ID, touch_links=touch_links)
        rospy.sleep(0.3)

    def detach_object(self):
        """Sgancia l'oggetto dall'end-effector e lo rimuove dalla scena."""
        ee = self.arm.get_end_effector_link()
        self.scene.remove_attached_object(ee, name=OBJECT_ID)
        rospy.sleep(0.2)
        self.scene.remove_world_object(OBJECT_ID)
        rospy.sleep(0.2)

    # ------------------------------------------------------------------
    # Visione
    # ------------------------------------------------------------------

    def _camera_tilt_angle(self, trans):
        """
        Calcola l'angolo tra l'asse ottico della camera e la verticale verso il basso.
        Usato per decidere se e' affidabile stimare l'angolo di rotazione dell'oggetto.
        """
        q   = trans.transform.rotation
        rot = tf_trans.quaternion_matrix([q.x, q.y, q.z, q.w])
        optical_axis_world = rot[:3, :3] @ np.array([0.0, 0.0, 1.0])
        down      = np.array([0.0, 0.0, -1.0])
        cos_angle = np.clip(np.dot(optical_axis_world, down), -1.0, 1.0)
        return np.arccos(cos_angle)

    def vision_callback(self, data):
        """
        Callback sulla maschera binaria pubblicata dal vision_node.
        Per ogni frame:
          1. Trova i contorni nella maschera
          2. Seleziona il contorno piu' grande che supera l'area minima
          3. Proietta il centroide del contorno sul piano del tavolo via ray-casting
          4. Applica il geofencing per ignorare oggetti gia' nei cestini
          5. Dopo N frame stabili, calcola la posa di presa e la salva

        La posa viene confermata solo dopo 6 frame stabili consecutivi per
        ridurre i falsi positivi dovuti a rumore o occlusioni temporanee.
        """
        if not self.allow_vision or self.target_found or not self.camera_info_received:
            return

        try:
            mask = self.bridge.imgmsg_to_cv2(data, "mono8")
        except Exception:
            return

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        # Ordina per area decrescente per analizzare prima l'oggetto piu' vicino/grande
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        valid_contour  = None
        target_x_final = 0.0
        target_y_final = 0.0
        angle_final    = 0.0

        for cnt in contours:
            if cv2.contourArea(cnt) < 300:
                continue

            rect      = cv2.minAreaRect(cnt)
            (u, v)    = rect[0]
            (w, h)    = rect[1]
            angle     = rect[2]
            if w < h:
                angle += 90.0

            # Recupera la trasformazione camera -> base
            try:
                trans = self.tf_buffer.lookup_transform(
                    ROBOT_BASE_FRAME, CAMERA_FRAME,
                    data.header.stamp, rospy.Duration(0.3)
                )
            except Exception:
                try:
                    trans = self.tf_buffer.lookup_transform(
                        ROBOT_BASE_FRAME, CAMERA_FRAME,
                        rospy.Time(0), rospy.Duration(0.1)
                    )
                except Exception:
                    continue

            # Proiezione del pixel sul piano Z del tavolo via ray-casting
            ray    = self.pin_cam.projectPixelTo3dRay((u, v))
            p0_cam = PoseStamped()
            p0_cam.pose.orientation.w = 1.0
            p1_cam = PoseStamped()
            p1_cam.pose.position.x    = ray[0]
            p1_cam.pose.position.y    = ray[1]
            p1_cam.pose.position.z    = ray[2]
            p1_cam.pose.orientation.w = 1.0

            p0_w = tf2_geometry_msgs.do_transform_pose(p0_cam, trans)
            p1_w = tf2_geometry_msgs.do_transform_pose(p1_cam, trans)

            x0, y0, z0 = p0_w.pose.position.x, p0_w.pose.position.y, p0_w.pose.position.z
            x1, y1, z1 = p1_w.pose.position.x, p1_w.pose.position.y, p1_w.pose.position.z

            if abs(z1 - z0) < 1e-5:
                continue

            t        = (Z_RAYCAST_PLANE - z0) / (z1 - z0)
            target_x = x0 + t * (x1 - x0) + self.xy_offset_x
            target_y = y0 + t * (y1 - y0) + self.xy_offset_y

            # Geofencing: ignora rilevamenti al di fuori dell'area di lavoro
            if target_y > 0.35 or target_y < -0.35:
                rospy.loginfo_throttle(
                    3.0, "[PICK] Geofencing: oggetto rilevato fuori area (y=%.2f), ignorato.", target_y
                )
                continue
            if target_x < 0.15 or target_x > 1.25:
                continue

            valid_contour  = cnt
            target_x_final = target_x
            target_y_final = target_y
            angle_final    = angle
            break

        if valid_contour is None:
            return

        # Richiede N frame consecutivi validi prima di confermare il target
        self.stable_frames += 1
        if self.stable_frames < 6:
            return

        # Stima dell'angolo di rotazione solo se la camera e' abbastanza verticale
        tilt = self._camera_tilt_angle(trans)
        if tilt < MAX_CAMERA_TILT_FOR_ANGLE:
            angle_rad = np.radians(angle_final)
        else:
            angle_rad = 0.0

        # Calcolo della posa di presa.
        # Z calcolata usando OBJECT_HEIGHT_ESTIMATE: indipendente dalla larghezza
        # dell'oggetto, funziona per qualunque geometria con altezza ~5 cm.
        z_grasp = Z_TABLE_WORLD + (OBJECT_HEIGHT_ESTIMATE / 2.0) + FINGER_TIP_OFFSET - CONTACT_MARGIN

        q_down  = tf_trans.quaternion_from_euler(np.pi, 0.0, 0.0)
        q_yaw   = tf_trans.quaternion_from_euler(0.0, 0.0, -np.pi / 4.0 + angle_rad)
        q_final = tf_trans.quaternion_multiply(q_down, q_yaw)

        self.target_pose = PoseStamped()
        self.target_pose.header.frame_id    = ROBOT_BASE_FRAME
        self.target_pose.pose.position.x    = target_x_final
        self.target_pose.pose.position.y    = target_y_final
        self.target_pose.pose.position.z    = z_grasp
        self.target_pose.pose.orientation.x = q_final[0]
        self.target_pose.pose.orientation.y = q_final[1]
        self.target_pose.pose.orientation.z = q_final[2]
        self.target_pose.pose.orientation.w = q_final[3]

        self.target_found = True
        self.allow_vision = False

        if self.sub_mask is not None:
            try:
                self.sub_mask.unregister()
            except Exception:
                pass
            self.sub_mask = None

    # ------------------------------------------------------------------
    # Loop principale
    # ------------------------------------------------------------------

    def run(self):
        """
        Macchina a stati principale.
        Cicla a 10 Hz attendendo lo stato WORKING per avviare la sequenza
        pick-and-place. Al termine pubblica SUCCESS, FAILED o NOT_FOUND
        verso il control_node e torna in IDLE.
        """
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.state == "GOING_HOME_FINAL":
                # Fine missione: nessun altro cubo da cercare, la ROI calibrata
                # su vision_node non serve piu'. Si raggiunge la HOME finale
                # richiesta dall'operatore e si segnala il completamento.
                rospy.loginfo("[PICK] Tutti i task completati. Vado alla posizione HOME finale.")
                self.go_to_final_home()
                self.state = "IDLE"
                self.pub_status.publish("MISSION_COMPLETE")

            elif self.state == "WORKING":
                self.allow_vision = True
                rospy.sleep(1.5)

                if self.target_found:
                    success = self.execute_pick()
                    if success:
                        rospy.loginfo("[PICK] Presa completata. Ritorno in HOME con l'oggetto.")
                        self.reset_to_home()

                        rospy.loginfo("[PICK] Avvio place_node per smistamento in: %s",
                                      self.current_basket)
                        self.place_completed = False
                        self.pub_place_cmd.publish(self.current_basket)

                        while not self.place_completed and not rospy.is_shutdown():
                            rospy.sleep(0.1)

                        rospy.loginfo("[PICK] Smistamento completato. Ritorno in HOME.")
                        self.reset_to_home()
                        self.pub_status.publish("SUCCESS")
                    else:
                        self.reset_to_home()
                        self.pub_status.publish("FAILED")
                else:
                    self.allow_vision = False
                    if self.sub_mask is not None:
                        try:
                            self.sub_mask.unregister()
                        except Exception:
                            pass
                        self.sub_mask = None
                    self.pub_status.publish("NOT_FOUND")
                    self.reset_to_home()

                self.target_found = False
                self.allow_vision = False
                self.state        = "IDLE"
                rospy.sleep(1.0)
                self.pub_status.publish("IDLE")

            rate.sleep()


if __name__ == '__main__':
    try:
        FrankaPicker().run()
    except rospy.ROSInterruptException:
        pass