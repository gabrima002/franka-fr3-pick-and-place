#!/usr/bin/env python3
"""
launch.py - Franka FR3 Bringup Launcher

Avvia tutti i nodi "di appoggio" (vision_node, pick_node, place_node),
definiti direttamente qui sotto - non serve nessun file .launch esterno -
e resta in ascolto su /task/status finche' non riceve il primo "IDLE"
pubblicato da pick_node.

Quel primo IDLE e' il segnale affidabile che:
  1. il robot ha completato go_to_ready_pose() -> e' in HOME,
     la posizione rispetto a cui e' tarata la ROI di vision_node;
  2. la camera_info e' stata ricevuta -> la pipeline di percezione
     e' calibrata e pronta.

Solo a quel punto stampa il via libera per lanciare control_node
(che va avviato A MANO in un altro terminale, perche' usa input()
da riga di comando e roslaunch non collega lo stdin).
"""

import rospy
import roslaunch
import roslaunch.scriptapi
from std_msgs.msg import String


# Nome del pacchetto in cui si trovano gli script (stesso di tutti gli altri nodi)
PACKAGE_NAME = 'franka_student'

# Timeout di sicurezza: se dopo N secondi non arriva nessun IDLE,
# probabilmente qualcosa non e' andato bene nell'homing o nella camera.
READY_TIMEOUT_SEC = 60.0


class BringupLauncher:
    """Avvia i nodi di appoggio e segnala quando il robot e' pronto in HOME."""

    def __init__(self):
        rospy.init_node('bringup_launcher', anonymous=True)

        self.ready = False
        self.first_status_received = False

        self.sub_status = rospy.Subscriber('/task/status', String, self.status_callback)

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)

        # ------------------------------------------------------------
        # Definizione diretta dei nodi da avviare.
        # node_type = nome del file eseguibile dentro il pacchetto
        # (roslaunch lo cerca allo stesso modo di `rosrun`).
        # ------------------------------------------------------------
        nodes = [
            roslaunch.core.Node(
                package=PACKAGE_NAME, node_type='vision_node.py',
                name='vision_node', output='screen', respawn=True
            ),
            roslaunch.core.Node(
                package=PACKAGE_NAME, node_type='pick_node.py',
                name='pick_node', output='screen', respawn=False
            ),
            roslaunch.core.Node(
                package=PACKAGE_NAME, node_type='place_node.py',
                name='place_node', output='screen', respawn=False
            ),
        ]

        self.launch = roslaunch.scriptapi.ROSLaunch()
        self.launch.start()
        self._launch_configs = []
        for node in nodes:
            process = self.launch.launch(node)
            self._launch_configs.append(process)

    def status_callback(self, msg):
        """Intercetta il primo IDLE pubblicato da pick_node: e' il segnale di 'pronto'."""
        if not self.first_status_received and msg.data == "IDLE":
            self.first_status_received = True
            self.ready = True
            rospy.loginfo("[LAUNCHER] Ricevuto IDLE da pick_node: robot in HOME, camera pronta.")

    def run(self):
        rospy.loginfo(
            "[LAUNCHER] Nodi di appoggio avviati (vision_node, pick_node, place_node)."
        )

        rospy.loginfo(
            "[LAUNCHER] In attesa che il robot raggiunga la posizione HOME "
            "e la camera sia calibrata (timeout %.0fs)...", READY_TIMEOUT_SEC
        )

        rate      = rospy.Rate(2)
        start_time = rospy.Time.now()

        while not rospy.is_shutdown() and not self.ready:
            if (rospy.Time.now() - start_time).to_sec() > READY_TIMEOUT_SEC:
                rospy.logwarn(
                    "[LAUNCHER] Timeout: nessun IDLE ricevuto da pick_node dopo %.0fs. "
                    "Controlla i log di pick_node (homing/MoveIt/camera potrebbero "
                    "essere bloccati).", READY_TIMEOUT_SEC
                )
                break
            rate.sleep()

        if self.ready:
            print("\n" + "=" * 62)
            print("  TUTTO PRONTO")
            print("  Robot in posizione HOME, camera calibrata, ROI valida.")
            print("  Apri un nuovo terminale e lancia:")
            print("      rosrun {}  control_node.py".format(PACKAGE_NAME))
            print("=" * 62 + "\n")
        else:
            print("\n" + "!" * 62)
            print("  ATTENZIONE: nessuna conferma di readiness ricevuta.")
            print("  Verifica manualmente che il robot sia in HOME prima")
            print("  di lanciare control_node.")
            print("!" * 62 + "\n")

        # Il launcher resta attivo per mantenere in vita i nodi figli
        # e gestirne uno shutdown pulito con Ctrl+C.
        try:
            self.launch.spin()
        except rospy.ROSInterruptException:
            pass
        finally:
            self.launch.shutdown()


if __name__ == '__main__':
    try:
        BringupLauncher().run()
    except rospy.ROSInterruptException:
        pass