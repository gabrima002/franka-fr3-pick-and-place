#!/usr/bin/env python3
"""
place_node.py - Franka FR3 Place Node (Versione Spazio Giunti)
Gestisce il movimento verso il cestino di destinazione e il rilascio del cubo.

Il nodo riceve la stringa identificativa del cestino su /task/place_command,
esegue il movimento nello spazio dei giunti (joint space) verso la configurazione mappata,
apre il gripper e pubblica DONE su /task/place_done.

Questo approccio garantisce movimenti fluidi, lineari e prevedibili tra la home e i cestini.
"""

import sys
import rospy
import moveit_commander
from std_msgs.msg import String


class FrankaPlacer:
    """Nodo ROS per il deposito dei cubi nei cestini di destinazione tramite spazio giunti."""

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('franka_place_node', anonymous=True)

        self.arm        = moveit_commander.MoveGroupCommander("fr3_arm")
        self.scene      = moveit_commander.PlanningSceneInterface()
        self.gripper_mg = moveit_commander.MoveGroupCommander("fr3_hand")

        # Frame di riferimento (lasciato per consistenza, anche se i giunti sono indipendenti dal frame)
        self.arm.set_pose_reference_frame("fr3_link0")

        # ------------------------------------------------------------
        # Configurazione dei 7 giunti (in radianti) per ciascun cestino.
        # Formato: [j1, j2, j3, j4, j5, j6, j7]
        # ------------------------------------------------------------
        self.drop_joint_poses = {
            "b1": [1.766, 0.462, 0.0, -2.356, 0.0, 2.818, 0.785],
            "b2": [1.248, 0.513, -0.049, -2.356, 0.0, 2.808, 0.785],
            "b3": [-1.494, 0.191, 0.0, -2.356, 0.0, 2.517, 0.785],
            "b4": [-2.025, 0.191, 0.0, -2.356, 0.0, 2.517, 0.785],
        }

        # Configurazione dei giunti di default
        self.default_drop_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

        self.sub_place_cmd  = rospy.Subscriber('/task/place_command', String, self.execute_place)
        self.pub_place_done = rospy.Publisher('/task/place_done',     String, queue_size=10)

    def execute_place(self, msg):
        """
        Callback principale del nodo.
        Riceve il nome del cestino, ricava i valori dei giunti target,
        pianifica ed esegue il movimento, apre il gripper e pubblica il DONE.
        """
        target_basket = msg.data.strip().lower()
        rospy.loginfo("[PLACE] Avvio smistamento verso: %s", target_basket)

        # Recupera la configurazione dei giunti dal dizionario
        target_joints = self.drop_joint_poses.get(target_basket, self.default_drop_joints)

        if target_basket not in self.drop_joint_poses:
            rospy.logwarn("[PLACE] Cestino '%s' non trovato nel dizionario. "
                          "Utilizzo configurazione di default.", target_basket)

        # --- Pianificazione e movimento nello Spazio dei Giunti ---
        self.arm.set_joint_value_target(target_joints)
        plan_result = self.arm.plan()

        # Gestione compatibilità MoveIt: plan() può restituire una tupla o il piano diretto
        if isinstance(plan_result, tuple):
            plan    = plan_result[1]
            success = plan_result[0]
        else:
            plan    = plan_result
            success = True

        if not success or plan is None or not hasattr(plan, 'joint_trajectory'):
            rospy.logwarn("[PLACE] Pianificazione fallita nello spazio giunti. Tentativo con go().")
            self.arm.go(wait=True)
        else:
            self.arm.execute(plan, wait=True)

        self.arm.stop()
        self.arm.clear_pose_targets()
        rospy.sleep(0.5)

        # --- Apertura gripper per rilascio del cubo ---
        self.gripper_mg.set_joint_value_target([0.04, 0.04])
        self.gripper_mg.go(wait=True)
        self.gripper_mg.stop()

        rospy.loginfo("[PLACE] Smistamento completato verso: %s", target_basket)
        self.pub_place_done.publish("DONE")

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        rospy.loginfo("[PLACE] Place Node (Spazio Giunti) pronto. In attesa su /task/place_command.")
        FrankaPlacer().run()
    except rospy.ROSInterruptException:
        pass