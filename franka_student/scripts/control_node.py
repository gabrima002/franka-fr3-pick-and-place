#!/usr/bin/env python3
"""
control_node.py - Franka FR3 Control Node
Supervisore della sequenza di pick-and-place.

All'avvio raccoglie la configurazione della gara dall'operatore tramite
interfaccia interattiva a riga di comando, costruisce la lista ordinata
dei task e li invia uno per volta al pick_node tramite il topic /task/command.

Protocollo di comunicazione con il pick_node:
  - Pubblica su  /task/command  la stringa "<target>,<cestino>"
  - Riceve da    /task/status   uno dei valori: IDLE | WORKING | SUCCESS | NOT_FOUND | FAILED

Logica di avanzamento:
  - SUCCESS:   il cubo e' stato consegnato; rimane sullo stesso task
               (potrebbero esserci altri cubi dello stesso tipo).
  - NOT_FOUND: nessun cubo del tipo corrente trovato; avanza al task successivo.
  - FAILED:    errore di presa; ritenta lo stesso task.
"""

import rospy
from std_msgs.msg import String

VALID_BASKETS = {"b1", "b2", "b3", "b4"}
VALID_COLORS  = {"red", "orange", "blue", "green"}

# Comando sentinella (deve coincidere con HOME_FINAL_CMD in pick_node.py):
# richiede il ritorno alla HOME "finale" di fine missione, distinta dalla
# HOME operativa su cui e' calibrata la ROI di vision_node. Va inviato SOLO
# dopo aver esaurito l'intera task list, altrimenti la ROI smetterebbe di
# essere valida per i task successivi.
HOME_FINAL_CMD = "HOME_FINAL"


class ControlNode:
    """Nodo supervisore della pipeline di pick-and-place."""

    def __init__(self):
        rospy.init_node('control_node', anonymous=True)

        self.pub_command = rospy.Publisher('/task/command', String, queue_size=10)
        self.sub_status  = rospy.Subscriber('/task/status', String, self.status_callback)

        self.current_status     = "IDLE"
        self.current_task_index = 0
        self.homing_requested   = False

        # Attende che i publisher siano registrati nel grafo ROS
        rospy.sleep(1.0)

        # ------------------------------------------------------------------
        # Interfaccia di configurazione interattiva
        # L'operatore inserisce i parametri della gara prima dell'avvio.
        # ------------------------------------------------------------------
        print("\n" + "=" * 55)
        print("FRANKA CHALLENGE - CONFIGURAZIONE GARA")
        print("=" * 55)

        try:
            # Parita' degli ArUco da prelevare
            parity = self._ask_parity(
                "1. Parita' marker ArUco da prelevare (pari/dispari): "
            )
            aruco_suffix = "even" if parity == "pari" else "odd"

            # Cestino di destinazione per il marker ArUco
            basket_aruco = self._ask_basket(
                "2. Cestino di destinazione per l'ArUco (b1/b2/b3/b4): "
            )

            # Primo colore target e relativo cestino
            col1_color, col1_basket = self._ask_color_and_basket(
                "3. Primo colore target e cestino, separati da spazio (es. red b2): ",
                exclude_colors=set()
            )

            # Secondo colore target e relativo cestino (deve essere diverso dal primo)
            col2_color, col2_basket = self._ask_color_and_basket(
                "4. Secondo colore target e cestino, separati da spazio (es. blue b3): ",
                exclude_colors={col1_color}
            )

            # Terzo colore target e relativo cestino (deve essere diverso dai precedenti)
            col3_color, col3_basket = self._ask_color_and_basket(
                "5. Terzo colore target e cestino, separati da spazio (es. green b4): ",
                exclude_colors={col1_color, col2_color}
            )

            # Costruzione della task list nel formato "<perception_suffix>,<cestino>".
            # Per gli ArUco il suffix include la parita' (even/odd): pick_node si
            # iscrive al topic /perception/mask_aruco_<even|odd> corrispondente,
            # gia' filtrato da vision_node, e resta sul task (SUCCESS -> IDLE)
            # finche' non trova piu' nessun marker di quella parita' (NOT_FOUND).
            self.task_list = [
                f"mask_aruco_{aruco_suffix},{basket_aruco}",
                f"HSV_{col1_color},{col1_basket}",
                f"HSV_{col2_color},{col2_basket}",
                f"HSV_{col3_color},{col3_basket}",
            ]

            print("\nConfigurazione completata. Task programmati:")
            for i, task in enumerate(self.task_list, start=1):
                print(f"  {i}. {task}")
            print("=" * 55 + "\n")

        except (IndexError, ValueError):
            print("\nERRORE: formato non valido. "
                  "Inserire COLORE e CESTINO separati da uno spazio.")
            exit(1)

    def _ask_parity(self, prompt):
        """Chiede pari/dispari all'operatore, ripetendo finche' non e' valido."""
        while True:
            answer = input(prompt).strip().lower()
            if answer in ("pari", "dispari"):
                return answer
            print("  -> Valore non valido. Digitare 'pari' o 'dispari'.")

    def _ask_basket(self, prompt):
        """Chiede un nome cestino (b1-b4), ripetendo finche' non e' valido."""
        while True:
            answer = input(prompt).strip().lower()
            if answer in VALID_BASKETS:
                return answer
            print(f"  -> Cestino non valido. Usare uno tra: {sorted(VALID_BASKETS)}.")

    def _ask_color_and_basket(self, prompt, exclude_colors=None):
        """
        Chiede 'colore cestino' separati da spazio, validando entrambi.
        Se exclude_colors e' fornito, rifiuta un colore gia' scelto in
        precedenza (evita due task identici per lo stesso colore).
        """
        exclude_colors = exclude_colors or set()
        while True:
            parts = input(prompt).strip().split()
            if len(parts) != 2:
                print(f"  -> Formato non valido. Colori ammessi: {sorted(VALID_COLORS)}, "
                      f"cestini ammessi: {sorted(VALID_BASKETS)}.")
                continue

            color, basket = parts[0].lower(), parts[1].lower()

            if color not in VALID_COLORS or basket not in VALID_BASKETS:
                print(f"  -> Formato non valido. Colori ammessi: {sorted(VALID_COLORS)}, "
                      f"cestini ammessi: {sorted(VALID_BASKETS)}.")
                continue

            if color in exclude_colors:
                print(f"  -> Colore '{color}' gia' assegnato a un altro task. "
                      f"Sceglierne uno diverso tra: {sorted(VALID_COLORS - exclude_colors)}.")
                continue

            return color, basket

    def status_callback(self, msg):
        """Aggiorna lo stato interno alla ricezione di un messaggio da /task/status."""
        self.current_status = msg.data

    def run(self):
        """
        Loop principale del supervisore.
        Scorre la task list e gestisce le transizioni di stato
        in base ai feedback ricevuti dal pick_node.
        """
        rate = rospy.Rate(1)  # 1 Hz e' sufficiente per la supervisione

        while not rospy.is_shutdown():

            # Verifica completamento di tutti i task: SOLO a questo punto
            # (tutti i cubi gestiti) si puo' chiedere il ritorno alla HOME
            # finale, perche' prima la ROI di vision_node deve restare
            # calibrata sulla HOME operativa usata durante il pick-and-place.
            if not self.task_list or self.current_task_index >= len(self.task_list):
                if not self.homing_requested:
                    rospy.loginfo(
                        "[CONTROL] Tutti i task completati. Invio richiesta di ritorno "
                        "alla HOME finale."
                    )
                    self.pub_command.publish(HOME_FINAL_CMD)
                    self.homing_requested = True
                elif self.current_status == "MISSION_COMPLETE":
                    rospy.loginfo(
                        "[CONTROL] Robot tornato alla HOME finale. Missione conclusa."
                    )
                    break
                rate.sleep()
                continue

            target_task = self.task_list[self.current_task_index]

            if self.current_status == "IDLE":
                # Invia il task corrente al pick_node
                rospy.loginfo("[CONTROL] Invio task: %s", target_task)
                self.current_status = "WORKING"
                self.pub_command.publish(target_task)

            elif self.current_status == "SUCCESS":
                # Cubo consegnato correttamente; verifica se ne esistono altri dello stesso tipo
                rospy.loginfo("[CONTROL] Task '%s' completato. Verifico ulteriori oggetti.", target_task)
                self.current_status = "IDLE"
                rospy.sleep(2.0)

            elif self.current_status == "NOT_FOUND":
                # Nessun oggetto del tipo corrente trovato; avanza al task successivo
                rospy.loginfo("[CONTROL] Nessun oggetto trovato per '%s'. Avanzo al task successivo.", target_task)
                self.current_task_index += 1
                self.current_status = "IDLE"
                rospy.sleep(1.0)

            elif self.current_status == "FAILED":
                # Errore di presa; ritenta lo stesso task
                rospy.logwarn("[CONTROL] Presa fallita per '%s'. Nuovo tentativo.", target_task)
                self.current_status = "IDLE"
                rospy.sleep(2.0)

            rate.sleep()


if __name__ == '__main__':
    try:
        ControlNode().run()
    except rospy.ROSInterruptException:
        pass