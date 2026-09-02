# Franka Emika FR3 — Autonomous Vision-Guided Pick and Place

Sistema autonomo di visione artificiale e manipolazione robotica per il manipolatore a 7 gradi di libertà Franka Emika FR3, sviluppato su ROS Noetic e MoveIt.

Il progetto implementa una pipeline modulare in grado di rilevare oggetti colorati e marker ArUco tramite telecamera RGB, calcolarne la posa 3D sul piano di lavoro mediante ray-casting ottico, ed eseguire task di pick-and-place coordinati con smistamento selettivo in cestini dedicati. Include inoltre un modulo parallelo di object detection generica basato su YOLOv8l con filtraggio temporale.

---

## Architettura di Sistema

L'infrastruttura software è strutturata in nodi ROS cooperanti coordinati da una macchina a stati:

```text
               +-----------------------------+
               |        control_node         |  <--- CLI Interattiva Utente
               +-----------------------------+
                      |               ^
        /task/command |               | /task/status
                      v               |
               +-----------------------------+
               |          pick_node          |  <--- MoveIt (Cartesiano + GraspAction)
               +-----------------------------+
                 |       ^               |
    /mask/target |       | /mask/<color> | /task/place_command
                 v       |               v
      +---------------------+     +----------------------+
      |     vision_node     |     |      place_node      |  <--- Joint Space MoveIt
      +---------------------+     +----------------------+
                 ^                           | /task/place_done
                 | /camera/...               v
        [ Camera RGB (D435) ]         [ Cestini B1..B4 ]
