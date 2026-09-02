# Franka Emika FR3 — Autonomous Vision-Guided Pick and Place

Sistema autonomo di visione artificiale e manipolazione robotica per il manipolatore a 7 gradi di libertà Franka Emika FR3, sviluppato su ROS Noetic e MoveIt[span_0](start_span)[span_0](end_span)[span_1](start_span)[span_1](end_span)[span_2](start_span)[span_2](end_span).

Il progetto implementa una pipeline modulare in grado di rilevare oggetti colorati e marker ArUco tramite telecamera RGB[span_3](start_span)[span_3](end_span), calcolarne la posa 3D sul piano di lavoro mediante ray-casting ottico[span_4](start_span)[span_4](end_span), ed eseguire task di pick-and-place coordinati con smistamento selettivo in cestini dedicati[span_5](start_span)[span_5](end_span)[span_6](start_span)[span_6](end_span)[span_7](start_span)[span_7](end_span). Include inoltre un modulo parallelo di object detection generica basato su YOLOv8l con filtraggio temporale[span_8](start_span)[span_8](end_span).

---

## Dimostrazione Funzionale

| Pick & Place Autonomo (Camera + MoveIt) | Rilevamento Real-Time (YOLOv8l + HSV) |
| :---: | :---: |
| ![Franka Pick and Place Demo](docs/pick_and_place_demo.gif) | ![Vision Detection Demo](docs/vision_detection.gif) |
| *Ciclo continuo di presa e smistamento nei cestini* | *Segmentazione colore / inferenza YOLOv8l* |

> *Nota: Carica le tue GIF animate dentro la cartella docs/ con i nomi pick_and_place_demo.gif e vision_detection.gif.*

---

## Architettura di Sistema

L'infrastruttura software è strutturata in nodi ROS cooperanti coordinati da una macchina a stati[span_9](start_span)[span_9](end_span)[span_10](start_span)[span_10](end_span):

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
