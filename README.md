# Franka Emika FR3 — Autonomous Vision-Guided Pick and Place

Sistema autonomo di visione artificiale e manipolazione robotica per il manipolatore a 7 gradi di libertà Franka Emika FR3, sviluppato su ROS Noetic e MoveIt.

Il progetto implementa una pipeline modulare in grado di rilevare oggetti colorati e marker ArUco tramite telecamera RGB, calcolarne la posa 3D sul piano di lavoro mediante ray-casting ottico, ed eseguire task di pick-and-place coordinati con smistamento selettivo in cestini dedicati. Include inoltre un modulo parallelo di object detection generica basato su YOLOv8l con filtraggio temporale.

> **Esame superato con il massimo dei voti** — L'intera sessione sperimentale è stata registrata e archiviata sotto forma di file ROS bag.

---

## Demo del Progetto

<div align="center">
  <img src="media/demo.gif" alt="Demo del progetto Franka Emika FR3" width="750" />
  
  <sub><i>*L'animazione mostra una breve sequenza dimostrativa del ciclo di rilevamento e presa. A causa dei limiti di dimensione imposti da GitHub, la registrazione integrale dell'esame e i relativi file ROS bag non possono essere caricati direttamente nella repository.*</i></sub>
</div>

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


