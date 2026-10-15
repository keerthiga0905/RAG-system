CIT CAMPUS NAVIGATION DATASET — 8-PLACE CALIBRATED STARTER PACKAGE

1. Overview
This dataset provides a simulated pedestrian navigation network for the Coimbatore Institute of Technology (CIT) campus, calibrated against open-source maps (OpenStreetMap and Google Maps references) anchored around 11.028° N, 77.027° E along Avinashi Road, Coimbatore.

2. Selected Campus Locations (8 Core Landmarks)
- CIT001: Campus Main Gate (11.02920, 77.02715) — Main entrance on Avinashi Road
- CIT002: Main Building / Main Block (11.02850, 77.02720) — Administrative heart, Principal's office
- CIT003: Library Block (11.02845, 77.02675) — Central library adjacent west of Main Building
- CIT004: Computing Block (11.02779, 77.02741) — Department of Computing and M.Sc. laboratories
- CIT005: IT Block (11.02790, 77.02680) — Information Technology department building
- CIT006: CIT Auditorium (11.02830, 77.02780) — Central institutional auditorium
- CIT007: MSc / MBA Block (11.02720, 77.02750) — Management studies and postgraduate block
- CIT008: Mechanical & Civil Engineering Block (11.02730, 77.02690) — Core engineering labs and workshops

3. Files in this Package
1. locations.csv:
   Contains the 8 verified locations, aliases, categories, and geographic coordinates.
2. walking_path_edges_template.csv:
   Contains 18 bidirectional traversable pedestrian path edges with distances in metres, walking times, path types, and landmarks.
3. route_training_template.csv:
   Provides verified turn-by-turn route steps in JSON format for student queries (e.g. Computing Block to Main Block, Library to IT Block, Main Block to Auditorium).
4. navigation_intents.jsonl:
   Query-to-intent training pairs for campus navigation queries.

4. Turn-by-Turn Instruction Format
Each route step includes:
- step: Sequential step number (1, 2, 3...)
- instruction: Direct imperative natural-language direction
- turn: Navigation turn action (straight, left, right, slight_right, arrive)
- distance_m: Segment distance in metres
- landmark: Visible reference along the pathway
