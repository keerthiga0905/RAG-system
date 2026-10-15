"""
CIT Campus Roadmap Navigation Inference Engine
Loads the trained CIT roadmap model (models/cit_roadmap_model.json)
and provides accurate, step-by-step route prediction whenever a student
asks to navigate between campus locations.
"""

import os
import re
import json
from typing import Dict, Any, Optional, Tuple, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_FILE = os.path.join(BASE_DIR, "models", "cit_roadmap_model.json")


class RoadmapNavigator:
    def __init__(self):
        self.model = None
        self.locations: Dict[str, Any] = {}
        self.edges: List[Dict[str, Any]] = []
        self.alias_map: Dict[str, str] = {}
        self.precomputed_routes: Dict[str, Any] = {}
        self._load_model()

    def _load_model(self):
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, "r", encoding="utf-8") as f:
                    self.model = json.load(f)
                self.locations = self.model.get("locations", {})
                self.edges = self.model.get("edges", [])
                self.alias_map = self.model.get("alias_map", {})
                self.precomputed_routes = self.model.get("precomputed_routes", {})
            except Exception as e:
                print(f"Warning: Failed to load cit_roadmap_model.json: {e}")

    def is_navigation_query(self, query: str) -> bool:
        """Determines if query is asking for campus navigation / directions."""
        q_lower = query.lower().strip()
        nav_keywords = [
            "navigate", "navigation", "walking route", "shortest walking route",
            "walking directions", "walking path", "how do i go", "how to go",
            "directions from", "directions to", "route from", "path from",
            "way to reach", "how to reach"
        ]
        if any(kw in q_lower for kw in nav_keywords):
            # Exclude city-level transport like bus/train to Coimbatore
            if any(w in q_lower for w in ["bus", "train", "flight", "reach coimbatore", "reach cit from railway"]):
                return False
            return True
        return False

    def match_location(self, text: str) -> Optional[str]:
        """Resolves natural language location phrase to a canonical CIT location ID."""
        cleaned = text.lower().strip()
        cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned).strip()
        cleaned = re.sub(r"[?!.,]$", "", cleaned).strip()
        cleaned_no_block = re.sub(r"\b(block|building|campus|department|dept|hall)\b", "", cleaned).strip()

        # 1. Exact match in alias map
        if cleaned in self.alias_map:
            return self.alias_map[cleaned]
        if cleaned_no_block in self.alias_map:
            return self.alias_map[cleaned_no_block]

        # 2. Substring matching with aliases sorted by length descending
        sorted_aliases = sorted(self.alias_map.keys(), key=lambda x: len(x), reverse=True)
        for alias in sorted_aliases:
            if alias in cleaned or (len(cleaned_no_block) >= 3 and cleaned_no_block in alias):
                return self.alias_map[alias]

        return None

    def extract_endpoints(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parses origin (from_where) and destination (to_where) from natural language:
        - "Navigate me from Computing block to Library block"
        - "Navigate from Main Block to Auditorium"
        - "Navigate to IT block from Library"
        - "How to reach Computing block from Auditorium"
        """
        q_lower = query.lower().strip()
        q_lower = re.sub(r"[?!.,]$", "", q_lower).strip()

        # Pattern 1: navigate/go/reach [to/me to] <DEST> from <START>
        p1 = re.search(r"(?:navigate(?:\s+me)?(?:\s+to)?|reach|go to|walk to|get to|directions to)\s+(.+?)\s+from\s+(.+?)(?:\?|$)", q_lower)
        if p1:
            dest_text = p1.group(1).strip()
            start_text = p1.group(2).strip()
            dest_id = self.match_location(dest_text)
            start_id = self.match_location(start_text)
            if dest_id and start_id:
                return start_id, dest_id

        # Pattern 2: navigate/go [me] from <START> to <DEST>
        p2 = re.search(r"(?:navigate(?:\s+me)?\s+)?from\s+(.+?)\s+(?:to|towards)\s+(.+?)(?:\?|$)", q_lower)
        if p2:
            start_text = p2.group(1).strip()
            dest_text = p2.group(2).strip()
            start_id = self.match_location(start_text)
            dest_id = self.match_location(dest_text)
            if start_id and dest_id:
                return start_id, dest_id

        # Pattern 3: between <START> and <DEST>
        p3 = re.search(r"(?:between)\s+(.+?)\s+(?:and)\s+(.+?)(?:\?|$)", q_lower)
        if p3:
            start_text = p3.group(1).strip()
            dest_text = p3.group(2).strip()
            start_id = self.match_location(start_text)
            dest_id = self.match_location(dest_text)
            if start_id and dest_id:
                return start_id, dest_id

        # Fallback: scan for any two known location names appearing in order
        found = []
        sorted_aliases = sorted(self.alias_map.keys(), key=lambda x: len(x), reverse=True)
        for alias in sorted_aliases:
            match = re.search(r"\b" + re.escape(alias) + r"\b", q_lower)
            if match:
                pos = match.start()
                lid = self.alias_map[alias]
                if lid not in [f[1] for f in found]:
                    found.append((pos, lid))

        found.sort(key=lambda x: x[0])
        if len(found) >= 2:
            # Check if 'from' word indicates start location position
            if "from " in q_lower:
                from_pos = q_lower.find("from ")
                if found[1][0] > from_pos > found[0][0]:
                    return found[1][1], found[0][1]
            return found[0][1], found[1][1]

        return None, None

    def navigate(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Computes roadmap-predicted walking route for student queries.
        Returns response dict or None if endpoints could not be identified.
        """
        start_id, dest_id = self.extract_endpoints(query)
        if not start_id or not dest_id:
            return None

        if start_id == dest_id:
            loc_name = self.locations.get(start_id, {}).get("name", "the selected location")
            return {
                "answer": f"You are already at {loc_name}.",
                "start_id": start_id,
                "dest_id": dest_id,
                "distance_m": 0,
                "seconds": 0,
                "success": True
            }

        route_key = f"{start_id}->{dest_id}"
        if route_key in self.precomputed_routes:
            route_data = self.precomputed_routes[route_key]
            return {
                "answer": route_data["route_text"],
                "start_id": start_id,
                "dest_id": dest_id,
                "start_name": route_data["start_name"],
                "dest_name": route_data["dest_name"],
                "distance_m": route_data["total_distance_m"],
                "seconds": route_data["total_seconds"],
                "path": route_data["path"],
                "success": True
            }

        return None


roadmap_navigator = RoadmapNavigator()
