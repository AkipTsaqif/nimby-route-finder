"""
Road Data Service

Fetches road/street data from OpenStreetMap via Overpass API.
Provides road direction and proximity information for pathfinding constraints.
"""

import math
import asyncio
import numpy as np
from typing import Tuple, Optional, Dict, List, Any
from dataclasses import dataclass
import httpx


@dataclass
class RoadSegment:
    """A road segment with start/end points and direction"""
    start_lat: float
    start_lng: float
    end_lat: float
    end_lng: float
    road_type: str  # highway, primary, secondary, etc.
    name: Optional[str] = None
    
    @property
    def direction_rad(self) -> float:
        """Calculate direction in radians (0 = North, clockwise)"""
        dlat = self.end_lat - self.start_lat
        dlng = self.end_lng - self.start_lng
        # atan2 gives angle from positive x-axis, we want from North
        angle = math.atan2(dlng, dlat)
        return angle
    
    @property
    def direction_deg(self) -> float:
        """Calculate direction in degrees (0 = North, clockwise)"""
        return math.degrees(self.direction_rad)
    
    @property
    def length_m(self) -> float:
        """Approximate length in meters"""
        lat_center = (self.start_lat + self.end_lat) / 2
        meters_per_deg_lat = 111320
        meters_per_deg_lng = 111320 * math.cos(math.radians(lat_center))
        
        dlat_m = (self.end_lat - self.start_lat) * meters_per_deg_lat
        dlng_m = (self.end_lng - self.start_lng) * meters_per_deg_lng
        
        return math.sqrt(dlat_m**2 + dlng_m**2)


class RoadService:
    """
    Service for fetching and processing road data from OpenStreetMap.
    
    Uses Overpass API to query road networks within a bounding box.
    """
    
    # Overpass API endpoints (multiple for redundancy)
    OVERPASS_ENDPOINTS = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]
    
    # Road types to fetch (ordered by importance)
    ROAD_TYPES = [
        "motorway", "motorway_link",
        "trunk", "trunk_link",
        "primary", "primary_link",
        "secondary", "secondary_link",
        "tertiary", "tertiary_link",
        "residential",
        "unclassified",
        "service",
    ]
    
    def __init__(self):
        self.cache: Dict[str, List[RoadSegment]] = {}
    
    async def get_roads_in_bounds(
        self,
        min_lat: float,
        min_lng: float,
        max_lat: float,
        max_lng: float,
        road_types: Optional[List[str]] = None
    ) -> List[RoadSegment]:
        """
        Fetch all roads within the given bounding box.
        
        Args:
            min_lat, min_lng, max_lat, max_lng: Bounding box
            road_types: List of OSM highway types to fetch (default: all common types)
        
        Returns:
            List of RoadSegment objects
        """
        cache_key = f"{min_lat:.4f},{min_lng:.4f},{max_lat:.4f},{max_lng:.4f}"
        
        if cache_key in self.cache:
            print(f"[Roads] Using cached road data")
            return self.cache[cache_key]
        
        if road_types is None:
            road_types = self.ROAD_TYPES
        
        # Build Overpass query
        highway_filter = "|".join(road_types)
        query = f"""
        [out:json][timeout:60];
        (
          way["highway"~"^({highway_filter})$"]({min_lat},{min_lng},{max_lat},{max_lng});
        );
        out body;
        >;
        out skel qt;
        """
        
        print(f"[Roads] Fetching roads from OSM for bbox: {min_lat:.4f},{min_lng:.4f} to {max_lat:.4f},{max_lng:.4f}")
        
        # Try each endpoint
        for endpoint in self.OVERPASS_ENDPOINTS:
            try:
                async with httpx.AsyncClient(timeout=90.0) as client:
                    response = await client.post(
                        endpoint,
                        data={"data": query},
                        headers={"Content-Type": "application/x-www-form-urlencoded"}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        segments = self._parse_overpass_response(data)
                        print(f"[Roads] Fetched {len(segments)} road segments")
                        self.cache[cache_key] = segments
                        return segments
                    else:
                        print(f"[Roads] Endpoint {endpoint} returned {response.status_code}")
                        
            except Exception as e:
                print(f"[Roads] Error with {endpoint}: {e}")
                continue
        
        print(f"[Roads] Failed to fetch roads from all endpoints")
        return []
    
    def _parse_overpass_response(self, data: Dict[str, Any]) -> List[RoadSegment]:
        """Parse Overpass API response into RoadSegment objects"""
        segments = []
        
        # First pass: collect all nodes
        nodes: Dict[int, Tuple[float, float]] = {}
        for element in data.get("elements", []):
            if element["type"] == "node":
                nodes[element["id"]] = (element["lat"], element["lon"])
        
        # Second pass: process ways
        for element in data.get("elements", []):
            if element["type"] != "way":
                continue
            
            tags = element.get("tags", {})
            highway_type = tags.get("highway", "unknown")
            name = tags.get("name")
            
            node_ids = element.get("nodes", [])
            
            # Create segments between consecutive nodes
            for i in range(len(node_ids) - 1):
                node1_id = node_ids[i]
                node2_id = node_ids[i + 1]
                
                if node1_id not in nodes or node2_id not in nodes:
                    continue
                
                lat1, lng1 = nodes[node1_id]
                lat2, lng2 = nodes[node2_id]
                
                segment = RoadSegment(
                    start_lat=lat1,
                    start_lng=lng1,
                    end_lat=lat2,
                    end_lng=lng2,
                    road_type=highway_type,
                    name=name
                )
                segments.append(segment)
        
        return segments
    
    def rasterize_roads(
        self,
        segments: List[RoadSegment],
        bounds: Tuple[float, float, float, float],
        rows: int,
        cols: int,
        cell_size_m: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Rasterize road segments into grid arrays.
        
        Args:
            segments: List of RoadSegment objects
            bounds: (min_lat, min_lng, max_lat, max_lng)
            rows, cols: Grid dimensions
            cell_size_m: Cell size in meters
        
        Returns:
            road_mask: Boolean array where True = road present
            road_direction: Float array with road direction in radians (-pi to pi)
            road_distance: Float array with distance to nearest road in meters
        """
        min_lat, min_lng, max_lat, max_lng = bounds
        
        road_mask = np.zeros((rows, cols), dtype=bool)
        road_direction = np.full((rows, cols), np.nan, dtype=np.float32)
        road_distance = np.full((rows, cols), np.inf, dtype=np.float32)
        
        if not segments:
            print(f"[Roads] No road segments to rasterize")
            return road_mask, road_direction, road_distance
        
        print(f"[Roads] Rasterizing {len(segments)} segments into {rows}x{cols} grid")
        
        # Helper to convert lat/lng to grid coordinates
        def latlon_to_grid(lat: float, lng: float) -> Tuple[int, int]:
            norm_lat = (lat - min_lat) / (max_lat - min_lat) if max_lat != min_lat else 0.5
            norm_lng = (lng - min_lng) / (max_lng - min_lng) if max_lng != min_lng else 0.5
            row = int((1 - norm_lat) * (rows - 1))
            col = int(norm_lng * (cols - 1))
            row = max(0, min(row, rows - 1))
            col = max(0, min(col, cols - 1))
            return row, col
        
        def grid_to_latlon(row: int, col: int) -> Tuple[float, float]:
            norm_row = row / (rows - 1) if rows > 1 else 0.5
            norm_col = col / (cols - 1) if cols > 1 else 0.5
            lat = max_lat - norm_row * (max_lat - min_lat)
            lng = min_lng + norm_col * (max_lng - min_lng)
            return lat, lng
        
        # Rasterize each segment using Bresenham-like approach
        for segment in segments:
            r1, c1 = latlon_to_grid(segment.start_lat, segment.start_lng)
            r2, c2 = latlon_to_grid(segment.end_lat, segment.end_lng)
            
            direction = segment.direction_rad
            
            # Draw line between start and end
            dr = abs(r2 - r1)
            dc = abs(c2 - c1)
            steps = max(dr, dc, 1)
            
            for i in range(steps + 1):
                t = i / steps if steps > 0 else 0
                r = int(r1 + t * (r2 - r1))
                c = int(c1 + t * (c2 - c1))
                
                if 0 <= r < rows and 0 <= c < cols:
                    road_mask[r, c] = True
                    road_direction[r, c] = direction
                    road_distance[r, c] = 0.0
        
        # Calculate distance to nearest road for all cells using distance transform
        from scipy.ndimage import distance_transform_edt
        
        # Distance transform gives distance in cells, multiply by cell size for meters
        distance_cells = distance_transform_edt(~road_mask)
        road_distance = distance_cells * cell_size_m
        
        # Propagate road direction to nearby cells using nearest neighbor
        # For cells without a road, find the nearest road cell and use its direction
        if np.any(road_mask):
            from scipy.ndimage import distance_transform_edt
            
            # Get indices of nearest road cell for each non-road cell
            road_indices = np.argwhere(road_mask)
            
            for r in range(rows):
                for c in range(cols):
                    if not road_mask[r, c] and road_distance[r, c] < 200:  # Only within 200m
                        # Find nearest road cell
                        distances = np.sqrt((road_indices[:, 0] - r)**2 + (road_indices[:, 1] - c)**2)
                        nearest_idx = np.argmin(distances)
                        nearest_r, nearest_c = road_indices[nearest_idx]
                        road_direction[r, c] = road_direction[nearest_r, nearest_c]
        
        road_cells = np.sum(road_mask)
        print(f"[Roads] Rasterized: {road_cells} road cells ({100*road_cells/(rows*cols):.1f}% of grid)")
        
        return road_mask, road_direction, road_distance
    
    def angle_difference(self, angle1: float, angle2: float) -> float:
        """
        Calculate the minimum angle difference between two directions.
        
        Args:
            angle1, angle2: Angles in radians
        
        Returns:
            Angle difference in radians (0 to pi)
        """
        diff = abs(angle1 - angle2)
        # Normalize to 0-2pi
        diff = diff % (2 * math.pi)
        # Get minimum difference (considering opposite directions)
        if diff > math.pi:
            diff = 2 * math.pi - diff
        return diff
    
    def is_nearly_parallel(
        self,
        path_direction: float,
        road_direction: float,
        threshold_deg: float = 30.0
    ) -> bool:
        """
        Check if path and road are nearly parallel (within threshold).
        
        Also considers opposite directions as parallel.
        
        Args:
            path_direction: Path heading in radians
            road_direction: Road direction in radians
            threshold_deg: Maximum angle deviation to consider parallel
        
        Returns:
            True if nearly parallel
        """
        if np.isnan(road_direction):
            return False
        
        diff = self.angle_difference(path_direction, road_direction)
        threshold_rad = math.radians(threshold_deg)
        
        # Check if parallel (same direction) or anti-parallel (opposite direction)
        return diff <= threshold_rad or (math.pi - diff) <= threshold_rad


# Global instance
road_service = RoadService()
