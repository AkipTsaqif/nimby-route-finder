"""
Water body detection using OpenStreetMap data.

Fetches actual water bodies (rivers, lakes, sea, etc.) from OSM Overpass API
and creates a raster mask for pathfinding constraints.
"""

import httpx
import numpy as np
from typing import Tuple, List, Optional
import math
import time


# Overpass API endpoints (use multiple for redundancy)
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


def fetch_water_bodies(
    min_lat: float, min_lon: float,
    max_lat: float, max_lon: float,
    timeout: float = 60.0
) -> Optional[dict]:
    """
    Fetch water bodies from OpenStreetMap within a bounding box.
    
    Queries for:
    - natural=water (lakes, ponds, reservoirs)
    - natural=coastline (sea/ocean boundaries)
    - waterway=* (rivers, streams, canals)
    - landuse=reservoir
    - natural=bay
    
    Returns GeoJSON-like dict with water polygons/lines.
    """
    # Overpass QL query for water features
    # We want polygons (ways and relations) for water bodies
    query = f"""
    [out:json][timeout:{int(timeout)}];
    (
      // Water bodies (lakes, ponds, reservoirs)
      way["natural"="water"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["natural"="water"]({min_lat},{min_lon},{max_lat},{max_lon});
      
      // Coastlines (sea/ocean)
      way["natural"="coastline"]({min_lat},{min_lon},{max_lat},{max_lon});
      
      // Waterways (rivers, streams, canals) - get as areas where available
      way["waterway"]["waterway"!="stream"]["waterway"!="ditch"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["waterway"]({min_lat},{min_lon},{max_lat},{max_lon});
      
      // Reservoirs
      way["landuse"="reservoir"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["landuse"="reservoir"]({min_lat},{min_lon},{max_lat},{max_lon});
      
      // Bays
      way["natural"="bay"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["natural"="bay"]({min_lat},{min_lon},{max_lat},{max_lon});
      
      // Sea areas
      way["place"="sea"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["place"="sea"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out body;
    >;
    out skel qt;
    """
    
    print(f"[Water] Fetching water bodies from OSM for bbox: ({min_lat:.4f},{min_lon:.4f}) to ({max_lat:.4f},{max_lon:.4f})")
    
    # Try each endpoint
    last_error = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            print(f"[Water] Trying endpoint: {endpoint}")
            with httpx.Client(timeout=timeout + 10) as client:
                response = client.post(
                    endpoint,
                    data={"data": query},
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    elements = data.get("elements", [])
                    print(f"[Water] Got {len(elements)} OSM elements")
                    return data
                elif response.status_code == 429:
                    print(f"[Water] Rate limited by {endpoint}, trying next...")
                    time.sleep(1)
                    continue
                else:
                    print(f"[Water] Error {response.status_code} from {endpoint}")
                    last_error = f"HTTP {response.status_code}"
                    continue
                    
        except httpx.TimeoutException:
            print(f"[Water] Timeout from {endpoint}")
            last_error = "Timeout"
            continue
        except Exception as e:
            print(f"[Water] Error from {endpoint}: {e}")
            last_error = str(e)
            continue
    
    print(f"[Water] All endpoints failed. Last error: {last_error}")
    return None


def parse_osm_water_geometries(osm_data: dict) -> List[List[Tuple[float, float]]]:
    """
    Parse OSM response into list of polygon coordinates.
    
    Returns list of polygons, where each polygon is a list of (lon, lat) tuples.
    """
    if not osm_data:
        return []
    
    elements = osm_data.get("elements", [])
    
    # Build node lookup
    nodes = {}
    for elem in elements:
        if elem.get("type") == "node":
            nodes[elem["id"]] = (elem["lon"], elem["lat"])
    
    # Extract ways
    polygons = []
    for elem in elements:
        if elem.get("type") == "way":
            node_refs = elem.get("nodes", [])
            if len(node_refs) < 3:
                continue
            
            # Build polygon from node references
            polygon = []
            for node_id in node_refs:
                if node_id in nodes:
                    polygon.append(nodes[node_id])
            
            if len(polygon) >= 3:
                polygons.append(polygon)
    
    # TODO: Handle relations (multipolygons) for complex water bodies
    # For now, we just use ways which covers most cases
    
    print(f"[Water] Parsed {len(polygons)} water polygons from OSM data")
    return polygons


def rasterize_water_mask(
    polygons: List[List[Tuple[float, float]]],
    min_lat: float, min_lon: float,
    max_lat: float, max_lon: float,
    grid_rows: int, grid_cols: int
) -> np.ndarray:
    """
    Convert water polygons to a raster mask.
    
    Uses scanline algorithm to fill polygons.
    
    Returns:
        Boolean numpy array where True = water
    """
    mask = np.zeros((grid_rows, grid_cols), dtype=bool)
    
    if not polygons:
        print(f"[Water] No polygons to rasterize")
        return mask
    
    # Calculate cell sizes
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    
    if lat_range <= 0 or lon_range <= 0:
        return mask
    
    cell_lat = lat_range / grid_rows
    cell_lon = lon_range / grid_cols
    
    def lonlat_to_cell(lon: float, lat: float) -> Tuple[int, int]:
        """Convert lon/lat to grid row/col."""
        col = int((lon - min_lon) / cell_lon)
        row = int((max_lat - lat) / cell_lat)  # Row 0 is at max_lat (north)
        return (row, col)
    
    # Rasterize each polygon
    filled_cells = 0
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        
        # Convert polygon to grid coordinates
        grid_polygon = [lonlat_to_cell(lon, lat) for lon, lat in polygon]
        
        # Simple scanline fill
        # Get bounding box of polygon in grid coords
        rows = [p[0] for p in grid_polygon]
        cols = [p[1] for p in grid_polygon]
        min_row = max(0, min(rows))
        max_row = min(grid_rows - 1, max(rows))
        min_col = max(0, min(cols))
        max_col = min(grid_cols - 1, max(cols))
        
        # Check if polygon is just a line (coastline)
        is_line = not _is_closed_polygon(polygon)
        
        if is_line:
            # For coastlines/lines, just draw the line with some width
            _rasterize_line(mask, grid_polygon, width=2)
        else:
            # For closed polygons, fill the interior
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    if _point_in_polygon(row, col, grid_polygon):
                        if 0 <= row < grid_rows and 0 <= col < grid_cols:
                            if not mask[row, col]:
                                mask[row, col] = True
                                filled_cells += 1
    
    water_pct = (filled_cells / mask.size) * 100
    print(f"[Water] Rasterized water mask: {filled_cells:,} cells ({water_pct:.1f}%)")
    return mask


def _is_closed_polygon(polygon: List[Tuple[float, float]]) -> bool:
    """Check if polygon is closed (first point == last point)."""
    if len(polygon) < 3:
        return False
    first = polygon[0]
    last = polygon[-1]
    # Check if within ~1m (roughly 0.00001 degrees)
    return abs(first[0] - last[0]) < 0.0001 and abs(first[1] - last[1]) < 0.0001


def _rasterize_line(
    mask: np.ndarray,
    points: List[Tuple[int, int]],
    width: int = 1
) -> None:
    """Draw a line on the mask using Bresenham's algorithm with width."""
    rows, cols = mask.shape
    
    for i in range(len(points) - 1):
        r0, c0 = points[i]
        r1, c1 = points[i + 1]
        
        # Bresenham's line algorithm
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc
        
        r, c = r0, c0
        while True:
            # Draw with width
            for wr in range(-width, width + 1):
                for wc in range(-width, width + 1):
                    nr, nc = r + wr, c + wc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        mask[nr, nc] = True
            
            if r == r1 and c == c1:
                break
            
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc


def _point_in_polygon(row: int, col: int, polygon: List[Tuple[int, int]]) -> bool:
    """
    Check if point is inside polygon using ray casting algorithm.
    """
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        ri, ci = polygon[i]
        rj, cj = polygon[j]
        
        if ((ci > col) != (cj > col)) and \
           (row < (rj - ri) * (col - ci) / (cj - ci + 0.0001) + ri):
            inside = not inside
        
        j = i
    
    return inside


def create_water_mask_from_osm(
    min_lat: float, min_lon: float,
    max_lat: float, max_lon: float,
    grid_rows: int, grid_cols: int,
    timeout: float = 60.0
) -> Tuple[np.ndarray, bool]:
    """
    Create a water mask for the given bounding box using OSM data.
    
    Returns:
        Tuple of (mask, success) where mask is boolean array and success
        indicates if OSM data was successfully fetched.
    """
    # Fetch water data from OSM
    osm_data = fetch_water_bodies(min_lat, min_lon, max_lat, max_lon, timeout)
    
    if osm_data is None:
        print(f"[Water] Failed to fetch OSM data, returning empty mask")
        return np.zeros((grid_rows, grid_cols), dtype=bool), False
    
    # Parse geometries
    polygons = parse_osm_water_geometries(osm_data)
    
    if not polygons:
        print(f"[Water] No water polygons found in area")
        return np.zeros((grid_rows, grid_cols), dtype=bool), True
    
    # Rasterize to mask
    mask = rasterize_water_mask(
        polygons,
        min_lat, min_lon, max_lat, max_lon,
        grid_rows, grid_cols
    )
    
    return mask, True


# For testing
if __name__ == "__main__":
    # Test with a small area
    mask, success = create_water_mask_from_osm(
        min_lat=-6.3, min_lon=105.7,
        max_lat=-6.1, max_lon=106.0,
        grid_rows=100, grid_cols=100
    )
    print(f"Success: {success}")
    print(f"Water cells: {np.sum(mask)} / {mask.size}")
