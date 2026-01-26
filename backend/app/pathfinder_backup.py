"""
Constrained A* Pathfinding for Railway Routes

State-lattice planning with:
- Maximum slope constraint (soft constraint with penalty for slight overages)
- Minimum curve radius constraint
- Elevation-aware cost function
- Water body avoidance
- Bidirectional search for better performance
"""

import heapq
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Set
from enum import IntEnum
from scipy.ndimage import uniform_filter, label


class Heading(IntEnum):
    """8 discrete heading directions (45° increments)"""
    N = 0    # North (up)
    NE = 1   # Northeast
    E = 2    # East (right)
    SE = 3   # Southeast
    S = 4    # South (down)
    SW = 5   # Southwest
    W = 6    # West (left)
    NW = 7   # Northwest


# Direction vectors for each heading (row_delta, col_delta)
DIRECTION_VECTORS = {
    Heading.N:  (-1, 0),
    Heading.NE: (-1, 1),
    Heading.E:  (0, 1),
    Heading.SE: (1, 1),
    Heading.S:  (1, 0),
    Heading.SW: (1, -1),
    Heading.W:  (0, -1),
    Heading.NW: (-1, -1),
}


@dataclass
class PathfindingConfig:
    """Configuration for the pathfinding algorithm"""
    max_slope_percent: float = 3.0  # Maximum grade (rise/run * 100)
    min_curve_radius_m: float = 500.0  # Minimum curve radius
    cell_size_m: float = 30.0  # Meters per grid cell (from DEM)
    
    # Cost weights
    distance_weight: float = 1.0
    elevation_change_weight: float = 1.5  # Penalty for climbing (reduced from 2.0)
    curvature_weight: float = 0.3  # Base penalty for turning
    
    # Curvature cost scaling - makes sharper turns progressively more expensive
    # This encourages larger radii even when smaller ones are allowed
    curvature_exponential: float = 2.0  # Exponent for curvature penalty (higher = prefer gentler curves)
    heading_consistency_bonus: float = 0.1  # Bonus per step for maintaining same heading
    max_consistency_bonus_steps: int = 10  # Max steps to accumulate consistency bonus
    
    # Soft constraint settings - allow slightly exceeding slope with heavy penalty
    hard_slope_limit_percent: float = 8.0  # Absolute maximum, never exceed
    slope_penalty_multiplier: float = 20.0  # Penalty for exceeding soft limit (reduced from 50)
    
    # Water avoidance
    water_penalty: float = 10000.0  # Very high cost to cross water
    water_elevation_threshold: float = 1.0  # Cells at or below this are suspicious (sea level is ~0m)
    water_flatness_threshold: float = 0.1  # Max elevation variance for water detection (water is VERY flat)
    
    # Search settings
    allow_steep_fallback: bool = True  # If no path found, try with relaxed constraints
    
    # Switchback control - mainline railways rarely use switchbacks
    allow_switchbacks: bool = False  # If False, heavily penalize direction reversals
    switchback_penalty: float = 5000.0  # High penalty for switchbacks (180° turns)
    min_switchback_interval: int = 50  # Minimum cells between switchbacks if allowed
    
    # Auto tunnel/bridge detection
    auto_tunnel_bridge: bool = False  # Enable automatic tunnel/bridge detection
    max_jump_distance_m: float = 500.0  # Maximum distance to search for similar elevation
    elevation_tolerance_m: float = 10.0  # Max elevation difference for auto-detection
    
    # Road parallelism constraints
    road_parallel_enabled: bool = False  # Enable road parallel constraints
    road_parallel_threshold_deg: float = 30.0  # Angle threshold for "nearly parallel" (within this of road direction)
    road_min_separation_m: float = 10.0  # Minimum separation from road when parallel
    road_max_separation_m: float = 50.0  # Maximum separation to apply parallel constraint
    road_parallel_penalty: float = 500.0  # Penalty for violating parallel/separation constraint
    
    def max_heading_change(self) -> int:
        """
        Calculate maximum allowed heading change based on min curve radius.
        At each step, we can only turn so much given our speed and radius constraint.
        
        For a grid cell of size d, turning by angle θ implies radius r ≈ d / (2 * sin(θ/2))
        So max θ ≈ 2 * arcsin(d / (2 * r))
        """
        if self.min_curve_radius_m <= 0:
            return 7  # Allow any turn
        
        half_angle = self.cell_size_m / (2 * self.min_curve_radius_m)
        if half_angle >= 1:
            return 7  # Grid too coarse, allow any turn
        
        max_angle_rad = 2 * math.asin(half_angle)
        max_angle_deg = math.degrees(max_angle_rad)
        
        # Each heading step is 45°
        max_steps = int(max_angle_deg / 45)
        # Allow at least 3 heading changes (135°) when cell size is large
        # This is critical for navigating mountainous terrain where we need
        # to make significant turns to follow contours
        min_heading_change = 3 if self.cell_size_m > 80 else (2 if self.cell_size_m > 50 else 1)
        return max(min_heading_change, min(max_steps, 7))

@dataclass(frozen=True)
class State:
    """A state in the search space: position + heading + initial heading for switchback detection"""
    row: int
    col: int
    heading: Heading
    initial_heading: Heading = None  # Track the heading when we started this "segment"
    steps_in_segment: int = 0  # Steps since last significant direction change
    
    def __lt__(self, other):
        # For heap ordering - only use position and current heading
        return (self.row, self.col, self.heading) < (other.row, other.col, other.heading)
    
    def __hash__(self):
        # Hash only on position and current heading to avoid state explosion
        return hash((self.row, self.col, self.heading))
    
    def __eq__(self, other):
        # Equality only on position and current heading
        if not isinstance(other, State):
            return False
        return self.row == other.row and self.col == other.col and self.heading == other.heading


class ConstrainedAStar:
    """
    A* pathfinder with railway-specific constraints:
    - Slope constraint: soft limit with penalty, hard maximum
    - Curvature constraint: limit heading change per step
    - Water avoidance: detect and penalize water bodies
    - Tunnel zones: ignore slope constraints
    - Bridge zones: allow water crossing
    """
    
    def __init__(
        self,
        elevation_grid: np.ndarray,
        bounds: Tuple[float, float, float, float],  # (min_lat, min_lng, max_lat, max_lng)
        transform: Dict[str, float],  # Grid transform parameters
        config: PathfindingConfig,
        road_mask: Optional[np.ndarray] = None,
        road_direction: Optional[np.ndarray] = None,
        road_distance: Optional[np.ndarray] = None
    ):
        self.elevation = elevation_grid
        self.bounds = bounds
        self.transform = transform
        self.config = config
        self.rows, self.cols = elevation_grid.shape
        
        # Precompute max heading change
        self.max_heading_change = config.max_heading_change()
        
        # Detect water bodies
        self.water_mask = self._detect_water_bodies()
        water_cells = np.sum(self.water_mask)
        if water_cells > 0:
            print(f"[Pathfinder] Detected {water_cells} water cells ({100*water_cells/(self.rows*self.cols):.1f}% of grid)")
        
        # Road data (optional)
        self.road_mask = road_mask if road_mask is not None else np.zeros((self.rows, self.cols), dtype=bool)
        self.road_direction = road_direction if road_direction is not None else np.full((self.rows, self.cols), np.nan, dtype=np.float32)
        self.road_distance = road_distance if road_distance is not None else np.full((self.rows, self.cols), np.inf, dtype=np.float32)
        
        if road_mask is not None and np.any(road_mask):
            print(f"[Pathfinder] Road data loaded: {np.sum(road_mask)} road cells")
        
        # Tunnel and bridge zone masks
        # Manual masks are for user-defined tunnel/bridge zones
        # Auto masks are for pathfinder-detected structures (visualization only, no slope relaxation)
        self.manual_tunnel_mask = np.zeros((self.rows, self.cols), dtype=bool)
        self.manual_bridge_mask = np.zeros((self.rows, self.cols), dtype=bool)
        self.auto_tunnel_mask = np.zeros((self.rows, self.cols), dtype=bool)
        self.auto_bridge_mask = np.zeros((self.rows, self.cols), dtype=bool)
        self.tunnel_zones: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        self.bridge_zones: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        self.auto_tunnel_segments: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        self.auto_bridge_segments: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    
    def add_tunnel_zone(self, entry: Tuple[float, float], exit: Tuple[float, float]):
        """
        Add a MANUAL tunnel zone between two lat/lng points.
        Within this zone, slope constraints are relaxed.
        """
        entry_grid = self.latlon_to_grid(entry[0], entry[1])
        exit_grid = self.latlon_to_grid(exit[0], exit[1])
        
        # Mark cells along the corridor as manual tunnel zone
        self._mark_corridor(self.manual_tunnel_mask, entry_grid, exit_grid, width=3)
        self.tunnel_zones.append((entry_grid, exit_grid))
        print(f"[Pathfinder] Added manual tunnel zone: {entry_grid} -> {exit_grid}")
    
    def add_bridge_zone(self, start: Tuple[float, float], end: Tuple[float, float]):
        """
        Add a MANUAL bridge zone between two lat/lng points.
        Within this zone, water crossing is allowed.
        """
        start_grid = self.latlon_to_grid(start[0], start[1])
        end_grid = self.latlon_to_grid(end[0], end[1])
        
        # Mark cells along the corridor as manual bridge zone
        self._mark_corridor(self.manual_bridge_mask, start_grid, end_grid, width=3)
        self.bridge_zones.append((start_grid, end_grid))
        print(f"[Pathfinder] Added manual bridge zone: {start_grid} -> {end_grid}")
    
    def _mark_corridor(self, mask: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], width: int = 3):
        """Mark cells along a corridor between two points"""
        r1, c1 = start
        r2, c2 = end
        
        # Use Bresenham-like approach to mark cells along the line
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        steps = max(dr, dc, 1)
        
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            r = int(r1 + t * (r2 - r1))
            c = int(c1 + t * (c2 - c1))
            
            # Mark a width x width area around each point
            half_w = width // 2
            for dr in range(-half_w, half_w + 1):
                for dc in range(-half_w, half_w + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.rows and 0 <= cc < self.cols:
                        mask[rr, cc] = True
    
    def is_tunnel_zone(self, row: int, col: int) -> bool:
        """Check if a cell is in ANY tunnel zone (manual or auto)"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return bool(self.manual_tunnel_mask[row, col]) or bool(self.auto_tunnel_mask[row, col])
        return False
    
    def is_manual_tunnel_zone(self, row: int, col: int) -> bool:
        """Check if a cell is in a MANUAL tunnel zone (for slope relaxation)"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return bool(self.manual_tunnel_mask[row, col])
        return False
    
    def is_bridge_zone(self, row: int, col: int) -> bool:
        """Check if a cell is in ANY bridge zone (manual or auto)"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return bool(self.manual_bridge_mask[row, col]) or bool(self.auto_bridge_mask[row, col])
        return False
    
    def is_manual_bridge_zone(self, row: int, col: int) -> bool:
        """Check if a cell is in a MANUAL bridge zone (for water crossing)"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return bool(self.manual_bridge_mask[row, col])
        return False
    
    def find_jump_candidates(self, row: int, col: int, heading: Heading) -> List[Tuple[Tuple[int, int], str, float]]:
        """
        Find potential tunnel/bridge endpoints when normal pathfinding is blocked.
        
        Searches within max_jump_distance_m for cells at similar elevation,
        preferring to continue in the current heading direction.
        
        Returns:
            List of ((row, col), type, cost) tuples where type is 'tunnel' or 'bridge'
        """
        if not self.config.auto_tunnel_bridge:
            return []
        
        candidates = []
        current_elev = self.get_elevation(row, col)
        
        if current_elev == float('inf'):
            return []
        
        # Calculate search radius in grid cells
        max_cells = int(self.config.max_jump_distance_m / self.config.cell_size_m)
        
        # Get the direction vector for current heading
        primary_dr, primary_dc = DIRECTION_VECTORS[heading]
        
        # Search in a cone in the direction of travel (favor forward movement)
        for distance in range(2, max_cells + 1):
            # Check cells at this distance, prioritizing forward direction
            for angle_offset in range(-2, 3):  # -90° to +90° from heading
                check_heading = Heading((heading + angle_offset) % 8)
                dr, dc = DIRECTION_VECTORS[check_heading]
                
                check_row = row + dr * distance
                check_col = col + dc * distance
                
                if not (0 <= check_row < self.rows and 0 <= check_col < self.cols):
                    continue
                
                target_elev = self.get_elevation(check_row, check_col)
                if target_elev == float('inf'):
                    continue
                
                elev_diff = abs(target_elev - current_elev)
                
                # Check if elevation is within tolerance
                if elev_diff > self.config.elevation_tolerance_m:
                    continue
                
                # Determine if this should be a tunnel or bridge
                # Check terrain between current and target
                is_water_crossing = False
                is_mountain_crossing = False
                min_elev_between = float('inf')
                max_elev_between = float('-inf')
                
                for t in range(1, distance):
                    inter_row = row + int(dr * t)
                    inter_col = col + int(dc * t)
                    
                    if 0 <= inter_row < self.rows and 0 <= inter_col < self.cols:
                        inter_elev = self.get_elevation(inter_row, inter_col)
                        min_elev_between = min(min_elev_between, inter_elev)
                        max_elev_between = max(max_elev_between, inter_elev)
                        
                        if self.is_water(inter_row, inter_col):
                            is_water_crossing = True
                
                # Determine structure type based on terrain
                if is_water_crossing or min_elev_between < current_elev - 5:
                    structure_type = 'bridge'
                elif max_elev_between > current_elev + 5:
                    structure_type = 'tunnel'
                else:
                    # Could be either - use the one that makes more sense
                    if max_elev_between > current_elev:
                        structure_type = 'tunnel'
                    else:
                        structure_type = 'bridge'
                
                # Calculate cost (longer = more expensive, angle deviation = more expensive)
                actual_distance = distance * self.config.cell_size_m
                angle_penalty = 1.0 + abs(angle_offset) * 0.2  # Penalty for turning
                
                if structure_type == 'tunnel':
                    cost_per_m = 100.0  # Tunnels are very expensive
                else:
                    cost_per_m = 150.0  # Bridges can be even more expensive
                
                total_cost = actual_distance * cost_per_m * angle_penalty
                
                candidates.append(((check_row, check_col), structure_type, total_cost))
        
        # Sort by cost
        candidates.sort(key=lambda x: x[2])
        
        # Return top candidates (limit to prevent explosion)
        return candidates[:5]

    def _detect_water_bodies(self) -> np.ndarray:
        """
        Detect water bodies based on elevation patterns.
        Water appears as large flat areas at or below sea level.
        
        Key insight: Real water bodies are EXTREMELY flat (variance near 0) and
        at consistent elevation (typically 0m for sea, or a specific level for lakes).
        Coastal plains and beaches have more variance and slight elevation gradients.
        
        Returns a boolean mask where True = water (avoid)
        """
        try:
            print(f"[Water Detection] Analyzing elevation grid (range: {self.elevation.min():.1f}m to {self.elevation.max():.1f}m)")
            
            # Criterion 1: Very low elevation (at or near sea level)
            # Mapbox typically returns ~0m or slightly negative for sea
            very_low = self.elevation <= 0.5  # Sea level with small buffer
            low_elevation = self.elevation <= self.config.water_elevation_threshold  # User-configurable
            
            # Criterion 2: EXTREMELY flat areas (near-zero local variance)
            window_size = max(3, min(11, self.rows // 20, self.cols // 20))
            if window_size % 2 == 0:
                window_size += 1
            
            mean = uniform_filter(self.elevation.astype(np.float64), size=window_size)
            mean_sq = uniform_filter((self.elevation.astype(np.float64))**2, size=window_size)
            variance = mean_sq - mean**2
            variance = np.maximum(variance, 0)
            
            flat_areas = variance < self.config.water_flatness_threshold
            
            # Criterion 3: Near-zero gradient
            grad_row = np.gradient(self.elevation, axis=0)
            grad_col = np.gradient(self.elevation, axis=1)
            gradient_magnitude = np.sqrt(grad_row**2 + grad_col**2)
            zero_gradient = gradient_magnitude < 0.1
            
            # Use a more aggressive detection: 
            # Option A: At or below sea level (<=0m) - DEFINITELY water, no other criteria needed
            # Option B: Very low elevation (<=0.5m) AND flat - almost certainly water
            # Option C: Low elevation + flat + zero gradient - likely water
            definite_water = self.elevation <= 0.0  # Sea level or below = water
            certain_water = very_low & flat_areas  # Very low AND flat
            likely_water = low_elevation & flat_areas & zero_gradient  # All criteria
            
            potential_water = definite_water | certain_water | likely_water
            
            # Criterion 4: Must be part of a reasonably large connected region
            labeled, num_features = label(potential_water)
            min_water_size = max(50, (self.rows * self.cols) // 200)  # At least 0.5% of grid or 50 cells
            
            water_mask = np.zeros_like(potential_water)
            for i in range(1, num_features + 1):
                region = labeled == i
                region_size = np.sum(region)
                if region_size >= min_water_size:
                    # Additional check: consistent elevation (std < 1.0m for water)
                    region_elevations = self.elevation[region]
                    region_std = np.std(region_elevations)
                    if region_std < 1.0:
                        water_mask[region] = True
            
            # Add a buffer zone around water bodies to prevent coastal edge-hugging
            # Also mark low-elevation cells adjacent to water as water
            from scipy.ndimage import binary_dilation
            
            # Dilate water mask by 2 cells to create buffer
            structure = np.ones((3, 3), dtype=bool)  # 3x3 kernel for dilation
            water_buffer = binary_dilation(water_mask, structure=structure, iterations=2)
            
            # Only apply buffer to low-elevation cells (< 3m) to avoid marking hills as water
            low_elev_buffer = (water_buffer & (self.elevation < 3.0))
            
            # Combine original water with low-elevation buffer
            final_water_mask = water_mask | low_elev_buffer
            
            buffer_added = np.sum(final_water_mask) - np.sum(water_mask)
            print(f"[Water Detection] Added {buffer_added} buffer cells around water bodies")
            
            return final_water_mask
            
        except Exception as e:
            print(f"[Pathfinder] Water detection failed: {e}, disabling water avoidance")
            return np.zeros((self.rows, self.cols), dtype=bool)
    
    def latlon_to_grid(self, lat: float, lng: float) -> Tuple[int, int]:
        """Convert lat/lon to grid coordinates"""
        min_lat, min_lng, max_lat, max_lng = self.bounds
        
        # Normalize to 0-1
        norm_lat = (lat - min_lat) / (max_lat - min_lat) if max_lat != min_lat else 0.5
        norm_lng = (lng - min_lng) / (max_lng - min_lng) if max_lng != min_lng else 0.5
        
        # Convert to grid (note: row 0 is at max_lat, so invert lat)
        row = int((1 - norm_lat) * (self.rows - 1))
        col = int(norm_lng * (self.cols - 1))
        
        # Clamp to valid range
        row = max(0, min(row, self.rows - 1))
        col = max(0, min(col, self.cols - 1))
        
        return row, col
    
    def grid_to_latlon(self, row: float, col: float) -> Tuple[float, float]:
        """Convert grid coordinates (int or float) to lat/lon"""
        min_lat, min_lng, max_lat, max_lng = self.bounds
        
        norm_row = row / (self.rows - 1) if self.rows > 1 else 0.5
        norm_col = col / (self.cols - 1) if self.cols > 1 else 0.5
        
        lat = max_lat - norm_row * (max_lat - min_lat)
        lng = min_lng + norm_col * (max_lng - min_lng)
        
        return lat, lng
    
    def get_elevation(self, row: int, col: int) -> float:
        """Get elevation at grid position"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return float(self.elevation[row, col])
        return float('inf')  # Out of bounds
    
    def _interpolate_elevation(self, row: float, col: float) -> float:
        """Get interpolated elevation at float grid position using bilinear interpolation"""
        row_int = int(row)
        col_int = int(col)
        row_frac = row - row_int
        col_frac = col - col_int
        
        # Handle edge cases
        if row_int < 0 or col_int < 0:
            return float('inf')
        if row_int >= self.rows - 1 or col_int >= self.cols - 1:
            # At or beyond edge, use nearest valid cell
            r = min(max(0, int(round(row))), self.rows - 1)
            c = min(max(0, int(round(col))), self.cols - 1)
            return float(self.elevation[r, c])
        
        # Bilinear interpolation
        e00 = float(self.elevation[row_int, col_int])
        e01 = float(self.elevation[row_int, col_int + 1])
        e10 = float(self.elevation[row_int + 1, col_int])
        e11 = float(self.elevation[row_int + 1, col_int + 1])
        
        # Interpolate along columns first
        e0 = e00 * (1 - col_frac) + e01 * col_frac
        e1 = e10 * (1 - col_frac) + e11 * col_frac
        
        # Then interpolate along rows
        return e0 * (1 - row_frac) + e1 * row_frac
    
    def calculate_slope(self, from_row: int, from_col: int, to_row: int, to_col: int) -> float:
        """Calculate slope percentage between two grid cells"""
        elev_from = self.get_elevation(from_row, from_col)
        elev_to = self.get_elevation(to_row, to_col)
        
        if elev_from == float('inf') or elev_to == float('inf'):
            return float('inf')
        
        # Calculate horizontal distance
        d_row = to_row - from_row
        d_col = to_col - from_col
        distance_cells = math.sqrt(d_row**2 + d_col**2)
        distance_m = distance_cells * self.config.cell_size_m
        
        if distance_m == 0:
            return 0.0
        
        # Slope as percentage
        elevation_change = elev_to - elev_from
        return (elevation_change / distance_m) * 100
    
    def heading_difference(self, h1: Heading, h2: Heading) -> int:
        """Calculate the minimum steps between two headings (0-4)"""
        diff = abs(int(h1) - int(h2))
        return min(diff, 8 - diff)
    
    def is_water(self, row: int, col: int) -> bool:
        """Check if a cell is detected as water"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return bool(self.water_mask[row, col])
        return False
    
    def get_neighbors(self, state: State, use_soft_slope: bool = True) -> List[Tuple[State, float]]:
        """
        Get valid neighboring states with their transition costs.
        Applies slope and curvature constraints.
        
        Args:
            use_soft_slope: If True, allow exceeding max_slope_percent up to hard_slope_limit
                           with a heavy penalty. If False, use hard constraint.
        """
        neighbors = []
        
        for new_heading in Heading:
            # Check curvature constraint
            heading_diff = self.heading_difference(state.heading, new_heading)
            if heading_diff > self.max_heading_change:
                continue
            
            # Get movement direction
            d_row, d_col = DIRECTION_VECTORS[new_heading]
            new_row = state.row + d_row
            new_col = state.col + d_col
            
            # Check bounds
            if not (0 <= new_row < self.rows and 0 <= new_col < self.cols):
                continue
            
            # Check if we're in a MANUALLY DEFINED tunnel zone (slope constraints relaxed)
            # Auto-detected tunnels don't relax slope - they just mark the path for visualization
            in_manual_tunnel = self.is_manual_tunnel_zone(state.row, state.col) and self.is_manual_tunnel_zone(new_row, new_col)
            
            # Check if we're in a MANUAL bridge zone (water crossing allowed)
            # Only manual bridges allow water crossing
            in_manual_bridge = self.is_manual_bridge_zone(state.row, state.col) and self.is_manual_bridge_zone(new_row, new_col)
            
            # Check slope constraint (relaxed only in MANUAL tunnel zones)
            slope = self.calculate_slope(state.row, state.col, new_row, new_col)
            abs_slope = abs(slope)
            
            # Only manual tunnel zones allow steeper grades (for rack railways etc)
            effective_hard_limit = 15.0 if in_manual_tunnel else self.config.hard_slope_limit_percent
            
            # Hard limit - never exceed this
            if abs_slope > effective_hard_limit:
                continue
            
            # Calculate base cost
            distance = math.sqrt(d_row**2 + d_col**2) * self.config.cell_size_m
            elevation_change = abs(self.get_elevation(new_row, new_col) - self.get_elevation(state.row, state.col))
            
            # Progressive curvature cost: exponential penalty for sharper turns
            # heading_diff: 0=straight, 1=45°, 2=90°, 3=135°, 4=180°
            # Use exponential scaling so slight curves are cheap, sharp curves are expensive
            if heading_diff == 0:
                curvature_cost = 0.0
                # Heading consistency bonus: reward going straight
                # Accumulates up to max_consistency_bonus_steps
                consistency_steps = min(state.steps_in_segment, self.config.max_consistency_bonus_steps)
                heading_consistency_bonus = -self.config.heading_consistency_bonus * distance * (1 + consistency_steps * 0.1)
            else:
                # Exponential curvature cost: penalty grows rapidly with sharpness
                # cost = base * (exponent^heading_diff - 1) 
                # With exponent=2: 0->0, 1->1, 2->3, 3->7, 4->15
                curvature_cost = self.config.curvature_weight * distance * (
                    self.config.curvature_exponential ** heading_diff - 1.0
                ) * 2.0
                heading_consistency_bonus = 0.0
            
            cost = (
                self.config.distance_weight * distance +
                self.config.elevation_change_weight * elevation_change +
                curvature_cost +
                heading_consistency_bonus
            )
            
            # Add tunnel construction cost (tunnels are expensive!)
            if in_manual_tunnel:
                tunnel_cost_per_m = 50.0  # High cost for tunnel construction
                cost += tunnel_cost_per_m * distance
            
            # Soft slope penalty: if exceeding max_slope_percent but under hard limit
            # (not applied in manual tunnel zones where steep grades are expected)
            if not in_manual_tunnel:
                if use_soft_slope and abs_slope > self.config.max_slope_percent:
                    slope_overage = abs_slope - self.config.max_slope_percent
                    slope_penalty = slope_overage * self.config.slope_penalty_multiplier * distance
                    cost += slope_penalty
                elif not use_soft_slope and abs_slope > self.config.max_slope_percent:
                    # Hard constraint mode - skip this neighbor
                    continue
            
            # Water penalty: heavily discourage crossing water (unless in MANUAL bridge zone)
            if self.is_water(new_row, new_col):
                if in_manual_bridge:
                    # Manual bridge zone: allow water crossing with bridge construction cost
                    bridge_cost_per_m = 100.0  # Very high cost for bridge construction
                    cost += bridge_cost_per_m * distance
                else:
                    # Not in bridge zone - SKIP this neighbor entirely (hard constraint)
                    continue
            
            # Switchback detection and penalty
            # A switchback is a 180° turn (heading_diff == 4)
            is_switchback = heading_diff == 4
            
            if is_switchback:
                if not self.config.allow_switchbacks:
                    # Switchbacks not allowed - SKIP entirely (hard constraint)
                    continue
                else:
                    # Switchbacks allowed - add moderate penalty
                    cost += self.config.switchback_penalty * 0.3
            
            # Track progressive reversal (zig-zag pattern detection)
            # If we have an initial heading and have been moving for a few steps,
            # check if we're now going backwards from where we started
            initial_heading = state.initial_heading if state.initial_heading is not None else state.heading
            steps_in_segment = state.steps_in_segment
            
            # After N steps, check if our new heading is significantly opposite to initial
            # This catches patterns like E→NE→N→NW→W which is a progressive reversal
            if steps_in_segment >= 2:
                reversal_amount = self.heading_difference(initial_heading, new_heading)
                if reversal_amount >= 4:  # 180° from initial
                    if not self.config.allow_switchbacks:
                        # Progressive switchback - SKIP (hard constraint)
                        continue
                    else:
                        # Add penalty for progressive reversal
                        cost += self.config.switchback_penalty * 0.5
            
            # Also penalize sharp turns (135°, heading_diff == 3)
            if heading_diff == 3:
                cost += self.config.switchback_penalty * 0.15
            
            # Update segment tracking:
            # If we're turning significantly (>90°), reset the segment
            if heading_diff >= 2:
                # Significant turn - start new segment
                new_initial = new_heading
                new_steps = 1
            else:
                # Continuing roughly same direction
                new_initial = initial_heading
                new_steps = steps_in_segment + 1
            
            # Road parallelism constraint
            # If enabled and we're near a road and nearly parallel, enforce constraints
            if self.config.road_parallel_enabled:
                road_dist = self.road_distance[new_row, new_col]
                road_dir = self.road_direction[new_row, new_col]
                
                # Check if we're within the influence zone of a road
                if road_dist <= self.config.road_max_separation_m and not np.isnan(road_dir):
                    # Convert heading to radians for comparison
                    # Heading: N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7
                    # Each step is 45 degrees, measured clockwise from North
                    path_direction_rad = (int(new_heading) * 45) * (math.pi / 180)
                    
                    # Calculate angle difference (considering both directions as parallel)
                    angle_diff = abs(path_direction_rad - road_dir)
                    angle_diff = angle_diff % (2 * math.pi)
                    if angle_diff > math.pi:
                        angle_diff = 2 * math.pi - angle_diff
                    # Also check anti-parallel (opposite direction is still parallel for a road)
                    anti_parallel_diff = abs(angle_diff - math.pi)
                    
                    min_angle_diff = min(angle_diff, anti_parallel_diff)
                    threshold_rad = self.config.road_parallel_threshold_deg * (math.pi / 180)
                    
                    is_nearly_parallel = min_angle_diff <= threshold_rad
                    
                    if is_nearly_parallel:
                        # We're nearly parallel to the road
                        # Check separation constraint
                        if road_dist < self.config.road_min_separation_m:
                            # Too close to the road - add heavy penalty or skip
                            cost += self.config.road_parallel_penalty * 2
                        elif road_dist > self.config.road_max_separation_m:
                            # Too far from road when parallel - small penalty
                            cost += self.config.road_parallel_penalty * 0.5
                        
                        # Bonus: If at ideal separation and parallel, reduce cost slightly
                        # This encourages the path to stay parallel at proper distance
                        ideal_separation = (self.config.road_min_separation_m + self.config.road_max_separation_m) / 2
                        if abs(road_dist - ideal_separation) < 10:
                            cost -= self.config.road_parallel_penalty * 0.3
                    else:
                        # Not parallel but near road
                        # If within min separation and not parallel, that's worse
                        if road_dist < self.config.road_min_separation_m:
                            cost += self.config.road_parallel_penalty
            
            new_state = State(new_row, new_col, new_heading, new_initial, new_steps)
            neighbors.append((new_state, cost))
        
        # If no valid neighbors found and auto tunnel/bridge is enabled,
        # try to find jump candidates (tunnels/bridges)
        if len(neighbors) == 0 and self.config.auto_tunnel_bridge:
            jump_candidates = self.find_jump_candidates(state.row, state.col, state.heading)
            for (jump_row, jump_col), structure_type, jump_cost in jump_candidates:
                # Determine the heading toward the jump target
                d_row = jump_row - state.row
                d_col = jump_col - state.col
                new_heading = self._direction_to_heading(d_row, d_col)
                
                # NOTE: We don't mark corridors here anymore - we'll identify
                # tunnels/bridges in the final path during path_to_geojson
                
                # Jump resets the segment tracking (new direction after tunnel/bridge)
                new_state = State(jump_row, jump_col, new_heading, new_heading, 0)
                neighbors.append((new_state, jump_cost))
        
        return neighbors
    
    def get_auto_structures(self) -> Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """
        Get the automatically detected tunnel and bridge segments.
        Returns dict with 'tunnels' and 'bridges' lists of (start, end) grid coordinate pairs.
        """
        return {
            'auto_tunnels': self.auto_tunnel_segments,
            'auto_bridges': self.auto_bridge_segments,
            'manual_tunnels': self.tunnel_zones,
            'manual_bridges': self.bridge_zones
        }
    
    def heuristic(self, state: State, goal_row: int, goal_col: int, goal_elev: float = None) -> float:
        """
        Admissible heuristic that accounts for both distance and elevation difference.
        
        For railway routing, we need to consider that climbing/descending adds path length
        due to the maximum grade constraint.
        
        Uses hard_slope_limit for the heuristic to ensure admissibility (underestimate).
        """
        d_row = goal_row - state.row
        d_col = goal_col - state.col
        horizontal_dist_cells = math.sqrt(d_row**2 + d_col**2)
        horizontal_dist_m = horizontal_dist_cells * self.config.cell_size_m
        
        # If we have goal elevation, compute minimum path length considering grade limit
        if goal_elev is not None:
            current_elev = self.get_elevation(state.row, state.col)
            elev_diff = abs(goal_elev - current_elev)
            
            # Use HARD slope limit for heuristic to ensure admissibility
            # This gives a tighter (but still admissible) estimate
            min_horizontal_for_elev = elev_diff / (self.config.hard_slope_limit_percent / 100)
            
            # The true minimum distance is the larger of: direct distance or grade-limited distance
            effective_dist = max(horizontal_dist_m, min_horizontal_for_elev)
            
            return effective_dist * self.config.distance_weight
        
        return horizontal_dist_m * self.config.distance_weight
    
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        start_heading: Optional[Heading] = None,
        goal_heading: Optional[Heading] = None
    ) -> Tuple[Optional[List[Tuple[int, int]]], Dict[str, Any]]:
        """
        Find a path from start to goal using constrained A*.
        
        Uses soft slope constraints: prefers staying under max_slope_percent
        but allows up to hard_slope_limit_percent with heavy penalty.
        
        Args:
            start: Starting position (row, col)
            goal: Goal position (row, col)
            start_heading: Optional fixed heading at start point (e.g., from previous waypoint)
            goal_heading: Optional required heading at goal point (e.g., toward next waypoint)
        
        Returns:
            path: List of (row, col) tuples, or None if no path found
            stats: Dictionary with search statistics
        """
        start_row, start_col = start
        goal_row, goal_col = goal
        
        print(f"[Pathfinder] Grid size: {self.rows}x{self.cols}")
        print(f"[Pathfinder] Start: ({start_row}, {start_col}), Goal: ({goal_row}, {goal_col})")
        if start_heading is not None:
            print(f"[Pathfinder] Start heading constraint: {start_heading.name}")
        if goal_heading is not None:
            print(f"[Pathfinder] Goal heading constraint: {goal_heading.name}")
        print(f"[Pathfinder] Target slope: {self.config.max_slope_percent}%, Hard limit: {self.config.hard_slope_limit_percent}%")
        print(f"[Pathfinder] Min radius: {self.config.min_curve_radius_m}m")
        print(f"[Pathfinder] Max heading change per step: {self.max_heading_change} (of 8)")
        
        # Check if start/goal are valid
        start_elev = self.get_elevation(start_row, start_col)
        goal_elev = self.get_elevation(goal_row, goal_col)
        print(f"[Pathfinder] Start elevation: {start_elev:.1f}m, Goal elevation: {goal_elev:.1f}m")
        
        if start_elev == float('inf') or goal_elev == float('inf'):
            return None, {"error": "Start or goal position is out of bounds"}
        
        # Check if start or goal is in water
        if self.is_water(start_row, start_col):
            print(f"[Pathfinder] Warning: Start position appears to be in water!")
        if self.is_water(goal_row, goal_col):
            print(f"[Pathfinder] Warning: Goal position appears to be in water!")
        
        # Calculate minimum theoretical distance considering grade limits
        direct_dist = math.sqrt((goal_row - start_row)**2 + (goal_col - start_col)**2) * self.config.cell_size_m
        elev_diff = abs(goal_elev - start_elev)
        min_dist_for_grade = elev_diff / (self.config.max_slope_percent / 100) if self.config.max_slope_percent > 0 else 0
        
        if min_dist_for_grade > direct_dist * 3:
            print(f"[Pathfinder] Warning: Elevation difference ({elev_diff:.0f}m) requires ~{min_dist_for_grade/1000:.1f}km at {self.config.max_slope_percent}% grade")
            print(f"[Pathfinder] Direct distance is only {direct_dist/1000:.1f}km - route will need significant detours")
        
        # Try with soft constraints first (allow slight slope exceedance with penalty)
        path, stats = self._run_astar(
            start_row, start_col, goal_row, goal_col, goal_elev,
            use_soft_slope=True,
            start_heading=start_heading,
            goal_heading=goal_heading
        )
        
        if path is not None:
            # Check if the path exceeds the soft limit
            max_slope = stats.get("max_slope_encountered", 0)
            if max_slope > self.config.max_slope_percent:
                stats["warning"] = f"Path uses grades up to {max_slope:.1f}% (exceeds {self.config.max_slope_percent}% target)"
                print(f"[Pathfinder] Warning: Best path has {max_slope:.1f}% max grade")
            return path, stats
        
        return None, stats
    
    def _run_astar(
        self,
        start_row: int, start_col: int,
        goal_row: int, goal_col: int,
        goal_elev: float,
        use_soft_slope: bool = True,
        start_heading: Optional[Heading] = None,
        goal_heading: Optional[Heading] = None
    ) -> Tuple[Optional[List[Tuple[int, int]]], Dict[str, Any]]:
        """
        Internal A* implementation.
        
        Args:
            use_soft_slope: If True, use soft slope constraints with penalties
            start_heading: Optional constrained heading at start
            goal_heading: Optional required heading at goal
        """
        # Initialize search with starting headings
        # If start_heading is specified, only use that heading
        # Otherwise, try ALL possible starting headings
        counter = 0
        open_set = []
        came_from: Dict[State, State] = {}
        g_score: Dict[State, float] = {}
        in_open: Set[State] = set()  # Track what's in open set for faster lookup
        
        start_headings = [start_heading] if start_heading is not None else list(Heading)
        
        for heading in start_headings:
            start_state = State(start_row, start_col, heading, heading, 0)  # initial_heading=heading, steps=0
            g_score[start_state] = 0
            h = self.heuristic(start_state, goal_row, goal_col, goal_elev)
            heapq.heappush(open_set, (h, counter, start_state))
            in_open.add(start_state)
            counter += 1
        
        nodes_expanded = 0
        max_queue_size = len(open_set)
        closed_set: Set[State] = set()
        
        while open_set:
            max_queue_size = max(max_queue_size, len(open_set))
            _, _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            closed_set.add(current)
            nodes_expanded += 1
            
            # Check if we've reached the goal
            # If goal_heading is specified, we must match that heading
            at_goal = current.row == goal_row and current.col == goal_col
            heading_ok = (goal_heading is None) or (current.heading == goal_heading)
            
            if at_goal and heading_ok:
                # Reconstruct path
                path = self._reconstruct_path(came_from, current)
                water_crossings = self._count_water_crossings(path)
                
                stats = {
                    "nodes_expanded": nodes_expanded,
                    "max_queue_size": max_queue_size,
                    "path_length": len(path),
                    "total_distance_m": self._calculate_path_distance(path),
                    "max_slope_encountered": self._calculate_max_slope(path),
                    "elevation_gain_m": self._calculate_elevation_gain(path),
                    "water_crossings": water_crossings,
                }
                print(f"[Pathfinder] Path found! {nodes_expanded} nodes, {len(path)} waypoints, {water_crossings} water crossings")
                return path, stats
            
            # Expand neighbors
            for neighbor, cost in self.get_neighbors(current, use_soft_slope=use_soft_slope):
                if neighbor in closed_set:
                    continue
                    
                tentative_g = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal_row, goal_col, goal_elev)
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))
            
            # Safety limit - increase for difficult terrain
            if nodes_expanded > 5000000:
                print(f"[Pathfinder] Search limit reached at {nodes_expanded} nodes")
                break
            
            # Progress update for long searches
            if nodes_expanded % 200000 == 0:
                # Calculate progress toward goal
                dist_to_goal = math.sqrt((current.row - goal_row)**2 + (current.col - goal_col)**2)
                print(f"[Pathfinder] Progress: {nodes_expanded} nodes, ~{dist_to_goal:.0f} cells from goal, queue: {len(open_set)}")
        
        # No path found
        print(f"[Pathfinder] No path found after {nodes_expanded} nodes expanded")
        self._print_debug_info(start_row, start_col, use_soft_slope)
        
        stats = {
            "nodes_expanded": nodes_expanded,
            "max_queue_size": max_queue_size,
            "path_length": 0,
            "error": f"No valid path found. Terrain may be too steep even for {self.config.hard_slope_limit_percent}% hard limit."
        }
        return None, stats
    
    def _print_debug_info(self, start_row: int, start_col: int, use_soft_slope: bool):
        """Print debugging information when pathfinding fails"""
        valid_from_start = 0
        blocked_by_slope = 0
        blocked_by_bounds = 0
        blocked_by_water = 0
        
        for heading in Heading:
            test_state = State(start_row, start_col, heading, heading, 0)
            neighbors = self.get_neighbors(test_state, use_soft_slope=use_soft_slope)
            valid_from_start += len(neighbors)
            
            # Check why blocked
            for new_heading in Heading:
                d_row, d_col = DIRECTION_VECTORS[new_heading]
                new_row = start_row + d_row
                new_col = start_col + d_col
                
                if not (0 <= new_row < self.rows and 0 <= new_col < self.cols):
                    blocked_by_bounds += 1
                    continue
                
                slope = self.calculate_slope(start_row, start_col, new_row, new_col)
                if abs(slope) > self.config.hard_slope_limit_percent:
                    blocked_by_slope += 1
                
                if self.is_water(new_row, new_col):
                    blocked_by_water += 1
        
        print(f"[Pathfinder] From start: {valid_from_start} valid moves across all headings")
        print(f"[Pathfinder] Blocked by slope: {blocked_by_slope}, by bounds: {blocked_by_bounds}, by water: {blocked_by_water}")
        
        if valid_from_start == 0:
            print(f"[Pathfinder] All directions blocked!")
            print(f"[Pathfinder] Slopes from start position:")
            for heading in Heading:
                d_row, d_col = DIRECTION_VECTORS[heading]
                new_row = start_row + d_row
                new_col = start_col + d_col
                if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                    slope = self.calculate_slope(start_row, start_col, new_row, new_col)
                    water = " [WATER]" if self.is_water(new_row, new_col) else ""
                    print(f"  {heading.name}: {slope:.2f}%{water}")
    
    def _count_water_crossings(self, path: List[Tuple[int, int]]) -> int:
        """Count how many water cells the path crosses"""
        count = 0
        for row, col in path:
            if self.is_water(row, col):
                count += 1
        return count
    
    def _direction_to_heading(self, d_row: int, d_col: int) -> Heading:
        """Convert a direction vector to the nearest heading"""
        if d_row == 0 and d_col == 0:
            return Heading.N
        
        angle = math.atan2(d_col, -d_row)  # Note: -d_row because row increases downward
        angle_deg = math.degrees(angle)
        if angle_deg < 0:
            angle_deg += 360
        
        # Map to nearest 45° increment
        heading_index = round(angle_deg / 45) % 8
        return Heading(heading_index)
    
    def _reconstruct_path(self, came_from: Dict[State, State], current: State) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to current"""
        path = [(current.row, current.col)]
        while current in came_from:
            current = came_from[current]
            path.append((current.row, current.col))
        path.reverse()
        return path
    
    def _calculate_path_distance(self, path: List[Tuple[int, int]]) -> float:
        """Calculate total path distance in meters"""
        total = 0.0
        for i in range(1, len(path)):
            r1, c1 = path[i-1]
            r2, c2 = path[i]
            dist_cells = math.sqrt((r2-r1)**2 + (c2-c1)**2)
            total += dist_cells * self.config.cell_size_m
        return total
    
    def _calculate_max_slope(self, path: List[Tuple[int, int]]) -> float:
        """Find the maximum slope along the path"""
        max_slope = 0.0
        for i in range(1, len(path)):
            r1, c1 = path[i-1]
            r2, c2 = path[i]
            slope = abs(self.calculate_slope(r1, c1, r2, c2))
            max_slope = max(max_slope, slope)
        return max_slope
    
    def _calculate_elevation_gain(self, path: List[Tuple[int, int]]) -> float:
        """Calculate total elevation gain (only uphill)"""
        gain = 0.0
        for i in range(1, len(path)):
            r1, c1 = path[i-1]
            r2, c2 = path[i]
            elev_change = self.get_elevation(r2, c2) - self.get_elevation(r1, c1)
            if elev_change > 0:
                gain += elev_change
        return gain
    
    def path_to_geojson(
        self, 
        path: List[Tuple[int, int]], 
        smooth: bool = True,
        smoothed_path: Optional[List[Tuple[float, float]]] = None,
        original_nodes: Optional[List[Tuple[int, int]]] = None
    ) -> dict:
        """
        Convert a grid path to GeoJSON LineString with elevation data.
        
        Args:
            path: Original grid path (row, col tuples)
            smooth: Whether to apply Douglas-Peucker smoothing
            smoothed_path: Pre-smoothed path with float coordinates (from spline)
            original_nodes: Original grid nodes for shadow visualization
        """
        coordinates = []
        elevations = []
        structure_types = []  # Track if each segment is normal, tunnel, or bridge
        
        # Detect jumps in the path (non-adjacent moves indicate tunnels/bridges)
        auto_tunnels = []
        auto_bridges = []
        
        # If we have a pre-smoothed path, use it for coordinates
        # but still detect tunnels/bridges from original grid path
        use_smoothed = smoothed_path is not None and len(smoothed_path) > 0
        
        # First, detect tunnels/bridges from original path
        for i, (row, col) in enumerate(path):
            if i > 0:
                prev_row, prev_col = path[i-1]
                dist = math.sqrt((row - prev_row)**2 + (col - prev_col)**2)
                
                if dist > 1.5:  # More than adjacent cell = jump
                    prev_lat, prev_lng = self.grid_to_latlon(prev_row, prev_col)
                    prev_elev = self.get_elevation(prev_row, prev_col)
                    lat, lng = self.grid_to_latlon(row, col)
                    elev = self.get_elevation(row, col)
                    
                    is_water = False
                    max_elev_between = elev
                    min_elev_between = elev
                    
                    steps = int(dist)
                    for t in range(1, steps):
                        frac = t / steps
                        check_row = int(prev_row + frac * (row - prev_row))
                        check_col = int(prev_col + frac * (col - prev_col))
                        if 0 <= check_row < self.rows and 0 <= check_col < self.cols:
                            if self.is_water(check_row, check_col):
                                is_water = True
                            check_elev = self.get_elevation(check_row, check_col)
                            max_elev_between = max(max_elev_between, check_elev)
                            min_elev_between = min(min_elev_between, check_elev)
                    
                    segment = {"start": [prev_lng, prev_lat], "end": [lng, lat]}
                    
                    if is_water or min_elev_between < min(prev_elev, elev) - 5:
                        auto_bridges.append(segment)
                    else:
                        auto_tunnels.append(segment)
        
        # Now generate coordinates from smoothed path or original
        if use_smoothed:
            for row_f, col_f in smoothed_path:
                # Interpolate position and elevation for float coordinates
                row_int = int(row_f)
                col_int = int(col_f)
                row_frac = row_f - row_int
                col_frac = col_f - col_int
                
                # Bilinear interpolation for position
                lat, lng = self.grid_to_latlon(row_f, col_f)  # Will use float version
                
                # Bilinear interpolation for elevation
                elev = self._interpolate_elevation(row_f, col_f)
                
                coordinates.append([lng, lat, elev])
                elevations.append(elev)
                
                # Determine structure type (simplified for smoothed path)
                row_check = int(round(row_f))
                col_check = int(round(col_f))
                if 0 <= row_check < self.rows and 0 <= col_check < self.cols:
                    if self.is_manual_tunnel_zone(row_check, col_check):
                        structure_types.append("tunnel")
                    elif self.is_manual_bridge_zone(row_check, col_check):
                        structure_types.append("bridge")
                    else:
                        structure_types.append("normal")
                else:
                    structure_types.append("normal")
        else:
            # Use original grid path
            for i, (row, col) in enumerate(path):
                lat, lng = self.grid_to_latlon(row, col)
                elev = self.get_elevation(row, col)
                coordinates.append([lng, lat, elev])
                elevations.append(elev)
                
                if i > 0:
                    prev_row, prev_col = path[i-1]
                    dist = math.sqrt((row - prev_row)**2 + (col - prev_col)**2)
                    
                    if dist > 1.5:
                        structure_types.append("bridge" if any(
                            b["end"] == [lng, lat] for b in auto_bridges
                        ) else "tunnel")
                    elif self.is_manual_tunnel_zone(row, col):
                        structure_types.append("tunnel")
                    elif self.is_manual_bridge_zone(row, col):
                        structure_types.append("bridge")
                    else:
                        structure_types.append("normal")
                else:
                    structure_types.append("normal")
        
        # Generate original nodes info for shadow visualization
        grid_nodes = []
        if original_nodes:
            for row, col in original_nodes:
                lat, lng = self.grid_to_latlon(row, col)
                elev = self.get_elevation(row, col)
                grid_nodes.append({
                    "lat": lat,
                    "lng": lng,
                    "elevation": round(elev, 1),
                    "row": row,
                    "col": col
                })
        
        # Apply Douglas-Peucker smoothing if requested and not using pre-smoothed
        distances = []
        if smooth and len(coordinates) > 3 and not use_smoothed:
            coordinates, elevations, distances = self._smooth_path(coordinates, elevations)
        else:
            # Calculate distances for path
            distances = [0.0]
            for i in range(1, len(coordinates)):
                d = self._haversine_distance(
                    coordinates[i-1][1], coordinates[i-1][0],
                    coordinates[i][1], coordinates[i][0]
                )
                distances.append(distances[-1] + d)
        
        # Create coordinate list for elevation chart with lat/lng for hover
        elevation_profile = []
        for i in range(len(coordinates)):
            elevation_profile.append({
                "distance": round(distances[i], 1),
                "elevation": round(elevations[i], 1),
                "lat": coordinates[i][1],
                "lng": coordinates[i][0]
            })
        
        return {
            "type": "Feature",
            "properties": {
                "elevations": elevations,
                "elevation_profile": elevation_profile,
                "distances": distances,
                "total_distance_m": distances[-1] if distances else 0,
                "waypoint_count": len(coordinates),
                "original_waypoints": len(path),
                "smoothed": smooth or use_smoothed,
                "min_curve_radius_m": self.config.min_curve_radius_m,
                "structure_types": structure_types,
                "auto_tunnels": auto_tunnels,
                "auto_bridges": auto_bridges,
                "grid_nodes": grid_nodes  # New: for shadow visualization
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            }
        }
    
    def _smooth_path(
        self, 
        coordinates: List[List[float]], 
        elevations: List[float]
    ) -> Tuple[List[List[float]], List[float], List[float]]:
        """
        Smooth a zig-zag grid path into a railway alignment with minimum curve radius.
        
        Uses circular arc filleting at corners to ensure minimum radius constraint.
        Returns: (smoothed_coordinates, smoothed_elevations, cumulative_distances)
        """
        import numpy as np
        
        if len(coordinates) < 3:
            # Not enough points to smooth, return with distances
            distances = [0.0]
            for i in range(1, len(coordinates)):
                d = self._haversine_distance(
                    coordinates[i-1][1], coordinates[i-1][0],
                    coordinates[i][1], coordinates[i][0]
                )
                distances.append(distances[-1] + d)
            return coordinates, elevations, distances
        
        # Convert to meters for radius calculations
        # Use local coordinate system centered on first point
        ref_lat = coordinates[0][1]
        ref_lng = coordinates[0][0]
        
        # Convert to local XY coordinates (meters)
        local_points = []
        for coord in coordinates:
            x = (coord[0] - ref_lng) * 111320 * math.cos(math.radians(ref_lat))
            y = (coord[1] - ref_lat) * 110540
            local_points.append((x, y, coord[2] if len(coord) > 2 else 0))
        
        # First simplify using Douglas-Peucker to reduce points
        simplified_indices = self._douglas_peucker_indices(local_points, epsilon=5.0)  # 5m tolerance
        simplified_points = [local_points[i] for i in simplified_indices]
        simplified_elevations = [elevations[i] for i in simplified_indices]
        
        if len(simplified_points) < 3:
            # Not enough points after simplification
            distances = [0.0]
            for i in range(1, len(coordinates)):
                d = self._haversine_distance(
                    coordinates[i-1][1], coordinates[i-1][0],
                    coordinates[i][1], coordinates[i][0]
                )
                distances.append(distances[-1] + d)
            return coordinates, elevations, distances
        
        # Apply minimum radius filleting at each corner
        min_radius = self.config.min_curve_radius_m
        smoothed_local = self._apply_corner_filleting(simplified_points, min_radius)
        
        # Interpolate elevations along the smoothed path
        # First calculate cumulative distances along simplified path
        simplified_distances = [0.0]
        for i in range(1, len(simplified_points)):
            dx = simplified_points[i][0] - simplified_points[i-1][0]
            dy = simplified_points[i][1] - simplified_points[i-1][1]
            simplified_distances.append(simplified_distances[-1] + math.sqrt(dx*dx + dy*dy))
        
        # Calculate distances along smoothed path
        smoothed_distances = [0.0]
        for i in range(1, len(smoothed_local)):
            dx = smoothed_local[i][0] - smoothed_local[i-1][0]
            dy = smoothed_local[i][1] - smoothed_local[i-1][1]
            smoothed_distances.append(smoothed_distances[-1] + math.sqrt(dx*dx + dy*dy))
        
        # Interpolate elevations to smoothed path
        total_simplified = simplified_distances[-1]
        total_smoothed = smoothed_distances[-1]
        
        smoothed_elevations = []
        for i, dist in enumerate(smoothed_distances):
            # Map smoothed distance to simplified distance
            if total_smoothed > 0:
                mapped_dist = (dist / total_smoothed) * total_simplified
            else:
                mapped_dist = 0
            
            # Find elevation at this distance
            elev = self._interpolate_elevation_at_distance(
                simplified_distances, simplified_elevations, mapped_dist
            )
            smoothed_elevations.append(elev)
        
        # Convert back to lat/lng
        result_coords = []
        for x, y in smoothed_local:
            lng = ref_lng + x / (111320 * math.cos(math.radians(ref_lat)))
            lat = ref_lat + y / 110540
            result_coords.append([lng, lat])
        
        # Add elevations to coordinates
        for i, coord in enumerate(result_coords):
            coord.append(smoothed_elevations[i])
        
        return result_coords, smoothed_elevations, smoothed_distances
    
    def _apply_corner_filleting(
        self, 
        points: List[Tuple[float, float, float]], 
        min_radius: float
    ) -> List[Tuple[float, float]]:
        """
        Apply circular arc filleting at each corner to ensure minimum radius.
        
        At each vertex where direction changes, replace the sharp corner with
        a circular arc of at least min_radius.
        """
        if len(points) < 3 or min_radius <= 0:
            return [(p[0], p[1]) for p in points]
        
        result = [(points[0][0], points[0][1])]  # Start point
        
        for i in range(1, len(points) - 1):
            p0 = (points[i-1][0], points[i-1][1])
            p1 = (points[i][0], points[i][1])
            p2 = (points[i+1][0], points[i+1][1])
            
            # Calculate vectors
            v1 = (p1[0] - p0[0], p1[1] - p0[1])
            v2 = (p2[0] - p1[0], p2[1] - p1[1])
            
            len1 = math.sqrt(v1[0]**2 + v1[1]**2)
            len2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if len1 < 0.01 or len2 < 0.01:
                result.append(p1)
                continue
            
            # Normalize vectors
            u1 = (v1[0] / len1, v1[1] / len1)
            u2 = (v2[0] / len2, v2[1] / len2)
            
            # Calculate turn angle
            dot = u1[0] * u2[0] + u1[1] * u2[1]
            dot = max(-1, min(1, dot))  # Clamp for numerical stability
            turn_angle = math.acos(dot)
            
            if turn_angle < 0.01:  # Nearly straight
                result.append(p1)
                continue
            
            # For a circular arc fillet:
            # tangent_distance = radius * tan(turn_angle / 2)
            half_angle = turn_angle / 2
            if half_angle > 0:
                tangent_distance = min_radius * math.tan(half_angle)
            else:
                result.append(p1)
                continue
            
            # Check if we have enough room on both segments
            max_tangent = min(len1 * 0.4, len2 * 0.4)  # Don't use more than 40% of segment
            if tangent_distance > max_tangent:
                # Scale down the radius to fit
                tangent_distance = max_tangent
                actual_radius = tangent_distance / math.tan(half_angle) if half_angle > 0.01 else min_radius
            else:
                actual_radius = min_radius
            
            # Calculate tangent points
            t1 = (p1[0] - u1[0] * tangent_distance, p1[1] - u1[1] * tangent_distance)
            t2 = (p1[0] + u2[0] * tangent_distance, p1[1] + u2[1] * tangent_distance)
            
            # Add tangent point 1
            result.append(t1)
            
            # Generate arc points between t1 and t2
            arc_points = self._generate_arc_points(t1, p1, t2, actual_radius, 8)
            result.extend(arc_points)
            
            # Add tangent point 2
            result.append(t2)
        
        # Add end point
        result.append((points[-1][0], points[-1][1]))
        
        return result
    
    def _generate_arc_points(
        self,
        start: Tuple[float, float],
        corner: Tuple[float, float],
        end: Tuple[float, float],
        radius: float,
        num_points: int
    ) -> List[Tuple[float, float]]:
        """
        Generate points along a circular arc from start to end around corner.
        """
        if num_points < 1:
            return []
        
        # Calculate arc center
        # The center is equidistant from start and end, at distance radius
        mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        
        # Direction from corner to midpoint
        to_mid = (mid[0] - corner[0], mid[1] - corner[1])
        to_mid_len = math.sqrt(to_mid[0]**2 + to_mid[1]**2)
        
        if to_mid_len < 0.01:
            return []
        
        # The center is along this direction, at distance such that
        # the center is at 'radius' from both start and end
        # Using geometry: center is at distance d from corner where
        # d = sqrt(radius^2 - (chord/2)^2) + distance_corner_to_mid
        chord = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        half_chord = chord / 2
        
        if half_chord > radius:
            # Chord is too long for this radius, just return midpoint
            return [mid]
        
        # Distance from midpoint to center (perpendicular to chord)
        sagitta = radius - math.sqrt(radius**2 - half_chord**2)
        center_dist = to_mid_len + sagitta
        
        center = (
            corner[0] + (to_mid[0] / to_mid_len) * center_dist,
            corner[1] + (to_mid[1] / to_mid_len) * center_dist
        )
        
        # Generate arc points
        start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
        end_angle = math.atan2(end[1] - center[1], end[0] - center[0])
        
        # Ensure we go the short way around
        angle_diff = end_angle - start_angle
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        elif angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Actual radius (distance from center to start/end points)
        actual_radius = math.sqrt((start[0] - center[0])**2 + (start[1] - center[1])**2)
        
        arc_points = []
        for j in range(1, num_points + 1):
            t = j / (num_points + 1)
            angle = start_angle + t * angle_diff
            x = center[0] + actual_radius * math.cos(angle)
            y = center[1] + actual_radius * math.sin(angle)
            arc_points.append((x, y))
        
        return arc_points
    
    def _interpolate_elevation_at_distance(
        self,
        distances: List[float],
        elevations: List[float],
        target_distance: float
    ) -> float:
        """Interpolate elevation at a given distance along the path."""
        if target_distance <= 0:
            return elevations[0]
        if target_distance >= distances[-1]:
            return elevations[-1]
        
        # Find the segment containing target_distance
        for i in range(1, len(distances)):
            if distances[i] >= target_distance:
                # Interpolate between i-1 and i
                t = (target_distance - distances[i-1]) / (distances[i] - distances[i-1])
                return elevations[i-1] + t * (elevations[i] - elevations[i-1])
        
        return elevations[-1]
    
    def _douglas_peucker_indices(
        self,
        points: List[Tuple[float, float, float]],
        epsilon: float
    ) -> List[int]:
        """
        Douglas-Peucker simplification, returns indices of points to keep.
        """
        if len(points) <= 2:
            return list(range(len(points)))
        
        # Find the point with maximum distance from the line between first and last
        max_dist = 0
        max_idx = 0
        
        for i in range(1, len(points) - 1):
            dist = self._point_line_distance_2d(
                (points[i][0], points[i][1]),
                (points[0][0], points[0][1]),
                (points[-1][0], points[-1][1])
            )
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        if max_dist > epsilon:
            left = self._douglas_peucker_indices(points[:max_idx + 1], epsilon)
            right = self._douglas_peucker_indices(points[max_idx:], epsilon)
            # Combine, adjusting right indices and avoiding duplicate
            return left[:-1] + [idx + max_idx for idx in right]
        else:
            return [0, len(points) - 1]
    
    def _point_line_distance_2d(
        self,
        point: Tuple[float, float],
        line_start: Tuple[float, float],
        line_end: Tuple[float, float]
    ) -> float:
        """Calculate perpendicular distance from point to line."""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        line_len_sq = (x2 - x1)**2 + (y2 - y1)**2
        if line_len_sq == 0:
            return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        return numerator / math.sqrt(line_len_sq)
    
    def _haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two lat/lng points in meters."""
        R = 6371000  # Earth radius in meters
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _douglas_peucker(
        self,
        coordinates: List[List[float]],
        elevations: List[float],
        epsilon: float
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Douglas-Peucker line simplification algorithm.
        Removes points that don't contribute significantly to the shape.
        """
        if len(coordinates) <= 2:
            return coordinates, elevations
        
        # Find the point with maximum distance from the line between first and last
        start = coordinates[0]
        end = coordinates[-1]
        
        max_dist = 0
        max_idx = 0
        
        for i in range(1, len(coordinates) - 1):
            dist = self._point_line_distance(coordinates[i], start, end)
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        # If max distance > epsilon, recursively simplify
        if max_dist > epsilon:
            # Recursive call on two halves
            left_coords, left_elevs = self._douglas_peucker(
                coordinates[:max_idx + 1], elevations[:max_idx + 1], epsilon
            )
            right_coords, right_elevs = self._douglas_peucker(
                coordinates[max_idx:], elevations[max_idx:], epsilon
            )
            
            # Combine (avoid duplicating the split point)
            return left_coords[:-1] + right_coords, left_elevs[:-1] + right_elevs
        else:
            # Just keep endpoints
            return [start, end], [elevations[0], elevations[-1]]
    
    def _point_line_distance(
        self,
        point: List[float],
        line_start: List[float],
        line_end: List[float]
    ) -> float:
        """Calculate perpendicular distance from point to line segment."""
        x0, y0 = point[0], point[1]
        x1, y1 = line_start[0], line_start[1]
        x2, y2 = line_end[0], line_end[1]
        
        # Line length squared
        line_len_sq = (x2 - x1)**2 + (y2 - y1)**2
        
        if line_len_sq == 0:
            # Line is a point
            return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
        # Perpendicular distance using cross product
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = math.sqrt(line_len_sq)
        
        return numerator / denominator


def heading_from_delta(d_row: int, d_col: int) -> Optional[Heading]:
    """Calculate heading from row/column delta."""
    for heading, (dr, dc) in DIRECTION_VECTORS.items():
        if dr == d_row and dc == d_col:
            return heading
    return None


def heading_from_points(from_point: Tuple[float, float], to_point: Tuple[float, float]) -> Heading:
    """
    Calculate heading from one lat/lng point to another.
    Returns the closest discrete heading.
    """
    lat1, lng1 = from_point
    lat2, lng2 = to_point
    
    # Calculate bearing (angle from North, clockwise)
    d_lat = lat2 - lat1
    d_lng = lng2 - lng1
    
    # Bearing in radians
    angle = math.atan2(d_lng, d_lat)  # Note: (d_lng, d_lat) for standard xy convention
    angle_deg = math.degrees(angle)
    
    # Normalize to 0-360
    if angle_deg < 0:
        angle_deg += 360
    
    # Map to discrete heading (each is 45 degrees)
    # N=0°, NE=45°, E=90°, SE=135°, S=180°, SW=225°, W=270°, NW=315°
    heading_index = round(angle_deg / 45) % 8
    return Heading(heading_index)


def smooth_curves_post_process(
    path: List[Tuple[int, int]],
    elevation_grid: np.ndarray,
    config: PathfindingConfig,
    min_curve_radius_m: float
) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
    """
    Post-process path to smooth curves using Catmull-Rom spline interpolation.
    This creates smooth curves while staying close to the original path.
    
    Args:
        path: Original path as list of (row, col) tuples (grid coordinates)
        elevation_grid: The elevation grid for terrain checking
        config: Pathfinding configuration
        min_curve_radius_m: Minimum curve radius to enforce
    
    Returns:
        Tuple of:
        - Smoothed path with float coordinates (many interpolated points for smooth curves)
        - Original grid nodes for shadow visualization
    """
    if len(path) < 3:
        return [(float(p[0]), float(p[1])) for p in path], list(path)
    
    # Store original nodes for shadow visualization
    original_nodes = list(path)
    
    # Convert all path points to float (keep ALL points as control points)
    # This ensures we stay close to the original path
    control_points = [(float(p[0]), float(p[1])) for p in path]
    
    # Apply Catmull-Rom spline interpolation with many intermediate points
    # This creates smooth curves between ALL original points
    smoothed = _catmull_rom_spline(control_points, segments_per_curve=8)
    
    return smoothed, original_nodes


def _identify_control_points(path: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
    """
    Identify key control points from path by detecting significant direction changes.
    Returns float coordinates for smooth interpolation.
    NOTE: This function is no longer used - we keep all points for better path fidelity.
    """
    if len(path) < 3:
        return [(float(p[0]), float(p[1])) for p in path]
    
    control_points = [(float(path[0][0]), float(path[0][1]))]  # Always include start
    
    # Calculate cumulative direction changes
    i = 1
    last_added_idx = 0
    
    while i < len(path) - 1:
        # Look at local curvature over a window
        window_size = min(5, len(path) - i - 1, i)
        if window_size < 1:
            i += 1
            continue
        
        # Vector from previous control point to current
        prev_cp = control_points[-1]
        curr = path[i]
        
        # Calculate direction change
        if i >= 1 and i < len(path) - 1:
            v1 = (curr[0] - path[i-1][0], curr[1] - path[i-1][1])
            v2 = (path[i+1][0] - curr[0], path[i+1][1] - curr[1])
            
            # Cross product to detect turns
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
                angle = math.degrees(math.acos(cos_angle))
                
                # Keep point if direction change > 15° or enough distance
                distance_from_last = math.sqrt(
                    (curr[0] - prev_cp[0])**2 + (curr[1] - prev_cp[1])**2
                )
                
                if angle > 15 or distance_from_last > 5:
                    # Use EXACT position, no averaging (to stay on valid terrain)
                    control_points.append((float(curr[0]), float(curr[1])))
                    last_added_idx = i
        
        i += 1
    
    # Always include end point
    control_points.append((float(path[-1][0]), float(path[-1][1])))
    
    return control_points


def _catmull_rom_spline(
    control_points: List[Tuple[float, float]],
    segments_per_curve: int = 5,
    alpha: float = 0.5  # Centripetal Catmull-Rom (0.5 avoids cusps)
) -> List[Tuple[float, float]]:
    """
    Generate smooth curve using Catmull-Rom spline interpolation.
    
    Args:
        control_points: List of control points
        segments_per_curve: Number of interpolated points between each control point pair
        alpha: Parameterization (0=uniform, 0.5=centripetal, 1=chordal)
    
    Returns:
        Smoothed path with interpolated points
    """
    if len(control_points) < 4:
        return control_points
    
    result = []
    
    # Add phantom points at start and end for full curve coverage
    p0 = (2 * control_points[0][0] - control_points[1][0],
          2 * control_points[0][1] - control_points[1][1])
    pn = (2 * control_points[-1][0] - control_points[-2][0],
          2 * control_points[-1][1] - control_points[-2][1])
    
    extended_points = [p0] + control_points + [pn]
    
    for i in range(len(extended_points) - 3):
        p0, p1, p2, p3 = extended_points[i:i+4]
        
        # Generate points along this segment
        for j in range(segments_per_curve):
            t = j / segments_per_curve
            
            # Catmull-Rom basis functions
            t2 = t * t
            t3 = t2 * t
            
            # Using standard Catmull-Rom matrix
            b0 = -0.5*t3 + t2 - 0.5*t
            b1 = 1.5*t3 - 2.5*t2 + 1.0
            b2 = -1.5*t3 + 2.0*t2 + 0.5*t
            b3 = 0.5*t3 - 0.5*t2
            
            x = b0*p0[0] + b1*p1[0] + b2*p2[0] + b3*p3[0]
            y = b0*p0[1] + b1*p1[1] + b2*p2[1] + b3*p3[1]
            
            result.append((x, y))
    
    # Add the last control point
    result.append(control_points[-1])
    
    return result


def _remove_float_zigzags(path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Remove zig-zag patterns from smoothed path.
    """
    if len(path) < 4:
        return path
    
    result = [path[0]]
    i = 1
    
    while i < len(path) - 1:
        prev = result[-1]
        curr = path[i]
        next_pt = path[i + 1]
        
        # Calculate vectors
        v1 = (curr[0] - prev[0], curr[1] - prev[1])
        v2 = (next_pt[0] - curr[0], next_pt[1] - curr[1])
        
        # Check for sharp reversal (zig-zag)
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 > 0.001 and mag2 > 0.001:
            cos_angle = dot / (mag1 * mag2)
            cos_angle = max(-1, min(1, cos_angle))
            angle = math.degrees(math.acos(cos_angle))
            
            # Skip points that create very sharp angles (< 60 degrees = backtracking)
            if angle < 60:
                i += 1
                continue
        
        result.append(curr)
        i += 1
    
    result.append(path[-1])
    return result
