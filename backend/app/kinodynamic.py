"""
Kinodynamic A* Pathfinding for Railway Routes

Continuous state-space planning with:
- Continuous position (x, y) in meters (local coordinate system)
- Continuous heading (radians)
- Direction gear (Forward/Reverse) for switchback support
- Motion primitives with configurable arc length
- Minimum curve radius constraint built into primitives
- Slope constraint sampling along arcs
"""

import math
import time
import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Set
from enum import IntEnum, auto
from functools import cached_property


# =============================================================================
# ENUMS
# =============================================================================

class DirectionGear(IntEnum):
    """Direction of travel along the track"""
    FORWARD = 1
    REVERSE = -1


class StructureType(IntEnum):
    """Type of railway structure at a point"""
    NORMAL = 0
    TUNNEL = 1
    BRIDGE = 2


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class KinodynamicConfig:
    """
    Configuration for the Kinodynamic A* pathfinder.
    
    Coordinate System:
    - All positions are in meters, using a local coordinate system
    - Origin is at the southwest corner of the bounding box
    - X increases eastward, Y increases northward
    - Heading 0 = North (positive Y), increases clockwise
    """
    
    # === Grid/Resolution Settings ===
    # step_distance: Length of each motion primitive arc (meters)
    # This replaces the old cell_size_m concept
    # Larger values = faster search, coarser path
    # Smaller values = slower search, smoother path
    step_distance_m: float = 30.0
    
    # === Bounding Box ===
    # These are set dynamically based on start/end points + padding
    bounds_min_x: float = 0.0
    bounds_min_y: float = 0.0
    bounds_max_x: float = 1000.0
    bounds_max_y: float = 1000.0
    padding_factor: float = 0.3  # Extra padding around search area
    
    # === Slope Constraints ===
    max_slope_percent: float = 3.0  # Target maximum grade (soft limit)
    hard_slope_limit_percent: float = 8.0  # Absolute maximum (hard limit)
    slope_penalty_multiplier: float = 20.0  # Penalty per % over soft limit
    
    # === Curvature Constraints ===
    min_curve_radius_m: float = 500.0  # Minimum curve radius
    
    # Number of discrete curvature options for motion primitives
    # More options = smoother curves but slower search
    # Curvatures will be: [-1/R, ..., 0, ..., +1/R]
    num_curvature_samples: int = 7  # Odd number for including straight
    
    # === Cost Weights ===
    distance_weight: float = 1.0
    elevation_gain_weight: float = 1.5  # Penalty for climbing
    curvature_weight: float = 0.3  # Base penalty for curved segments
    curvature_change_weight: float = 0.5  # Penalty for changing curvature (jerk)
    
    # === Switchback Settings ===
    allow_switchbacks: bool = False
    switchback_penalty: float = 5000.0  # Cost to perform a switchback
    min_switchback_distance_m: float = 500.0  # Minimum distance between switchbacks
    
    # === Water Avoidance ===
    water_penalty: float = 10000.0  # Cost to cross water (if not bridged)
    water_elevation_threshold: float = 1.0  # Elevation threshold for water detection
    allow_water_crossing: bool = False  # If True, allow crossing water with penalty instead of blocking
    max_water_crossing_m: float = 100.0  # Maximum width of water that can be crossed (for small creeks)
    
    # === Structure Costs ===
    tunnel_cost_per_m: float = 50.0  # Extra cost per meter in tunnel
    bridge_cost_per_m: float = 100.0  # Extra cost per meter on bridge
    
    # === Auto Tunnel/Bridge Detection ===
    auto_tunnel_bridge: bool = False
    max_jump_distance_m: float = 500.0
    elevation_tolerance_m: float = 10.0
    
    # === Road Parallelism ===
    road_parallel_enabled: bool = False
    road_parallel_threshold_deg: float = 30.0
    road_min_separation_m: float = 10.0
    road_max_separation_m: float = 50.0
    road_parallel_penalty: float = 500.0
    
    # === Hybrid A* Optimizations ===
    # Spatial hashing resolution for closed set (coarser = faster but less precise)
    position_bucket_size: float = 5.0  # meters per bucket
    heading_bucket_degrees: float = 10.0  # degrees per bucket
    
    # Heuristic weighting (epsilon-admissible search)
    # Higher values = faster but less optimal paths
    heuristic_weight: float = 1.5  # Multiplier for heuristic
    
    # Analytic expansion (shot-to-goal)
    analytic_expansion_distance: float = 5.0  # Try direct path when dist < step_size * this
    analytic_expansion_samples: int = 10  # Number of samples to check along direct path
    
    # Relaxed goal tolerance (relative to step size)
    goal_tolerance_multiplier: float = 1.5  # goal_tolerance = step_size * this
    
    # === Bidirectional Search ===
    bidirectional_search: bool = False  # Search from both start and goal
    
    # === Search Limits ===
    max_iterations: int = 5_000_000
    progress_interval: int = 100_000
    
    def get_curvatures(self) -> List[float]:
        """
        Get the list of curvature values for motion primitives.
        
        Curvature κ = 1/radius. Positive = turn right, Negative = turn left.
        Returns curvatures from max-left to max-right, including 0 (straight).
        """
        if self.min_curve_radius_m <= 0:
            # No curvature limit - just use straight
            return [0.0]
        
        max_curvature = 1.0 / self.min_curve_radius_m
        
        if self.num_curvature_samples <= 1:
            return [0.0]
        
        # Generate symmetric curvature samples
        # e.g., for 7 samples: [-max, -2/3*max, -1/3*max, 0, 1/3*max, 2/3*max, max]
        n = self.num_curvature_samples
        curvatures = []
        for i in range(n):
            # Map i from [0, n-1] to [-1, 1]
            t = (2 * i / (n - 1)) - 1 if n > 1 else 0
            curvatures.append(t * max_curvature)
        
        return curvatures
    
    def is_in_bounds(self, x: float, y: float, epsilon: float = 1e-6) -> bool:
        """Check if a position is within the search bounds (including padding).
        
        Uses a small epsilon to handle floating-point errors at boundaries.
        """
        return (self.bounds_min_x - epsilon <= x <= self.bounds_max_x + epsilon and
                self.bounds_min_y - epsilon <= y <= self.bounds_max_y + epsilon)


# =============================================================================
# STATE REPRESENTATION
# =============================================================================

@dataclass
class State:
    """
    Continuous state for Hybrid A*.
    
    Position is in local meters coordinate system:
    - x: Easting (meters from origin)
    - y: Northing (meters from origin)
    - heading: Direction of travel in radians (0 = North, increases clockwise)
    - direction_gear: FORWARD or REVERSE
    - last_switchback_pos: (x, y) of last switchback, or None
    
    The state is hashable for use in closed set, using discretized values
    (spatial hashing) for efficient lookup.
    """
    x: float
    y: float
    heading: float  # radians, 0 = North, clockwise positive
    direction_gear: DirectionGear = DirectionGear.FORWARD
    
    # Switchback tracking - not part of hash/equality (tracked separately)
    last_switchback_x: Optional[float] = field(default=None, compare=False, hash=False)
    last_switchback_y: Optional[float] = field(default=None, compare=False, hash=False)
    
    # === Spatial Hashing (Quantization) for Hybrid A* ===
    # Coarser buckets = faster search, less precise deduplication
    # position_resolution: meters per bucket (e.g., 5.0 = 5m buckets)
    # heading_resolution: radians per bucket (e.g., 0.175 = ~10 degrees)
    _position_resolution: float = field(default=5.0, repr=False, compare=False, hash=False)
    _heading_resolution: float = field(default=0.175, repr=False, compare=False, hash=False)  # ~10 degrees
    
    def __post_init__(self):
        # Normalize heading to [0, 2π)
        self.heading = self.heading % (2 * math.pi)
    
    @property
    def last_switchback_pos(self) -> Optional[Tuple[float, float]]:
        """Get last switchback position as tuple, or None."""
        if self.last_switchback_x is not None and self.last_switchback_y is not None:
            return (self.last_switchback_x, self.last_switchback_y)
        return None
    
    def distance_since_switchback(self) -> float:
        """Calculate distance from current position to last switchback."""
        if self.last_switchback_pos is None:
            return float('inf')
        dx = self.x - self.last_switchback_x
        dy = self.y - self.last_switchback_y
        return math.sqrt(dx * dx + dy * dy)
    
    def with_switchback_at_current(self) -> 'State':
        """Create a new state with switchback recorded at current position."""
        return State(
            x=self.x,
            y=self.y,
            heading=self.heading,
            direction_gear=self.direction_gear,
            last_switchback_x=self.x,
            last_switchback_y=self.y,
            _position_resolution=self._position_resolution,
            _heading_resolution=self._heading_resolution
        )
    
    def _discretize(self) -> Tuple[int, int, int, int]:
        """
        Discretize state for spatial hashing.
        
        Uses coarse buckets for efficient closed-set lookup:
        - Position: 5m buckets by default (configurable)
        - Heading: ~10 degree buckets by default (configurable)
        - Gear: Forward/Reverse
        """
        dx = int(self.x // self._position_resolution)
        dy = int(self.y // self._position_resolution)
        dh = int(self.heading // self._heading_resolution) % int(2 * math.pi / self._heading_resolution + 0.5)
        dg = int(self.direction_gear)
        return (dx, dy, dh, dg)
    
    def __hash__(self):
        return hash(self._discretize())
    
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return self._discretize() == other._discretize()
    
    def __lt__(self, other):
        """For heap ordering."""
        if not isinstance(other, State):
            return NotImplemented
        return self._discretize() < other._discretize()
    
    def to_latlon(self, origin_lat: float, origin_lng: float) -> Tuple[float, float]:
        """
        Convert local (x, y) back to (lat, lng).
        
        Args:
            origin_lat, origin_lng: The origin of the local coordinate system
        
        Returns:
            (lat, lng) tuple
        """
        # Approximate meters per degree at origin
        meters_per_deg_lat = 110540.0
        meters_per_deg_lng = 111320.0 * math.cos(math.radians(origin_lat))
        
        lat = origin_lat + (self.y / meters_per_deg_lat)
        lng = origin_lng + (self.x / meters_per_deg_lng)
        
        return (lat, lng)
    
    @classmethod
    def from_latlon(
        cls,
        lat: float,
        lng: float,
        origin_lat: float,
        origin_lng: float,
        heading: float = 0.0,
        direction_gear: DirectionGear = DirectionGear.FORWARD
    ) -> 'State':
        """
        Create a State from lat/lng coordinates.
        
        Args:
            lat, lng: Position in degrees
            origin_lat, origin_lng: Origin of local coordinate system
            heading: Heading in radians
            direction_gear: Direction of travel
        
        Returns:
            State object
        """
        meters_per_deg_lat = 110540.0
        meters_per_deg_lng = 111320.0 * math.cos(math.radians(origin_lat))
        
        x = (lng - origin_lng) * meters_per_deg_lng
        y = (lat - origin_lat) * meters_per_deg_lat
        
        return cls(
            x=x,
            y=y,
            heading=heading,
            direction_gear=direction_gear
        )


# =============================================================================
# MOTION PRIMITIVES
# =============================================================================

@dataclass
class MotionPrimitive:
    """
    A motion primitive representing an arc segment.
    
    The primitive describes movement from the current state:
    - curvature: 1/radius (positive = right turn, negative = left turn, 0 = straight)
    - arc_length: Distance traveled along the arc
    - is_switchback: Whether this primitive involves reversing direction
    
    The resulting position and heading can be computed from these parameters.
    """
    curvature: float  # 1/radius, radians per meter
    arc_length: float  # meters
    is_switchback: bool = False
    
    @cached_property
    def delta_heading(self) -> float:
        """Change in heading (radians) for this primitive."""
        return self.curvature * self.arc_length
    
    def apply(self, state: State, config: KinodynamicConfig) -> Optional[State]:
        """
        Apply this motion primitive to a state to get the resulting state.
        
        Uses exact arc geometry:
        - For straight (κ=0): simple linear motion
        - For curved (κ≠0): arc motion with radius R = 1/κ
        
        Returns None if the resulting state is invalid (out of bounds, etc.)
        """
        heading = state.heading
        gear = state.direction_gear
        
        # Handle switchback
        new_gear = gear
        new_last_sb_x = state.last_switchback_x
        new_last_sb_y = state.last_switchback_y
        
        if self.is_switchback:
            # Check if switchback is allowed
            if not config.allow_switchbacks:
                return None
            
            # Check minimum distance constraint
            if state.distance_since_switchback() < config.min_switchback_distance_m:
                return None
            
            # Reverse the gear
            new_gear = DirectionGear.REVERSE if gear == DirectionGear.FORWARD else DirectionGear.FORWARD
            new_last_sb_x = state.x
            new_last_sb_y = state.y
            
            # For switchback, we flip heading by π (reverse direction)
            heading = (heading + math.pi) % (2 * math.pi)
        
        # Apply the arc motion
        # Direction multiplier based on gear (REVERSE travels backward)
        direction = float(new_gear)
        
        if abs(self.curvature) < 1e-9:
            # Straight line motion
            dx = self.arc_length * math.sin(heading) * direction
            dy = self.arc_length * math.cos(heading) * direction
            new_x = state.x + dx
            new_y = state.y + dy
            new_heading = heading
        else:
            # Arc motion using a cleaner geometric approach
            # 
            # We have:
            # - Current position (x, y)
            # - Current heading θ (0 = North, clockwise positive)
            # - Curvature κ (positive = right turn, negative = left turn)
            # - Arc length s
            # - Direction d (1 = forward, -1 = reverse)
            #
            # The change in heading is: Δθ = s * κ * d
            # 
            # For the position change, we integrate along the arc:
            # The displacement is the chord of the arc, pointing from start to end
            
            delta_theta = self.arc_length * self.curvature * direction
            new_heading = (heading + delta_theta) % (2 * math.pi)
            
            # Calculate position change using the arc chord
            # For an arc with angle delta_theta and radius R = 1/|κ|:
            # - Chord length = 2 * R * sin(|delta_theta|/2)
            # - Chord direction = heading + delta_theta/2 (midpoint direction)
            
            abs_delta_theta = abs(delta_theta)
            abs_radius = abs(1.0 / self.curvature)
            
            if abs_delta_theta < 1e-9:
                # Nearly straight - use linear approximation
                dx = self.arc_length * math.sin(heading) * direction
                dy = self.arc_length * math.cos(heading) * direction
            else:
                # Chord length
                chord_length = 2.0 * abs_radius * math.sin(abs_delta_theta / 2.0)
                
                # Chord direction (average heading during the arc)
                chord_heading = heading + delta_theta / 2.0
                
                # Position change
                dx = chord_length * math.sin(chord_heading) * (1 if direction * self.curvature * direction >= 0 or True else -1)
                dy = chord_length * math.cos(chord_heading) * (1 if direction * self.curvature * direction >= 0 or True else -1)
                
                # Simpler: the chord always points in the direction of travel
                # Just use chord_length in the chord_heading direction
                dx = chord_length * math.sin(chord_heading)
                dy = chord_length * math.cos(chord_heading)
            
            new_x = state.x + dx
            new_y = state.y + dy
        
        # Check bounds
        if not config.is_in_bounds(new_x, new_y):
            return None
        
        return State(
            x=new_x,
            y=new_y,
            heading=new_heading,
            direction_gear=new_gear,
            last_switchback_x=new_last_sb_x,
            last_switchback_y=new_last_sb_y,
            _position_resolution=state._position_resolution,
            _heading_resolution=state._heading_resolution
        )
    
    def sample_positions(
        self,
        state: State,
        num_samples: int = 5
    ) -> List[Tuple[float, float]]:
        """
        Sample positions along this primitive for constraint checking.
        
        Returns list of (x, y) positions along the arc.
        """
        positions = []
        heading = state.heading
        direction = float(state.direction_gear)
        
        for i in range(num_samples + 1):
            t = i / num_samples
            arc_dist = t * self.arc_length
            
            if abs(self.curvature) < 1e-9:
                # Straight line
                x = state.x + arc_dist * math.sin(heading) * direction
                y = state.y + arc_dist * math.cos(heading) * direction
            else:
                # Arc - use chord-based geometry (same as apply)
                delta_theta = arc_dist * self.curvature * direction
                abs_delta_theta = abs(delta_theta)
                abs_radius = abs(1.0 / self.curvature)
                
                if abs_delta_theta < 1e-9:
                    x = state.x + arc_dist * math.sin(heading) * direction
                    y = state.y + arc_dist * math.cos(heading) * direction
                else:
                    chord_length = 2.0 * abs_radius * math.sin(abs_delta_theta / 2.0)
                    chord_heading = heading + delta_theta / 2.0
                    x = state.x + chord_length * math.sin(chord_heading)
                    y = state.y + chord_length * math.cos(chord_heading)
            
            positions.append((x, y))
        
        return positions


@dataclass
class MotionPrimitiveSet:
    """
    A precomputed set of motion primitives for the search.
    
    Primitives are generated based on the configuration:
    - Curvature options from -max_κ to +max_κ
    - Optional switchback primitive
    """
    primitives: List[MotionPrimitive]
    switchback_primitive: Optional[MotionPrimitive]
    
    @classmethod
    def from_config(cls, config: KinodynamicConfig) -> 'MotionPrimitiveSet':
        """Create a primitive set from configuration."""
        primitives = []
        
        # Get curvature samples
        curvatures = config.get_curvatures()
        
        # Create forward motion primitives
        for kappa in curvatures:
            primitives.append(MotionPrimitive(
                curvature=kappa,
                arc_length=config.step_distance_m,
                is_switchback=False
            ))
        
        # Create switchback primitive (if allowed)
        switchback = None
        if config.allow_switchbacks:
            # Switchback is modeled as a stop-and-reverse
            # Uses straight motion with switchback flag
            switchback = MotionPrimitive(
                curvature=0.0,
                arc_length=config.step_distance_m * 0.5,  # Shorter initial move after switch
                is_switchback=True
            )
        
        return cls(primitives=primitives, switchback_primitive=switchback)
    
    def get_applicable_primitives(
        self,
        state: State,
        config: KinodynamicConfig
    ) -> List[MotionPrimitive]:
        """
        Get primitives that can be applied from the given state.
        
        Filters out primitives that would violate constraints.
        """
        applicable = list(self.primitives)  # All forward primitives
        
        # Add switchback if conditions are met
        if self.switchback_primitive is not None:
            if state.distance_since_switchback() >= config.min_switchback_distance_m:
                applicable.append(self.switchback_primitive)
        
        return applicable


# =============================================================================
# COORDINATE TRANSFORMS
# =============================================================================

@dataclass
class CoordinateTransform:
    """
    Handles conversion between lat/lng and local meter coordinates.
    
    The local coordinate system:
    - Origin at (origin_lat, origin_lng)
    - X axis points East
    - Y axis points North
    - Units are meters
    """
    origin_lat: float
    origin_lng: float
    meters_per_deg_lat: float = field(init=False)
    meters_per_deg_lng: float = field(init=False)
    
    def __post_init__(self):
        self.meters_per_deg_lat = 110540.0
        self.meters_per_deg_lng = 111320.0 * math.cos(math.radians(self.origin_lat))
    
    def to_local(self, lat: float, lng: float) -> Tuple[float, float]:
        """Convert lat/lng to local (x, y) in meters."""
        x = (lng - self.origin_lng) * self.meters_per_deg_lng
        y = (lat - self.origin_lat) * self.meters_per_deg_lat
        return (x, y)
    
    def to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """Convert local (x, y) in meters to lat/lng."""
        lat = self.origin_lat + (y / self.meters_per_deg_lat)
        lng = self.origin_lng + (x / self.meters_per_deg_lng)
        return (lat, lng)
    
    def heading_from_points(
        self,
        from_lat: float, from_lng: float,
        to_lat: float, to_lng: float
    ) -> float:
        """
        Calculate heading (radians) from one point to another.
        
        Returns heading in radians, 0 = North, clockwise positive.
        """
        x1, y1 = self.to_local(from_lat, from_lng)
        x2, y2 = self.to_local(to_lat, to_lng)
        
        dx = x2 - x1
        dy = y2 - y1
        
        # atan2(dx, dy) gives angle from North (Y-axis), clockwise
        heading = math.atan2(dx, dy)
        
        # Normalize to [0, 2π)
        return heading % (2 * math.pi)
    
    @classmethod
    def from_bounds(
        cls,
        min_lat: float, min_lng: float,
        max_lat: float, max_lng: float
    ) -> 'CoordinateTransform':
        """Create a transform with origin at the southwest corner."""
        return cls(origin_lat=min_lat, origin_lng=min_lng)


# =============================================================================
# ELEVATION GRID INTERFACE
# =============================================================================

@dataclass
class ElevationGrid:
    """
    Interface for querying elevation data.
    
    Wraps a numpy array of elevation values with coordinate transforms
    and bilinear interpolation.
    """
    data: np.ndarray  # 2D array of elevations (row 0 = north)
    bounds: Tuple[float, float, float, float]  # (min_lat, min_lng, max_lat, max_lng)
    transform: CoordinateTransform = field(init=False)
    
    # Grid dimensions in local coordinates
    width_m: float = field(init=False)
    height_m: float = field(init=False)
    cell_size_x: float = field(init=False)
    cell_size_y: float = field(init=False)
    
    def __post_init__(self):
        min_lat, min_lng, max_lat, max_lng = self.bounds
        self.transform = CoordinateTransform.from_bounds(min_lat, min_lng, max_lat, max_lng)
        
        # Calculate dimensions
        _, _ = self.transform.to_local(min_lat, min_lng)  # (0, 0)
        self.width_m, self.height_m = self.transform.to_local(max_lat, max_lng)
        
        rows, cols = self.data.shape
        self.cell_size_x = self.width_m / (cols - 1) if cols > 1 else self.width_m
        self.cell_size_y = self.height_m / (rows - 1) if rows > 1 else self.height_m
    
    def get_elevation(self, x: float, y: float) -> float:
        """
        Get interpolated elevation at local coordinates (x, y).
        
        Uses bilinear interpolation.
        Returns inf if out of bounds.
        """
        if x < 0 or x > self.width_m or y < 0 or y > self.height_m:
            return float('inf')
        
        rows, cols = self.data.shape
        
        # Convert to grid coordinates
        # Note: row 0 is at max_lat (north), so we flip y
        col_f = x / self.cell_size_x if self.cell_size_x > 0 else 0
        row_f = (self.height_m - y) / self.cell_size_y if self.cell_size_y > 0 else 0
        
        # Clamp to valid range
        col_f = max(0, min(col_f, cols - 1))
        row_f = max(0, min(row_f, rows - 1))
        
        # Bilinear interpolation
        col0 = int(col_f)
        row0 = int(row_f)
        col1 = min(col0 + 1, cols - 1)
        row1 = min(row0 + 1, rows - 1)
        
        col_frac = col_f - col0
        row_frac = row_f - row0
        
        e00 = float(self.data[row0, col0])
        e01 = float(self.data[row0, col1])
        e10 = float(self.data[row1, col0])
        e11 = float(self.data[row1, col1])
        
        e0 = e00 * (1 - col_frac) + e01 * col_frac
        e1 = e10 * (1 - col_frac) + e11 * col_frac
        
        return e0 * (1 - row_frac) + e1 * row_frac
    
    def get_slope_along_segment(
        self,
        x1: float, y1: float,
        x2: float, y2: float
    ) -> float:
        """
        Calculate slope percentage along a segment.
        
        Returns: slope as percentage (rise/run * 100)
        """
        elev1 = self.get_elevation(x1, y1)
        elev2 = self.get_elevation(x2, y2)
        
        if elev1 == float('inf') or elev2 == float('inf'):
            return float('inf')
        
        dx = x2 - x1
        dy = y2 - y1
        horizontal_dist = math.sqrt(dx * dx + dy * dy)
        
        if horizontal_dist < 0.01:  # Avoid division by zero
            return 0.0
        
        elevation_change = elev2 - elev1
        return (elevation_change / horizontal_dist) * 100
    
    def get_max_slope_along_primitive(
        self,
        positions: List[Tuple[float, float]]
    ) -> float:
        """
        Get the maximum absolute slope along a series of positions.
        """
        max_slope = 0.0
        for i in range(1, len(positions)):
            x1, y1 = positions[i - 1]
            x2, y2 = positions[i]
            slope = abs(self.get_slope_along_segment(x1, y1, x2, y2))
            max_slope = max(max_slope, slope)
        return max_slope


# =============================================================================
# CONSTRAINT MASKS
# =============================================================================

@dataclass
class ConstraintMasks:
    """
    Binary masks for various constraints (water, tunnels, bridges, roads).
    
    All masks use the same grid as the elevation data.
    """
    water_mask: np.ndarray  # True = water (avoid)
    tunnel_mask: np.ndarray  # True = tunnel zone (slope relaxed)
    bridge_mask: np.ndarray  # True = bridge zone (water crossing allowed)
    
    # Road data (optional)
    road_mask: Optional[np.ndarray] = None  # True = road present
    road_direction: Optional[np.ndarray] = None  # Direction in radians
    road_distance: Optional[np.ndarray] = None  # Distance to nearest road (meters)
    
    # Reference to elevation grid for coordinate conversion
    elevation_grid: Optional[ElevationGrid] = None
    
    def is_water(self, x: float, y: float) -> bool:
        """Check if position is in water."""
        if self.elevation_grid is None:
            return False
        row, col = self._xy_to_grid(x, y)
        if row is None:
            return False
        return bool(self.water_mask[row, col])
    
    def is_tunnel_zone(self, x: float, y: float) -> bool:
        """Check if position is in a tunnel zone."""
        if self.elevation_grid is None:
            return False
        row, col = self._xy_to_grid(x, y)
        if row is None:
            return False
        return bool(self.tunnel_mask[row, col])
    
    def is_bridge_zone(self, x: float, y: float) -> bool:
        """Check if position is in a bridge zone."""
        if self.elevation_grid is None:
            return False
        row, col = self._xy_to_grid(x, y)
        if row is None:
            return False
        return bool(self.bridge_mask[row, col])
    
    def get_road_info(self, x: float, y: float) -> Tuple[float, float]:
        """
        Get road information at position.
        
        Returns: (distance_to_road_m, road_direction_rad)
        """
        if self.road_distance is None or self.elevation_grid is None:
            return (float('inf'), float('nan'))
        
        row, col = self._xy_to_grid(x, y)
        if row is None:
            return (float('inf'), float('nan'))
        
        distance = float(self.road_distance[row, col])
        direction = float(self.road_direction[row, col]) if self.road_direction is not None else float('nan')
        
        return (distance, direction)
    
    def _xy_to_grid(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Convert local (x, y) to grid (row, col)."""
        if self.elevation_grid is None:
            return None
        
        rows, cols = self.water_mask.shape
        
        col = int(x / self.elevation_grid.cell_size_x) if self.elevation_grid.cell_size_x > 0 else 0
        row = int((self.elevation_grid.height_m - y) / self.elevation_grid.cell_size_y) if self.elevation_grid.cell_size_y > 0 else 0
        
        if 0 <= row < rows and 0 <= col < cols:
            return (row, col)
        return None


# =============================================================================
# PATH RESULT
# =============================================================================

@dataclass
class PathResult:
    """
    Result of a pathfinding operation.
    """
    success: bool
    states: List[State] = field(default_factory=list)
    primitives: List[MotionPrimitive] = field(default_factory=list)
    
    # Statistics
    total_distance_m: float = 0.0
    elevation_gain_m: float = 0.0
    max_slope_percent: float = 0.0
    num_switchbacks: int = 0
    nodes_expanded: int = 0
    max_queue_size: int = 0
    
    # Error/warning messages
    message: str = ""
    warning: Optional[str] = None
    
    def to_coordinates(
        self,
        transform: CoordinateTransform,
        elevation_grid: ElevationGrid
    ) -> List[Tuple[float, float, float]]:
        """
        Convert path to list of (lng, lat, elevation) coordinates.
        """
        coords = []
        for state in self.states:
            lat, lng = transform.to_latlon(state.x, state.y)
            elev = elevation_grid.get_elevation(state.x, state.y)
            coords.append((lng, lat, elev))
        return coords


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_heading(heading: float) -> float:
    """Normalize heading to [0, 2π)."""
    return heading % (2 * math.pi)


def heading_difference(h1: float, h2: float) -> float:
    """
    Calculate the minimum angular difference between two headings.
    
    Returns value in [0, π].
    """
    diff = abs(normalize_heading(h1) - normalize_heading(h2))
    if diff > math.pi:
        diff = 2 * math.pi - diff
    return diff


def is_heading_nearly_opposite(h1: float, h2: float, tolerance: float = 0.3) -> bool:
    """Check if two headings are nearly opposite (for switchback detection)."""
    diff = heading_difference(h1, h2)
    return abs(diff - math.pi) < tolerance


# =============================================================================
# PORTAL (TUNNEL/BRIDGE) DEFINITIONS
# =============================================================================

@dataclass
class Portal:
    """
    A tunnel or bridge portal connecting two points.
    
    When the pathfinder reaches the entry point, it can "teleport" to the exit,
    bypassing terrain constraints between them.
    """
    entry_x: float
    entry_y: float
    exit_x: float
    exit_y: float
    structure_type: StructureType  # TUNNEL or BRIDGE
    
    # How close must we be to "enter" the portal (meters)
    entry_tolerance: float = 50.0
    
    def distance_to_entry(self, x: float, y: float) -> float:
        """Calculate distance from a point to the portal entry."""
        dx = x - self.entry_x
        dy = y - self.entry_y
        return math.sqrt(dx * dx + dy * dy)
    
    def is_at_entry(self, x: float, y: float) -> bool:
        """Check if position is close enough to enter the portal."""
        return self.distance_to_entry(x, y) <= self.entry_tolerance
    
    def get_exit_heading(self) -> float:
        """Calculate heading from entry to exit."""
        dx = self.exit_x - self.entry_x
        dy = self.exit_y - self.entry_y
        return math.atan2(dx, dy) % (2 * math.pi)
    
    def get_length(self) -> float:
        """Calculate the length of the portal (entry to exit)."""
        dx = self.exit_x - self.entry_x
        dy = self.exit_y - self.entry_y
        return math.sqrt(dx * dx + dy * dy)
    
    @classmethod
    def from_latlon(
        cls,
        entry_lat: float, entry_lng: float,
        exit_lat: float, exit_lng: float,
        transform: 'CoordinateTransform',
        structure_type: StructureType,
        entry_tolerance: float = 50.0
    ) -> 'Portal':
        """Create a portal from lat/lng coordinates."""
        entry_x, entry_y = transform.to_local(entry_lat, entry_lng)
        exit_x, exit_y = transform.to_local(exit_lat, exit_lng)
        return cls(
            entry_x=entry_x, entry_y=entry_y,
            exit_x=exit_x, exit_y=exit_y,
            structure_type=structure_type,
            entry_tolerance=entry_tolerance
        )


@dataclass
class PortalRegistry:
    """
    Registry of all manual tunnel/bridge portals.
    """
    portals: List[Portal] = field(default_factory=list)
    
    def add_tunnel(
        self,
        entry_x: float, entry_y: float,
        exit_x: float, exit_y: float,
        entry_tolerance: float = 50.0
    ):
        """Add a tunnel portal."""
        self.portals.append(Portal(
            entry_x=entry_x, entry_y=entry_y,
            exit_x=exit_x, exit_y=exit_y,
            structure_type=StructureType.TUNNEL,
            entry_tolerance=entry_tolerance
        ))
    
    def add_bridge(
        self,
        entry_x: float, entry_y: float,
        exit_x: float, exit_y: float,
        entry_tolerance: float = 50.0
    ):
        """Add a bridge portal."""
        self.portals.append(Portal(
            entry_x=entry_x, entry_y=entry_y,
            exit_x=exit_x, exit_y=exit_y,
            structure_type=StructureType.BRIDGE,
            entry_tolerance=entry_tolerance
        ))
    
    def find_available_portals(self, x: float, y: float) -> List[Portal]:
        """Find all portals that can be entered from the given position."""
        return [p for p in self.portals if p.is_at_entry(x, y)]


# =============================================================================
# NEIGHBOR RESULT
# =============================================================================

@dataclass
class Neighbor:
    """
    Result of generating a neighbor state.
    
    Contains the new state, the cost to reach it, and metadata about the transition.
    """
    state: State
    cost: float  # G-cost increment to reach this neighbor
    primitive: Optional[MotionPrimitive] = None  # The motion primitive used (if any)
    structure_type: StructureType = StructureType.NORMAL
    
    # For debugging/visualization
    is_switchback: bool = False
    is_portal: bool = False
    is_auto_structure: bool = False  # Auto-detected tunnel/bridge


# =============================================================================
# NEIGHBOR GENERATOR
# =============================================================================

class NeighborGenerator:
    """
    Generates valid neighbors from a state, applying all constraints.
    
    Handles three types of moves:
    a. Standard Motion (Rails): Left arc, Right arc, Straight
    b. Special Motion: Switchbacks
    c. Special Motion: Structures (Portals & Auto-Detection)
    """
    
    def __init__(
        self,
        config: KinodynamicConfig,
        elevation_grid: ElevationGrid,
        constraints: ConstraintMasks,
        portals: PortalRegistry,
        primitives: Optional[MotionPrimitiveSet] = None
    ):
        self.config = config
        self.elevation_grid = elevation_grid
        self.constraints = constraints
        self.portals = portals
        self.primitives = primitives or MotionPrimitiveSet.from_config(config)
        
        # Cache for failed moves (for auto-structure detection)
        self._last_failed_moves: List[Tuple[State, MotionPrimitive, str]] = []
    
    def get_neighbors(
        self,
        state: State,
        goal_x: Optional[float] = None,
        goal_y: Optional[float] = None
    ) -> List[Neighbor]:
        """
        Generate all valid neighbors from the given state.
        
        Args:
            state: Current state
            goal_x, goal_y: Optional goal coordinates for auto-tunnel/bridge detection.
                           If provided, auto-structures are generated when no neighbors
                           move toward the goal (not just when zero neighbors exist).
        
        Returns a list of Neighbor objects, each containing:
        - The new state
        - The cost to reach it
        - Metadata about the transition
        """
        neighbors: List[Neighbor] = []
        self._last_failed_moves.clear()
        
        # === a. Standard Motion (Rails) ===
        standard_neighbors = self._generate_standard_moves(state)
        neighbors.extend(standard_neighbors)
        
        # === b. Special Motion: Switchbacks ===
        if self.config.allow_switchbacks:
            switchback_neighbor = self._generate_switchback(state)
            if switchback_neighbor is not None:
                neighbors.append(switchback_neighbor)
        
        # === c. Special Motion: Structures ===
        # c1. Manual Portals
        portal_neighbors = self._generate_portal_moves(state)
        neighbors.extend(portal_neighbors)
        
        # c2. Auto-Detection: Tunnel/Bridge when stuck
        if self.config.auto_tunnel_bridge:
            should_try_auto_structure = False
            
            if len(standard_neighbors) == 0:
                # No valid standard moves at all
                should_try_auto_structure = True
            elif goal_x is not None and goal_y is not None:
                # Check if ANY neighbor moves toward the goal
                current_dist = math.sqrt((goal_x - state.x)**2 + (goal_y - state.y)**2)
                has_move_toward_goal = any(
                    math.sqrt((goal_x - n.state.x)**2 + (goal_y - n.state.y)**2) < current_dist
                    for n in standard_neighbors
                )
                if not has_move_toward_goal:
                    # All moves go away from goal - try auto-structure
                    should_try_auto_structure = True
            
            if should_try_auto_structure:
                auto_structure_neighbors = self._generate_auto_structures(state, goal_x, goal_y)
                neighbors.extend(auto_structure_neighbors)
        
        return neighbors
    
    # =========================================================================
    # a. STANDARD MOTION (RAILS)
    # =========================================================================
    
    def _generate_standard_moves(self, state: State) -> List[Neighbor]:
        """
        Generate neighbors using standard motion primitives.
        
        Applies terrain constraints and road parallelism penalties.
        """
        neighbors: List[Neighbor] = []
        
        for primitive in self.primitives.primitives:
            if primitive.is_switchback:
                continue  # Handle separately
            
            # Apply the primitive to get new state
            new_state = primitive.apply(state, self.config)
            if new_state is None:
                continue  # Out of bounds
            
            # Sample positions along the arc for constraint checking
            positions = primitive.sample_positions(state, num_samples=5)
            
            # === Check Terrain Constraints ===
            terrain_result = self._check_terrain_constraints(state, new_state, positions, primitive)
            if terrain_result is None:
                # Failed terrain check - record for auto-structure detection
                self._last_failed_moves.append((state, primitive, "terrain"))
                continue
            
            base_cost, max_slope, in_tunnel, in_bridge = terrain_result
            
            # === Check Water Constraints ===
            water_result = self._check_water_constraints(positions, in_bridge)
            if water_result is None:
                self._last_failed_moves.append((state, primitive, "water"))
                continue
            
            water_cost = water_result
            
            # === Road Parallelism / Repulsion ===
            road_cost = self._calculate_road_cost(state, new_state, positions)
            
            # === Calculate Total Cost ===
            curvature_cost = self._calculate_curvature_cost(primitive)
            
            total_cost = base_cost + water_cost + road_cost + curvature_cost
            
            # Determine structure type
            if in_tunnel:
                structure_type = StructureType.TUNNEL
            elif in_bridge:
                structure_type = StructureType.BRIDGE
            else:
                structure_type = StructureType.NORMAL
            
            neighbors.append(Neighbor(
                state=new_state,
                cost=total_cost,
                primitive=primitive,
                structure_type=structure_type,
                is_switchback=False,
                is_portal=False,
                is_auto_structure=False
            ))
        
        return neighbors
    
    def _check_terrain_constraints(
        self,
        from_state: State,
        to_state: State,
        positions: List[Tuple[float, float]],
        primitive: MotionPrimitive
    ) -> Optional[Tuple[float, float, bool, bool]]:
        """
        Check terrain (slope) constraints along a motion primitive.
        
        Returns:
            (base_cost, max_slope, in_tunnel, in_bridge) if valid
            None if the move violates hard constraints
        """
        max_slope = 0.0
        total_elevation_gain = 0.0
        in_tunnel = False
        in_bridge = False
        
        for i in range(1, len(positions)):
            x1, y1 = positions[i - 1]
            x2, y2 = positions[i]
            
            # Check if in tunnel/bridge zone
            if self.constraints.is_tunnel_zone(x1, y1) and self.constraints.is_tunnel_zone(x2, y2):
                in_tunnel = True
            if self.constraints.is_bridge_zone(x1, y1) and self.constraints.is_bridge_zone(x2, y2):
                in_bridge = True
            
            # Calculate slope
            slope = abs(self.elevation_grid.get_slope_along_segment(x1, y1, x2, y2))
            max_slope = max(max_slope, slope)
            
            # Calculate elevation gain
            elev1 = self.elevation_grid.get_elevation(x1, y1)
            elev2 = self.elevation_grid.get_elevation(x2, y2)
            if elev1 != float('inf') and elev2 != float('inf'):
                if elev2 > elev1:
                    total_elevation_gain += (elev2 - elev1)
        
        # Apply slope constraints
        # In tunnel zones, we allow steeper grades (rack railway support)
        effective_hard_limit = 15.0 if in_tunnel else self.config.hard_slope_limit_percent
        
        # For now, make hard slope limit a very high penalty instead of blocking
        # This helps debugging and ensures we can find *some* path
        hard_slope_exceeded = max_slope > effective_hard_limit
        
        # Calculate base cost
        distance_cost = primitive.arc_length * self.config.distance_weight
        elevation_cost = total_elevation_gain * self.config.elevation_gain_weight
        
        # Soft slope penalty
        if max_slope > self.config.max_slope_percent and not in_tunnel:
            slope_overage = max_slope - self.config.max_slope_percent
            slope_penalty = slope_overage * self.config.slope_penalty_multiplier * primitive.arc_length
        else:
            slope_penalty = 0.0
        
        # Add extreme penalty for exceeding hard slope limit (instead of blocking)
        if hard_slope_exceeded:
            slope_penalty += 50000.0  # Very high penalty but not blocking
        
        # Structure construction costs
        structure_cost = 0.0
        if in_tunnel:
            structure_cost = self.config.tunnel_cost_per_m * primitive.arc_length
        elif in_bridge:
            structure_cost = self.config.bridge_cost_per_m * primitive.arc_length
        
        base_cost = distance_cost + elevation_cost + slope_penalty + structure_cost
        
        return (base_cost, max_slope, in_tunnel, in_bridge)
    
    def _check_water_constraints(
        self,
        positions: List[Tuple[float, float]],
        in_bridge: bool
    ) -> Optional[float]:
        """
        Check water constraints along a path.
        
        Returns:
            Additional water cost if valid (0 if no water, or bridge zone)
            High penalty if crosses water without a bridge (soft constraint when allow_water_crossing=True)
            None if water crossing is blocked (hard constraint when allow_water_crossing=False)
        """
        water_cost = 0.0
        water_points = 0
        
        for x, y in positions:
            if self.constraints.is_water(x, y):
                if in_bridge:
                    # Water crossing allowed in bridge zone
                    pass
                elif self.config.allow_water_crossing:
                    # Water crossing with penalty (soft constraint)
                    water_points += 1
                    water_cost += self.config.water_penalty
                else:
                    # Water crossing blocked (hard constraint)
                    return None
        
        # If allowing water crossing, check if it's too wide
        if water_points > 0 and self.config.allow_water_crossing:
            # Estimate water width: each position sample is roughly step_distance / 10
            water_width_estimate = water_points * (self.config.step_distance_m / 10)
            if water_width_estimate > self.config.max_water_crossing_m:
                # Too wide to cross without a bridge
                return None
        
        return water_cost  # Return penalty (may be 0 if no water)
    
    def _calculate_road_cost(
        self,
        from_state: State,
        to_state: State,
        positions: List[Tuple[float, float]]
    ) -> float:
        """
        Calculate road parallelism / repulsion cost.
        
        Applies penalties when:
        - Too close to road (< min_separation) → High repulsion
        - Near road but crossing at sharp angle → Repulsion
        """
        if not self.config.road_parallel_enabled:
            return 0.0
        
        total_penalty = 0.0
        
        for x, y in positions:
            road_dist, road_dir = self.constraints.get_road_info(x, y)
            
            if road_dist == float('inf') or math.isnan(road_dir):
                continue  # No road data at this position
            
            # Calculate path heading at this point
            path_heading = to_state.heading
            
            # Calculate crossing angle (angle between path and road)
            crossing_angle = heading_difference(path_heading, road_dir)
            # Also consider anti-parallel as parallel
            if crossing_angle > math.pi / 2:
                crossing_angle = math.pi - crossing_angle
            
            crossing_angle_deg = math.degrees(crossing_angle)
            threshold_deg = self.config.road_parallel_threshold_deg
            
            # Check constraints
            if road_dist < self.config.road_min_separation_m:
                # Too close to road - HIGH repulsion penalty
                total_penalty += self.config.road_parallel_penalty * 2.0
            
            elif road_dist < self.config.road_max_separation_m:
                # Within influence zone
                if crossing_angle_deg > threshold_deg:
                    # Crossing road at sharp angle - repulsion
                    angle_factor = (crossing_angle_deg - threshold_deg) / (90.0 - threshold_deg)
                    total_penalty += self.config.road_parallel_penalty * angle_factor
                else:
                    # Nearly parallel and at good distance - slight bonus
                    # (Negative penalty)
                    ideal_dist = (self.config.road_min_separation_m + self.config.road_max_separation_m) / 2
                    if abs(road_dist - ideal_dist) < 10:
                        total_penalty -= self.config.road_parallel_penalty * 0.2
        
        return max(0.0, total_penalty)  # Ensure non-negative
    
    def _calculate_curvature_cost(self, primitive: MotionPrimitive) -> float:
        """Calculate cost based on curvature (tighter curves are more expensive)."""
        if abs(primitive.curvature) < 1e-9:
            return 0.0  # Straight - no curvature cost
        
        # Curvature cost scales with curvature magnitude
        # Sharper curves (higher κ) are more expensive
        curvature_magnitude = abs(primitive.curvature)
        max_curvature = 1.0 / self.config.min_curve_radius_m if self.config.min_curve_radius_m > 0 else 1.0
        
        # Normalize to [0, 1] and apply weight
        normalized_curvature = curvature_magnitude / max_curvature
        
        return self.config.curvature_weight * primitive.arc_length * normalized_curvature
    
    # =========================================================================
    # b. SPECIAL MOTION: SWITCHBACKS
    # =========================================================================
    
    def _generate_switchback(self, state: State) -> Optional[Neighbor]:
        """
        Generate a switchback neighbor if conditions are met.
        
        A switchback reverses the direction of travel (Forward ↔ Reverse)
        while staying at (approximately) the same position.
        """
        # Check minimum distance constraint
        if state.distance_since_switchback() < self.config.min_switchback_distance_m:
            return None
        
        # Use the switchback primitive if available
        if self.primitives.switchback_primitive is None:
            return None
        
        new_state = self.primitives.switchback_primitive.apply(state, self.config)
        if new_state is None:
            return None
        
        # Switchback cost
        cost = self.config.switchback_penalty
        
        # Add small distance cost for the movement after switchback
        cost += self.primitives.switchback_primitive.arc_length * self.config.distance_weight
        
        return Neighbor(
            state=new_state,
            cost=cost,
            primitive=self.primitives.switchback_primitive,
            structure_type=StructureType.NORMAL,
            is_switchback=True,
            is_portal=False,
            is_auto_structure=False
        )
    
    # =========================================================================
    # c. SPECIAL MOTION: STRUCTURES
    # =========================================================================
    
    def _generate_portal_moves(self, state: State) -> List[Neighbor]:
        """
        Generate moves through manual portals (tunnels/bridges).
        
        If the current state is at a portal entry, create a neighbor at the exit.
        """
        neighbors: List[Neighbor] = []
        
        available_portals = self.portals.find_available_portals(state.x, state.y)
        
        for portal in available_portals:
            # Calculate exit heading (direction from entry to exit)
            exit_heading = portal.get_exit_heading()
            portal_length = portal.get_length()
            
            # Create state at portal exit
            new_state = State(
                x=portal.exit_x,
                y=portal.exit_y,
                heading=exit_heading,
                direction_gear=state.direction_gear,  # Preserve direction
                last_switchback_x=state.last_switchback_x,
                last_switchback_y=state.last_switchback_y,
                _position_resolution=state._position_resolution,
                _heading_resolution=state._heading_resolution
            )
            
            # Check bounds
            if not self.config.is_in_bounds(new_state.x, new_state.y):
                continue
            
            # Calculate cost (based on structure type)
            if portal.structure_type == StructureType.TUNNEL:
                cost = portal_length * self.config.tunnel_cost_per_m
            else:  # BRIDGE
                cost = portal_length * self.config.bridge_cost_per_m
            
            # Add base distance cost
            cost += portal_length * self.config.distance_weight
            
            neighbors.append(Neighbor(
                state=new_state,
                cost=cost,
                primitive=None,  # No primitive for portal moves
                structure_type=portal.structure_type,
                is_switchback=False,
                is_portal=True,
                is_auto_structure=False
            ))
        
        return neighbors
    
    def _generate_auto_structures(
        self,
        state: State,
        goal_x: Optional[float] = None,
        goal_y: Optional[float] = None
    ) -> List[Neighbor]:
        """
        Auto-detect possible tunnel/bridge when standard moves fail.
        
        Attempts "long jump" projections to find reachable positions
        at similar elevation. If goal is provided, prioritizes directions
        toward the goal.
        """
        neighbors: List[Neighbor] = []
        
        # Determine search heading: prefer toward goal if available
        if goal_x is not None and goal_y is not None:
            # Calculate heading toward goal
            dx = goal_x - state.x
            dy = goal_y - state.y
            goal_heading = math.atan2(dx, dy)  # atan2(x, y) for our coord system
            base_heading = goal_heading
        else:
            # Fall back to current heading
            base_heading = state.heading
        
        # Try multiple headings (primarily toward goal/forward, plus deviations)
        heading_offsets = [0, -0.2, 0.2, -0.4, 0.4, -0.6, 0.6]  # radians (~0°, ±11°, ±23°, ±34°)
        
        found_structures = []  # Track what we find for logging
        
        for heading_offset in heading_offsets:
            search_heading = base_heading + heading_offset  # Use base_heading (toward goal if available)
            
            # Search at increasing distances
            for distance in range(
                int(self.config.step_distance_m * 2),
                int(self.config.max_jump_distance_m),
                int(self.config.step_distance_m)
            ):
                # Calculate target position (note: don't multiply by gear, we want to go toward goal)
                target_x = state.x + distance * math.sin(search_heading)
                target_y = state.y + distance * math.cos(search_heading)
                
                # Check bounds
                if not self.config.is_in_bounds(target_x, target_y):
                    continue
                
                # Check elevation difference
                start_elev = self.elevation_grid.get_elevation(state.x, state.y)
                end_elev = self.elevation_grid.get_elevation(target_x, target_y)
                
                if start_elev == float('inf') or end_elev == float('inf'):
                    continue
                
                elev_diff = abs(end_elev - start_elev)
                
                if elev_diff > self.config.elevation_tolerance_m:
                    continue  # Elevation difference too large
                
                # Determine structure type by analyzing terrain between
                structure_type = self._detect_structure_type(
                    state.x, state.y, target_x, target_y,
                    start_elev, end_elev
                )
                
                if structure_type is None:
                    continue  # Not a valid structure
                
                # Create new state
                new_state = State(
                    x=target_x,
                    y=target_y,
                    heading=search_heading,
                    direction_gear=state.direction_gear,
                    last_switchback_x=state.last_switchback_x,
                    last_switchback_y=state.last_switchback_y,
                    _position_resolution=state._position_resolution,
                    _heading_resolution=state._heading_resolution
                )
                
                # Calculate cost
                if structure_type == StructureType.TUNNEL:
                    cost = distance * self.config.tunnel_cost_per_m
                else:  # BRIDGE
                    cost = distance * self.config.bridge_cost_per_m
                
                # Add distance cost and heading deviation penalty
                cost += distance * self.config.distance_weight
                cost += abs(heading_offset) * 100  # Penalty for deviating from heading
                
                neighbors.append(Neighbor(
                    state=new_state,
                    cost=cost,
                    primitive=None,
                    structure_type=structure_type,
                    is_switchback=False,
                    is_portal=False,
                    is_auto_structure=True
                ))
                
                struct_name = "TUNNEL" if structure_type == StructureType.TUNNEL else "BRIDGE"
                found_structures.append(f"{struct_name} {distance}m")
        
        # Only log summary at stall diagnostic time, not during normal search
        # (Logging happens in _diagnose_stall instead)
        
        # Sort by cost and return top candidates
        neighbors.sort(key=lambda n: n.cost)
        return neighbors[:5]  # Limit to prevent explosion
    
    def _detect_structure_type(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        elev1: float, elev2: float
    ) -> Optional[StructureType]:
        """
        Detect whether a jump should be a tunnel or bridge.
        
        Returns:
            StructureType.TUNNEL if terrain between is higher
            StructureType.BRIDGE if terrain between is lower or water
            None if not a valid structure (terrain is similar)
        """
        avg_elev = (elev1 + elev2) / 2
        
        # Sample terrain between the two points
        num_samples = 10
        is_water_crossing = False
        max_elev_between = avg_elev
        min_elev_between = avg_elev
        
        for i in range(1, num_samples):
            t = i / num_samples
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            # Check water
            if self.constraints.is_water(x, y):
                is_water_crossing = True
            
            # Check elevation
            elev = self.elevation_grid.get_elevation(x, y)
            if elev != float('inf'):
                max_elev_between = max(max_elev_between, elev)
                min_elev_between = min(min_elev_between, elev)
        
        # Determine structure type
        if is_water_crossing:
            return StructureType.BRIDGE
        
        # If terrain rises significantly above endpoints → TUNNEL
        if max_elev_between > avg_elev + 5:
            return StructureType.TUNNEL
        
        # If terrain drops significantly below endpoints → BRIDGE
        if min_elev_between < avg_elev - 5:
            return StructureType.BRIDGE
        
        # Terrain is similar - check if slope was the issue
        # If max slope between points exceeds hard limit, it's a valid structure
        slope = abs(elev2 - elev1) / math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * 100
        if slope < self.config.max_slope_percent:
            return None  # No structure needed
        
        # Default to tunnel for steep terrain
        return StructureType.TUNNEL


# =============================================================================
# A* Search Implementation
# =============================================================================

@dataclass
class PathNode:
    """Node in the A* search tree."""
    state: State
    g_cost: float  # Cost from start to this node
    f_cost: float  # g_cost + heuristic (total estimated cost)
    parent: Optional['PathNode'] = None
    primitive: Optional[MotionPrimitive] = None  # How we got here
    structure_type: Optional[StructureType] = None
    is_switchback: bool = False
    is_portal: bool = False
    is_auto_structure: bool = False
    
    def __lt__(self, other: 'PathNode') -> bool:
        """For priority queue ordering."""
        return self.f_cost < other.f_cost


@dataclass
class PathSegment:
    """A segment of the final path with detailed information."""
    x: float
    y: float
    heading: float  # Degrees
    elevation: float
    curvature: float  # 1/m, 0 for straight
    direction_gear: DirectionGear
    structure_type: Optional[StructureType] = None
    is_switchback: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'x': self.x,
            'y': self.y,
            'heading': self.heading,
            'elevation': self.elevation,
            'curvature': self.curvature,
            'direction': 'forward' if self.direction_gear == DirectionGear.FORWARD else 'reverse',
            'structure': self.structure_type.value if self.structure_type else None,
            'is_switchback': self.is_switchback
        }


@dataclass
class PathResult:
    """Result of the pathfinding operation."""
    success: bool
    path: List[PathSegment]
    total_cost: float
    iterations: int
    nodes_expanded: int
    elapsed_time: float
    message: str
    # Failure info (populated when success=False)
    failure_x: Optional[float] = None  # Local X coordinate where search failed
    failure_y: Optional[float] = None  # Local Y coordinate where search failed
    failure_segment: Optional[int] = None  # Which segment (0-indexed) failed
    best_distance_remaining: Optional[float] = None  # Distance to goal when failed
    # Bidirectional search backward failure location
    failure_x_backward: Optional[float] = None  # Where backward search got stuck
    failure_y_backward: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'success': self.success,
            'path': [seg.to_dict() for seg in self.path],
            'total_cost': self.total_cost,
            'iterations': self.iterations,
            'nodes_expanded': self.nodes_expanded,
            'elapsed_time': self.elapsed_time,
            'message': self.message
        }
        if not self.success:
            result['failure_x'] = self.failure_x
            result['failure_y'] = self.failure_y
            result['failure_segment'] = self.failure_segment
            result['best_distance_remaining'] = self.best_distance_remaining
            result['failure_x_backward'] = self.failure_x_backward
            result['failure_y_backward'] = self.failure_y_backward
        return result


class KinodynamicPathfinder:
    """
    Kinodynamic A* pathfinder with continuous state space.
    
    Uses motion primitives (arcs and lines) to search through
    (x, y, heading, direction_gear) state space while respecting
    curvature, slope, and terrain constraints.
    """
    
    def __init__(
        self,
        config: KinodynamicConfig,
        elevation_grid: ElevationGrid,
        constraints: ConstraintMasks,
        portal_registry: Optional[PortalRegistry] = None,
        road_geometry: Optional[np.ndarray] = None
    ):
        self.config = config
        self.elevation_grid = elevation_grid
        self.constraints = constraints
        self.portal_registry = portal_registry or PortalRegistry()
        self.road_geometry = road_geometry
        
        # Create motion primitive set from config
        self.primitives = MotionPrimitiveSet.from_config(config)
        
        # Create neighbor generator
        self.neighbor_generator = NeighborGenerator(
            config=config,
            elevation_grid=elevation_grid,
            constraints=constraints,
            portals=self.portal_registry,
            primitives=self.primitives
        )
    
    def heuristic(self, state: State, goal_x: float, goal_y: float, weight_override: float = None) -> float:
        """
        Weighted heuristic for Hybrid A*: Euclidean distance to goal * weight.
        
        Using heuristic_weight > 1.0 makes search greedier (epsilon-admissible).
        This sacrifices optimality for speed - finds solutions faster but
        they may be up to (1 + epsilon) times the optimal cost.
        
        With heuristic_weight = 1.5, we accept paths up to 50% longer than optimal
        in exchange for much faster search convergence.
        
        Args:
            weight_override: If provided, use this weight instead of config default.
                            Used for stall recovery to temporarily reduce greediness.
        """
        dx = goal_x - state.x
        dy = goal_y - state.y
        dist = math.sqrt(dx * dx + dy * dy)
        # Apply weighted heuristic for greedier search
        weight = weight_override if weight_override is not None else self.config.heuristic_weight
        return dist * self.config.distance_weight * weight
    
    def is_goal(self, state: State, goal_x: float, goal_y: float, tolerance: float = 10.0) -> bool:
        """Check if state is close enough to goal."""
        dx = goal_x - state.x
        dy = goal_y - state.y
        return dx * dx + dy * dy <= tolerance * tolerance
    
    def _diagnose_stall(
        self,
        current_best: 'PathNode',
        goal_x: float,
        goal_y: float,
        initial_distance: float,
        best_distance_so_far: float
    ) -> None:
        """
        Perform detailed diagnostics when search is stalled.
        
        Analyzes:
        1. Terrain obstacles (slopes, ridges, valleys)
        2. Water obstacles
        3. Heading constraints (can't turn sharp enough)
        4. Whether auto-tunnel/bridge could help
        """
        state = current_best.state
        pos_x, pos_y = state.x, state.y
        
        # Convert to lat/lng for easy verification with Google Maps
        pos_lat, pos_lng = self.elevation_grid.transform.to_latlon(pos_x, pos_y)
        goal_lat, goal_lng = self.elevation_grid.transform.to_latlon(goal_x, goal_y)
        
        print(f"\n{'='*60}")
        print(f"[STALL DIAGNOSTIC] Position: ({pos_x:.0f}, {pos_y:.0f})")
        print(f"[STALL DIAGNOSTIC] Position (lat/lng): {pos_lat:.6f}, {pos_lng:.6f}")
        print(f"[STALL DIAGNOSTIC] Goal: ({goal_x:.0f}, {goal_y:.0f})")
        print(f"[STALL DIAGNOSTIC] Goal (lat/lng): {goal_lat:.6f}, {goal_lng:.6f}")
        print(f"[STALL DIAGNOSTIC] Progress: {(1 - best_distance_so_far/initial_distance)*100:.1f}%")
        print(f"[STALL DIAGNOSTIC] Current heading: {math.degrees(state.heading):.1f}°, gear: {state.direction_gear}")
        
        # Calculate direction to goal
        dx = goal_x - pos_x
        dy = goal_y - pos_y
        goal_heading = math.degrees(math.atan2(dx, dy)) % 360
        heading_diff = abs(goal_heading - math.degrees(state.heading) % 360)
        if heading_diff > 180:
            heading_diff = 360 - heading_diff
        print(f"[STALL DIAGNOSTIC] Heading to goal: {goal_heading:.1f}° (diff: {heading_diff:.1f}°)")
        
        # Check elevation at current position
        elev_here = self.elevation_grid.get_elevation(pos_x, pos_y)
        print(f"[STALL DIAGNOSTIC] Elevation at stall: {elev_here:.1f}m")
        
        # Sample elevations toward goal to detect terrain obstacles
        step = self.config.step_distance_m
        print(f"[STALL DIAGNOSTIC] Terrain scan toward goal:")
        
        obstacles_found = []
        prev_sample_t = 0.0
        prev_elev = elev_here
        for i in range(1, 6):  # Sample 5 steps toward goal
            sample_t = i / 5 * min(best_distance_so_far, step * 5)
            sample_x = pos_x + dx * sample_t / best_distance_so_far
            sample_y = pos_y + dy * sample_t / best_distance_so_far
            
            sample_elev = self.elevation_grid.get_elevation(sample_x, sample_y)
            is_water = self.constraints.is_water(sample_x, sample_y)
            is_bridge = self.constraints.is_bridge_zone(sample_x, sample_y)
            is_tunnel = self.constraints.is_tunnel_zone(sample_x, sample_y)
            
            # Calculate slope from previous point
            slope_rise = sample_elev - prev_elev
            slope_run = sample_t - prev_sample_t  # Distance between consecutive samples
            slope_pct = abs(slope_rise / slope_run * 100) if slope_run > 0 else 0
            
            status = []
            if is_water and not is_bridge:
                status.append("🌊 WATER")
                obstacles_found.append(f"Water at {sample_t:.0f}m")
            if slope_pct > self.config.hard_slope_limit_percent:
                status.append(f"⛰️ STEEP {slope_pct:.1f}%")
                obstacles_found.append(f"Steep slope ({slope_pct:.1f}%) at {sample_t:.0f}m")
            elif slope_pct > self.config.max_slope_percent:
                status.append(f"📈 slope {slope_pct:.1f}%")
            if is_tunnel:
                status.append("🚇 tunnel-zone")
            if is_bridge:
                status.append("🌉 bridge-zone")
            
            status_str = " ".join(status) if status else "clear"
            print(f"[STALL DIAGNOSTIC]   +{sample_t:.0f}m: elev={sample_elev:.1f}m, {status_str}")
            prev_elev = sample_elev
            prev_sample_t = sample_t
        
        # Check neighbors and why they aren't helping
        neighbors = self.neighbor_generator.get_neighbors(state, goal_x, goal_y)
        print(f"[STALL DIAGNOSTIC] Neighbors: {len(neighbors)} valid moves")
        
        if len(neighbors) == 0:
            print(f"[STALL DIAGNOSTIC] ⚠️ NO VALID MOVES! Checking why...")
            # Check each primitive directly to see what's blocking
            for prim in self.primitives.primitives[:5]:
                new_state = prim.apply(state, self.config)
                if new_state is None:
                    print(f"[STALL DIAGNOSTIC]   {prim.curvature:.4f} curvature: OUT OF BOUNDS")
                    continue
                # Check terrain
                positions = prim.sample_positions(state, num_samples=5)
                for px, py in positions:
                    if self.constraints.is_water(px, py):
                        print(f"[STALL DIAGNOSTIC]   {prim.curvature:.4f} curvature: BLOCKED BY WATER at ({px:.0f},{py:.0f})")
                        break
                    slope = abs(self.elevation_grid.get_slope_along_segment(
                        positions[0][0], positions[0][1], px, py
                    ))
                    if slope > self.config.hard_slope_limit_percent:
                        print(f"[STALL DIAGNOSTIC]   {prim.curvature:.4f} curvature: BLOCKED BY SLOPE {slope:.1f}%")
                        break
        else:
            # Analyze why neighbors don't improve distance
            print(f"[STALL DIAGNOSTIC] Neighbor analysis:")
            toward_goal = 0
            away_from_goal = 0
            for n in neighbors:
                n_dist = math.sqrt((goal_x - n.state.x)**2 + (goal_y - n.state.y)**2)
                if n_dist < best_distance_so_far:
                    toward_goal += 1
                else:
                    away_from_goal += 1
            print(f"[STALL DIAGNOSTIC]   {toward_goal} moves toward goal, {away_from_goal} moves away")
            
            # Show the actual neighbor headings and distances
            if toward_goal == 0:
                print(f"[STALL DIAGNOSTIC]   Detailed neighbor info (first 5):")
                for i, n in enumerate(neighbors[:5]):
                    n_dist = math.sqrt((goal_x - n.state.x)**2 + (goal_y - n.state.y)**2)
                    delta_dist = n_dist - best_distance_so_far
                    heading_deg = math.degrees(n.state.heading) % 360
                    print(f"[STALL DIAGNOSTIC]     #{i+1}: heading={heading_deg:.1f}°, dist to goal={n_dist:.0f}m (Δ={delta_dist:+.0f}m), cost={n.cost:.0f}")
            
            if toward_goal > 0:
                print(f"[STALL DIAGNOSTIC]   ⚠️ Moves toward goal exist but aren't being selected!")
                print(f"[STALL DIAGNOSTIC]   This suggests the weighted heuristic is causing issues")
                # Show best neighbor toward goal
                best_toward = min(
                    [n for n in neighbors if math.sqrt((goal_x - n.state.x)**2 + (goal_y - n.state.y)**2) < best_distance_so_far],
                    key=lambda n: math.sqrt((goal_x - n.state.x)**2 + (goal_y - n.state.y)**2),
                    default=None
                )
                if best_toward:
                    bt_dist = math.sqrt((goal_x - best_toward.state.x)**2 + (goal_y - best_toward.state.y)**2)
                    print(f"[STALL DIAGNOSTIC]   Best toward-goal move: dist={bt_dist:.0f}m, cost={best_toward.cost:.1f}")
        
        # Summarize obstacles
        if obstacles_found:
            print(f"[STALL DIAGNOSTIC] 🚧 OBSTACLES DETECTED:")
            for obs in obstacles_found:
                print(f"[STALL DIAGNOSTIC]   - {obs}")
            
            # Suggest solutions
            if any("Water" in o for o in obstacles_found):
                print(f"[STALL DIAGNOSTIC] 💡 TIP: Add a bridge marker to cross water, or enable water crossing")
            if any("Steep" in o for o in obstacles_found):
                print(f"[STALL DIAGNOSTIC] 💡 TIP: Add a tunnel marker, increase hard_slope_limit, or enable switchbacks")
        
        print(f"{'='*60}\n")

    def try_analytic_expansion(
        self, 
        current: 'PathNode',
        goal_x: float, 
        goal_y: float,
        step_size: float
    ) -> Optional['PathNode']:
        """
        Try to connect directly to goal with a straight or curved path.
        
        This is the "shot-to-goal" optimization in Hybrid A*:
        When we're close to the goal, try a direct connection instead of
        continuing the search. If the direct path is obstacle-free, we
        terminate early.
        
        Args:
            current: Current path node
            goal_x, goal_y: Goal position
            step_size: Current step size for sampling
            
        Returns:
            PathNode at goal if direct connection works, None otherwise
        """
        dx = goal_x - current.state.x
        dy = goal_y - current.state.y
        dist = math.sqrt(dx * dx + dy * dy)
        
        # Only try if within analytic expansion distance
        expansion_threshold = step_size * self.config.analytic_expansion_distance
        if dist > expansion_threshold:
            return None
        
        # Calculate heading to goal
        goal_heading = math.atan2(dx, dy)  # atan2(dx, dy) for north-up
        
        # Check if we can reach goal with current direction gear
        # (Don't switch direction in the final approach)
        heading_diff = abs(self._angle_diff(current.state.heading, goal_heading))
        
        # If heading difference is too large, can't do direct connection
        max_turn = math.pi / 2  # 90 degrees max for direct shot
        if heading_diff > max_turn:
            return None
        
        # Sample points along the direct path to check for obstacles
        num_samples = max(3, int(dist / (step_size * 0.5)))
        
        for i in range(1, num_samples + 1):
            t = i / num_samples
            sample_x = current.state.x + dx * t
            sample_y = current.state.y + dy * t
            
            # Check bounds (get_elevation returns inf if out of bounds)
            if (sample_x < 0 or sample_x > self.elevation_grid.width_m or
                sample_y < 0 or sample_y > self.elevation_grid.height_m):
                return None
            
            # Check for hard constraints using ConstraintMasks methods
            # Block if in water (unless in bridge zone)
            if self.constraints.is_water(sample_x, sample_y):
                if not self.constraints.is_bridge_zone(sample_x, sample_y):
                    return None
            
            # Check slope along the path
            if i > 0:
                prev_t = (i - 1) / num_samples
                prev_x = current.state.x + dx * prev_t
                prev_y = current.state.y + dy * prev_t
                
                elev_prev = self.elevation_grid.get_elevation(prev_x, prev_y)
                elev_curr = self.elevation_grid.get_elevation(sample_x, sample_y)
                
                segment_dist = dist / num_samples
                if segment_dist > 0:
                    slope = (elev_curr - elev_prev) / segment_dist
                    slope_pct = abs(slope) * 100
                    
                    # Block if exceeds hard slope limit
                    if slope_pct > self.config.hard_slope_limit_percent:
                        return None
        
        # Direct path is clear! Create goal node
        goal_state = State(
            x=goal_x,
            y=goal_y,
            heading=goal_heading,
            direction_gear=current.state.direction_gear,
            last_switchback_x=current.state.last_switchback_x,
            last_switchback_y=current.state.last_switchback_y
        )
        
        # Calculate cost for direct segment
        direct_cost = dist * self.config.distance_weight
        
        goal_node = PathNode(
            state=goal_state,
            g_cost=current.g_cost + direct_cost,
            f_cost=current.g_cost + direct_cost,  # h=0 at goal
            parent=current,
            primitive=None,  # Direct connection, no primitive
            structure_type=None,
            is_switchback=False,
            is_portal=False,
            is_auto_structure=False
        )
        
        return goal_node
    
    def _angle_diff(self, a: float, b: float) -> float:
        """Calculate the smallest difference between two angles in radians."""
        diff = b - a
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def find_path(
        self,
        start_x: float, start_y: float,
        goal_x: float, goal_y: float,
        initial_heading: Optional[float] = None,
        initial_direction_gear: DirectionGear = DirectionGear.FORWARD,
        max_iterations: int = 100000,
        goal_tolerance: Optional[float] = None
    ) -> PathResult:
        """
        Find a path from start to goal using Hybrid A*.
        
        Args:
            start_x, start_y: Start position in local coordinates
            goal_x, goal_y: Goal position in local coordinates
            initial_heading: Starting heading in degrees (0=North). 
                           If None, computed from start→goal direction.
            initial_direction_gear: Starting direction gear.
            max_iterations: Maximum search iterations
            goal_tolerance: Distance threshold to consider goal reached.
                           If None, uses step_distance_m * goal_tolerance_multiplier.
            
        Returns:
            PathResult with success status, path segments, and statistics
        """
        # Compute goal tolerance from step size if not provided
        if goal_tolerance is None:
            goal_tolerance = self.config.step_distance_m * self.config.goal_tolerance_multiplier
        
        # Compute initial heading if not provided
        if initial_heading is None:
            dx = goal_x - start_x
            dy = goal_y - start_y
            initial_heading = math.degrees(math.atan2(dx, dy)) % 360
        
        # Delegate to internal segment finder
        return self._find_path_segment(
            start_x=start_x,
            start_y=start_y,
            goal_x=goal_x,
            goal_y=goal_y,
            initial_heading=initial_heading,
            initial_direction_gear=initial_direction_gear,
            last_switchback_x=None,
            last_switchback_y=None,
            max_iterations=max_iterations,
            goal_tolerance=goal_tolerance
        )
    
    def _reconstruct_path(self, goal_node: PathNode) -> List[PathSegment]:
        """
        Reconstruct the path from goal back to start.
        
        Uses the motion primitives' sample_positions to generate
        smooth intermediate points along arcs.
        """
        # Collect nodes from goal to start
        nodes: List[PathNode] = []
        current = goal_node
        while current is not None:
            nodes.append(current)
            current = current.parent
        
        # Reverse to get start-to-goal order
        nodes.reverse()
        
        path_segments: List[PathSegment] = []
        
        for i, node in enumerate(nodes):
            if i == 0:
                # Start node - just add the position
                elev = self.elevation_grid.get_elevation(node.state.x, node.state.y)
                if elev == float('inf'):
                    elev = 0.0
                
                path_segments.append(PathSegment(
                    x=node.state.x,
                    y=node.state.y,
                    heading=node.state.heading,
                    elevation=elev,
                    curvature=0.0,
                    direction_gear=node.state.direction_gear,
                    structure_type=None,
                    is_switchback=False
                ))
            else:
                parent = nodes[i - 1]
                
                # For portals and auto-structures, just add the endpoint
                if node.is_portal or node.is_auto_structure:
                    elev = self.elevation_grid.get_elevation(node.state.x, node.state.y)
                    if elev == float('inf'):
                        elev = 0.0
                    
                    path_segments.append(PathSegment(
                        x=node.state.x,
                        y=node.state.y,
                        heading=node.state.heading,
                        elevation=elev,
                        curvature=0.0,
                        direction_gear=node.state.direction_gear,
                        structure_type=node.structure_type,
                        is_switchback=False
                    ))
                
                elif node.is_switchback:
                    # Switchback: just flip at same position
                    elev = self.elevation_grid.get_elevation(node.state.x, node.state.y)
                    if elev == float('inf'):
                        elev = 0.0
                    
                    path_segments.append(PathSegment(
                        x=node.state.x,
                        y=node.state.y,
                        heading=node.state.heading,
                        elevation=elev,
                        curvature=0.0,
                        direction_gear=node.state.direction_gear,
                        structure_type=None,
                        is_switchback=True
                    ))
                
                elif node.primitive is not None:
                    # Motion primitive: sample positions along the arc
                    positions = node.primitive.sample_positions(
                        parent.state,
                        num_samples=5
                    )
                    
                    # Calculate heading increment for each sample
                    delta_heading_total = node.primitive.arc_length * node.primitive.curvature * float(parent.state.direction_gear)
                    delta_heading_per_step = delta_heading_total / 5
                    
                    # Skip first position (it's the parent)
                    for j, (px, py) in enumerate(positions[1:], 1):
                        sample_heading = parent.state.heading + delta_heading_per_step * j
                        sample_heading = math.degrees(sample_heading % (2 * math.pi))
                        
                        elev = self.elevation_grid.get_elevation(px, py)
                        if elev == float('inf'):
                            elev = 0.0
                        
                        # Mark the last position with any special flags
                        is_last = (j == len(positions) - 1)
                        
                        path_segments.append(PathSegment(
                            x=px,
                            y=py,
                            heading=sample_heading,
                            elevation=elev,
                            curvature=node.primitive.curvature,
                            direction_gear=node.state.direction_gear,
                            structure_type=node.structure_type if is_last else None,
                            is_switchback=False
                        ))
        
        return path_segments
    
    def find_path_through_waypoints(
        self,
        waypoints: List[Tuple[float, float]],
        waypoint_headings: Optional[List[Optional[float]]] = None,
        initial_heading: Optional[float] = None,
        initial_direction_gear: DirectionGear = DirectionGear.FORWARD,
        max_iterations_per_segment: int = 100000,
        goal_tolerance: Optional[float] = None
    ) -> PathResult:
        """
        Find a path through a sequence of waypoints.
        
        The path is computed segment by segment:
        Start -> Waypoint 1 -> Waypoint 2 -> ... -> End
        
        Heading and direction_gear are preserved at each transition to ensure
        curve continuity between segments.
        
        Args:
            waypoints: List of (x, y) tuples. Must have at least 2 points.
                      First point is start, last point is end.
            waypoint_headings: Optional list of headings (degrees) for each waypoint.
                              None values in the list mean auto-compute heading.
                              If entire list is None, all headings are auto-computed.
            initial_heading: Starting heading in degrees (0=North).
                           If None and waypoint_headings[0] is None, computed from 
                           start→first waypoint direction.
            initial_direction_gear: Starting direction gear.
            max_iterations_per_segment: Maximum iterations for each segment.
            goal_tolerance: Distance threshold to consider waypoint reached.
                           If None, uses step_distance_m * goal_tolerance_multiplier.
            
        Returns:
            PathResult with combined path from all segments.
        """
        # Compute goal tolerance from step size if not provided
        if goal_tolerance is None:
            goal_tolerance = self.config.step_distance_m * self.config.goal_tolerance_multiplier
        
        # Initialize waypoint_headings if not provided
        if waypoint_headings is None:
            waypoint_headings = [None] * len(waypoints)
        elif len(waypoint_headings) != len(waypoints):
            # Pad or truncate to match waypoints length
            waypoint_headings = list(waypoint_headings) + [None] * (len(waypoints) - len(waypoint_headings))
            waypoint_headings = waypoint_headings[:len(waypoints)]
            
        if len(waypoints) < 2:
            return PathResult(
                success=False,
                path=[],
                total_cost=float('inf'),
                iterations=0,
                nodes_expanded=0,
                elapsed_time=0.0,
                message="Need at least 2 waypoints (start and end)"
            )
        
        start_time = time.time()
        
        all_segments: List[PathSegment] = []
        total_cost = 0.0
        total_iterations = 0
        total_nodes_expanded = 0
        
        # Current state tracking for continuity
        current_heading = initial_heading
        current_direction_gear = initial_direction_gear
        current_last_switchback_x: Optional[float] = None
        current_last_switchback_y: Optional[float] = None
        
        # Process each segment
        for i in range(len(waypoints) - 1):
            start_x, start_y = waypoints[i]
            goal_x, goal_y = waypoints[i + 1]
            
            segment_name = f"Segment {i + 1}/{len(waypoints) - 1}"
            
            # Determine initial heading for this segment
            # Priority: 1) Explicitly set waypoint heading, 2) current_heading from previous segment, 3) auto-compute
            segment_heading = waypoint_headings[i]
            if segment_heading is not None:
                # Use explicitly set heading for this waypoint
                current_heading = segment_heading
            elif i == 0 and current_heading is None:
                # First segment with no heading: compute from start→goal direction
                dx = goal_x - start_x
                dy = goal_y - start_y
                current_heading = math.degrees(math.atan2(dx, dy)) % 360
            # else: use current_heading from previous segment (continuity)
            
            # Find path for this segment
            segment_result = self._find_path_segment(
                start_x=start_x,
                start_y=start_y,
                goal_x=goal_x,
                goal_y=goal_y,
                initial_heading=current_heading,
                initial_direction_gear=current_direction_gear,
                last_switchback_x=current_last_switchback_x,
                last_switchback_y=current_last_switchback_y,
                max_iterations=max_iterations_per_segment,
                goal_tolerance=goal_tolerance
            )
            
            if not segment_result.success:
                elapsed = time.time() - start_time
                # Combine already-completed segments with partial path from failed segment
                partial_path = all_segments.copy()
                if segment_result.path:
                    # Add the partial path from the failed segment
                    if partial_path:
                        # Skip first point to avoid duplicate
                        partial_path.extend(segment_result.path[1:])
                    else:
                        partial_path.extend(segment_result.path)
                
                return PathResult(
                    success=False,
                    path=partial_path,  # Return partial path including failed segment's progress
                    total_cost=total_cost,
                    iterations=total_iterations + segment_result.iterations,
                    nodes_expanded=total_nodes_expanded + segment_result.nodes_expanded,
                    elapsed_time=elapsed,
                    message=f"Failed at {segment_name}: {segment_result.message}",
                    failure_x=segment_result.failure_x,
                    failure_y=segment_result.failure_y,
                    failure_segment=i,
                    best_distance_remaining=segment_result.best_distance_remaining
                )
            
            # Accumulate statistics
            total_cost += segment_result.total_cost
            total_iterations += segment_result.iterations
            total_nodes_expanded += segment_result.nodes_expanded
            
            # Add segments (skip first point of subsequent segments to avoid duplicates)
            if i == 0:
                all_segments.extend(segment_result.path)
            else:
                # Skip the first segment point (it duplicates the last point of previous segment)
                all_segments.extend(segment_result.path[1:] if segment_result.path else [])
            
            # Extract end state for continuity to next segment
            if segment_result.path:
                last_segment = segment_result.path[-1]
                # Convert heading from degrees back to radians for internal use
                current_heading = last_segment.heading
                current_direction_gear = last_segment.direction_gear
                
                # Track switchback state - find the last switchback in this segment
                for seg in reversed(segment_result.path):
                    if seg.is_switchback:
                        current_last_switchback_x = seg.x
                        current_last_switchback_y = seg.y
                        break
        
        elapsed = time.time() - start_time
        
        return PathResult(
            success=True,
            path=all_segments,
            total_cost=total_cost,
            iterations=total_iterations,
            nodes_expanded=total_nodes_expanded,
            elapsed_time=elapsed,
            message=f"Path found through {len(waypoints)} waypoints in {total_iterations} iterations, {elapsed:.2f}s"
        )
    
    def _find_path_segment_bidirectional(
        self,
        start_x: float, start_y: float,
        goal_x: float, goal_y: float,
        initial_heading: float,
        initial_direction_gear: DirectionGear,
        last_switchback_x: Optional[float],
        last_switchback_y: Optional[float],
        max_iterations: int,
        goal_tolerance: float
    ) -> PathResult:
        """
        Bidirectional A* search - searches from both start and goal simultaneously.
        
        When the two searches meet, we've found a path that goes around obstacles
        that would otherwise cause one-sided search to get stuck.
        """
        start_time = time.time()
        
        # Convert heading to radians
        heading_rad = math.radians(initial_heading)
        
        # Calculate goal heading (reverse direction - pointing back toward start)
        dx = start_x - goal_x
        dy = start_y - goal_y
        goal_heading_rad = math.atan2(dx, dy)  # Heading from goal toward start
        
        # === FORWARD SEARCH (from start) ===
        forward_start = State(
            x=start_x, y=start_y,
            heading=heading_rad,
            direction_gear=initial_direction_gear,
            last_switchback_x=last_switchback_x,
            last_switchback_y=last_switchback_y
        )
        forward_open: List[Tuple[float, int, PathNode]] = []
        forward_closed: Set[Tuple[int, int, int, int]] = set()
        forward_best_g: Dict[Tuple[int, int, int, int], float] = {}
        forward_nodes: Dict[Tuple[int, int, int, int], PathNode] = {}  # For reconstruction
        
        h_forward = self.heuristic(forward_start, goal_x, goal_y)
        forward_node = PathNode(state=forward_start, g_cost=0.0, f_cost=h_forward)
        heapq.heappush(forward_open, (forward_node.f_cost, 0, forward_node))
        forward_counter = 1
        
        # === BACKWARD SEARCH (from goal) ===
        backward_start = State(
            x=goal_x, y=goal_y,
            heading=goal_heading_rad,
            direction_gear=initial_direction_gear,  # Same gear
            last_switchback_x=None,
            last_switchback_y=None
        )
        backward_open: List[Tuple[float, int, PathNode]] = []
        backward_closed: Set[Tuple[int, int, int, int]] = set()
        backward_best_g: Dict[Tuple[int, int, int, int], float] = {}
        backward_nodes: Dict[Tuple[int, int, int, int], PathNode] = {}  # For reconstruction
        
        h_backward = self.heuristic(backward_start, start_x, start_y)
        backward_node = PathNode(state=backward_start, g_cost=0.0, f_cost=h_backward)
        heapq.heappush(backward_open, (backward_node.f_cost, 0, backward_node))
        backward_counter = 1
        
        iterations = 0
        nodes_expanded = 0
        progress_interval = 10000
        
        initial_distance = math.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
        best_forward_dist = initial_distance
        best_backward_dist = initial_distance
        
        # Track best progress node (closest to goal) for partial path on failure
        best_forward_progress_node: Optional[PathNode] = forward_node
        best_forward_progress_dist: float = initial_distance
        
        # Track best backward progress node (closest to start) for partial path on failure
        best_backward_progress_node: Optional[PathNode] = backward_node
        best_backward_progress_dist: float = initial_distance
        
        # Meeting point tracking
        best_meeting_cost = float('inf')
        best_meeting_forward_node: Optional[PathNode] = None
        best_meeting_backward_node: Optional[PathNode] = None
        
        # Spatial index for quick meeting point detection
        # Uses coarser buckets than state discretization for meeting detection
        meeting_bucket_size = self.config.step_distance_m * 2
        
        def position_bucket(x: float, y: float) -> Tuple[int, int]:
            return (int(x / meeting_bucket_size), int(y / meeting_bucket_size))
        
        # Store only one node per bucket for efficiency
        forward_positions: Dict[Tuple[int, int], PathNode] = {}
        backward_positions: Dict[Tuple[int, int], PathNode] = {}
        
        # Track expansions for balanced alternation
        forward_expansions = 0
        backward_expansions = 0
        
        while (forward_open or backward_open) and iterations < max_iterations:
            iterations += 1
            
            # BALANCED alternation: ensure both searches get fair iterations
            # Use strict alternation, but skip if one search is empty
            expand_forward = True
            if forward_open and backward_open:
                # Alternate strictly, but if one side is way behind, give it more turns
                if forward_expansions > backward_expansions + 1000:
                    expand_forward = False
                elif backward_expansions > forward_expansions + 1000:
                    expand_forward = True
                else:
                    # Strict alternation
                    expand_forward = (iterations % 2 == 1)
            elif backward_open:
                expand_forward = False
            elif forward_open:
                expand_forward = True
            else:
                break  # Both queues empty
            
            if expand_forward:
                if not forward_open:
                    continue  # Skip this iteration, try backward next time
                _, _, current = heapq.heappop(forward_open)
                state_key = current.state._discretize()
                
                if state_key in forward_closed:
                    continue  # Already expanded, skip
                
                forward_closed.add(state_key)
                forward_nodes[state_key] = current
                nodes_expanded += 1
                forward_expansions += 1
                
                # Track position for meeting detection (only store one node per bucket to reduce memory/lookup)
                pos_bucket = position_bucket(current.state.x, current.state.y)
                # Only store if we don't already have a node in this bucket (first one is usually best)
                if pos_bucket not in forward_positions:
                    forward_positions[pos_bucket] = current
                
                # Check if we've reached the actual goal
                if self.is_goal(current.state, goal_x, goal_y, goal_tolerance):
                    path_segments = self._reconstruct_path(current)
                    elapsed = time.time() - start_time
                    return PathResult(
                        success=True, path=path_segments,
                        total_cost=current.g_cost, iterations=iterations,
                        nodes_expanded=nodes_expanded, elapsed_time=elapsed,
                        message=f"Segment found (forward reached goal) in {iterations} iterations"
                    )
                
                # Check for meeting with backward search (check nearby buckets too)
                for nearby_bucket in [pos_bucket,
                                      (pos_bucket[0]-1, pos_bucket[1]),
                                      (pos_bucket[0]+1, pos_bucket[1]),
                                      (pos_bucket[0], pos_bucket[1]-1),
                                      (pos_bucket[0], pos_bucket[1]+1)]:
                    if nearby_bucket in backward_positions:
                        back_node = backward_positions[nearby_bucket]
                        dist = math.sqrt(
                            (current.state.x - back_node.state.x)**2 +
                            (current.state.y - back_node.state.y)**2
                        )
                        if dist < goal_tolerance * 2:
                            # Found a meeting point!
                            meeting_cost = current.g_cost + back_node.g_cost + dist
                            if meeting_cost < best_meeting_cost:
                                best_meeting_cost = meeting_cost
                                best_meeting_forward_node = current
                                best_meeting_backward_node = back_node
                
                # Update best forward distance and track best progress node
                dist_to_goal = math.sqrt(
                    (goal_x - current.state.x)**2 + (goal_y - current.state.y)**2
                )
                if dist_to_goal < best_forward_progress_dist:
                    best_forward_progress_dist = dist_to_goal
                    best_forward_progress_node = current
                best_forward_dist = min(best_forward_dist, dist_to_goal)
                
                # Expand forward neighbors
                neighbors = self.neighbor_generator.get_neighbors(
                    current.state, goal_x, goal_y
                )
                for neighbor in neighbors:
                    new_g = current.g_cost + neighbor.cost
                    new_state_key = neighbor.state._discretize()
                    
                    if new_state_key in forward_closed:
                        continue
                    if new_state_key in forward_best_g and forward_best_g[new_state_key] <= new_g:
                        continue
                    
                    forward_best_g[new_state_key] = new_g
                    h = self.heuristic(neighbor.state, goal_x, goal_y)
                    new_f = new_g + h
                    
                    new_node = PathNode(
                        state=neighbor.state, g_cost=new_g, f_cost=new_f,
                        parent=current, primitive=neighbor.primitive,
                        structure_type=neighbor.structure_type,
                        is_switchback=neighbor.is_switchback,
                        is_portal=neighbor.is_portal,
                        is_auto_structure=neighbor.is_auto_structure
                    )
                    heapq.heappush(forward_open, (new_f, forward_counter, new_node))
                    forward_counter += 1
            
            else:  # Backward expansion
                if not backward_open:
                    continue  # Skip, try forward next time
                _, _, current = heapq.heappop(backward_open)
                state_key = current.state._discretize()
                
                if state_key in backward_closed:
                    continue  # Already expanded, skip
                
                backward_closed.add(state_key)
                backward_nodes[state_key] = current
                nodes_expanded += 1
                backward_expansions += 1
                
                # Track position for meeting detection (only store one node per bucket)
                pos_bucket = position_bucket(current.state.x, current.state.y)
                if pos_bucket not in backward_positions:
                    backward_positions[pos_bucket] = current
                
                # Check if backward reached the start
                if self.is_goal(current.state, start_x, start_y, goal_tolerance):
                    # Reconstruct backward path and reverse it
                    path_segments = self._reconstruct_path_reversed(current)
                    elapsed = time.time() - start_time
                    return PathResult(
                        success=True, path=path_segments,
                        total_cost=current.g_cost, iterations=iterations,
                        nodes_expanded=nodes_expanded, elapsed_time=elapsed,
                        message=f"Segment found (backward reached start) in {iterations} iterations"
                    )
                
                # Check for meeting with forward search (check nearby buckets too)
                for nearby_bucket in [pos_bucket,
                                      (pos_bucket[0]-1, pos_bucket[1]),
                                      (pos_bucket[0]+1, pos_bucket[1]),
                                      (pos_bucket[0], pos_bucket[1]-1),
                                      (pos_bucket[0], pos_bucket[1]+1)]:
                    if nearby_bucket in forward_positions:
                        fwd_node = forward_positions[nearby_bucket]
                        dist = math.sqrt(
                            (current.state.x - fwd_node.state.x)**2 +
                            (current.state.y - fwd_node.state.y)**2
                        )
                        if dist < goal_tolerance * 2:
                            meeting_cost = fwd_node.g_cost + current.g_cost + dist
                            if meeting_cost < best_meeting_cost:
                                best_meeting_cost = meeting_cost
                                best_meeting_forward_node = fwd_node
                                best_meeting_backward_node = current
                
                # Update best backward distance and track best backward progress node
                dist_to_start = math.sqrt(
                    (start_x - current.state.x)**2 + (start_y - current.state.y)**2
                )
                if dist_to_start < best_backward_progress_dist:
                    best_backward_progress_dist = dist_to_start
                    best_backward_progress_node = current
                best_backward_dist = min(best_backward_dist, dist_to_start)
                
                # Expand backward neighbors (toward start)
                neighbors = self.neighbor_generator.get_neighbors(
                    current.state, start_x, start_y
                )
                for neighbor in neighbors:
                    new_g = current.g_cost + neighbor.cost
                    new_state_key = neighbor.state._discretize()
                    
                    if new_state_key in backward_closed:
                        continue
                    if new_state_key in backward_best_g and backward_best_g[new_state_key] <= new_g:
                        continue
                    
                    backward_best_g[new_state_key] = new_g
                    h = self.heuristic(neighbor.state, start_x, start_y)
                    new_f = new_g + h
                    
                    new_node = PathNode(
                        state=neighbor.state, g_cost=new_g, f_cost=new_f,
                        parent=current, primitive=neighbor.primitive,
                        structure_type=neighbor.structure_type,
                        is_switchback=neighbor.is_switchback,
                        is_portal=neighbor.is_portal,
                        is_auto_structure=neighbor.is_auto_structure
                    )
                    heapq.heappush(backward_open, (new_f, backward_counter, new_node))
                    backward_counter += 1
            
            # Progress logging
            if iterations % progress_interval == 0:
                elapsed = time.time() - start_time
                forward_progress = (1 - best_forward_dist / initial_distance) * 100
                backward_progress = (1 - best_backward_dist / initial_distance) * 100
                total_progress = forward_progress + backward_progress
                
                meeting_info = ""
                if best_meeting_forward_node is not None:
                    meeting_info = f", meeting found (cost={best_meeting_cost:.0f})"
                
                print(f"[Kinodynamic-Bidir] {iterations:,} iters, "
                      f"fwd:{forward_expansions:,}exp/{len(forward_open):,}q/{forward_progress:.1f}%, "
                      f"bwd:{backward_expansions:,}exp/{len(backward_open):,}q/{backward_progress:.1f}%, "
                      f"{elapsed:.1f}s{meeting_info}")
                
                # === STALL DETECTION for bidirectional search ===
                # Check if both searches are stuck (no progress in last interval)
                fwd_stalled = abs(forward_progress - getattr(self, '_last_fwd_progress', 0)) < 0.1
                bwd_stalled = abs(backward_progress - getattr(self, '_last_bwd_progress', 0)) < 0.1
                self._last_fwd_progress = forward_progress
                self._last_bwd_progress = backward_progress
                
                if fwd_stalled and bwd_stalled and iterations > 50000:
                    # Both searches stalled - diagnose why
                    print(f"\n[Kinodynamic-Bidir] ⚠️ BOTH SEARCHES STALLED!")
                    
                    # Forward search diagnosis
                    if best_forward_progress_node:
                        fwd_state = best_forward_progress_node.state
                        fwd_lat, fwd_lng = self.elevation_grid.transform.to_latlon(fwd_state.x, fwd_state.y)
                        fwd_elev = self.elevation_grid.get_elevation(fwd_state.x, fwd_state.y)
                        print(f"[Kinodynamic-Bidir] Forward stuck at: ({fwd_state.x:.0f}, {fwd_state.y:.0f})")
                        print(f"[Kinodynamic-Bidir]   Lat/Lng: {fwd_lat:.6f}, {fwd_lng:.6f}")
                        print(f"[Kinodynamic-Bidir]   Elevation: {fwd_elev:.1f}m, Heading: {math.degrees(fwd_state.heading):.1f}°")
                        
                        # Check terrain toward goal
                        dx = goal_x - fwd_state.x
                        dy = goal_y - fwd_state.y
                        dist_to_goal = math.sqrt(dx*dx + dy*dy)
                        for step in [100, 200, 300]:
                            if step < dist_to_goal:
                                sx = fwd_state.x + dx * step / dist_to_goal
                                sy = fwd_state.y + dy * step / dist_to_goal
                                se = self.elevation_grid.get_elevation(sx, sy)
                                slope = abs(se - fwd_elev) / step * 100
                                water = "🌊" if self.constraints.is_water(sx, sy) else ""
                                steep = "⛰️" if slope > self.config.hard_slope_limit_percent else ""
                                print(f"[Kinodynamic-Bidir]   +{step}m toward goal: elev={se:.1f}m, slope={slope:.1f}% {steep}{water}")
                    
                    # Backward search diagnosis
                    if best_backward_progress_node:
                        bwd_state = best_backward_progress_node.state
                        bwd_lat, bwd_lng = self.elevation_grid.transform.to_latlon(bwd_state.x, bwd_state.y)
                        bwd_elev = self.elevation_grid.get_elevation(bwd_state.x, bwd_state.y)
                        print(f"[Kinodynamic-Bidir] Backward stuck at: ({bwd_state.x:.0f}, {bwd_state.y:.0f})")
                        print(f"[Kinodynamic-Bidir]   Lat/Lng: {bwd_lat:.6f}, {bwd_lng:.6f}")
                        print(f"[Kinodynamic-Bidir]   Elevation: {bwd_elev:.1f}m, Heading: {math.degrees(bwd_state.heading):.1f}°")
                        
                        # Check terrain toward start
                        dx = start_x - bwd_state.x
                        dy = start_y - bwd_state.y
                        dist_to_start = math.sqrt(dx*dx + dy*dy)
                        for step in [100, 200, 300]:
                            if step < dist_to_start:
                                sx = bwd_state.x + dx * step / dist_to_start
                                sy = bwd_state.y + dy * step / dist_to_start
                                se = self.elevation_grid.get_elevation(sx, sy)
                                slope = abs(se - bwd_elev) / step * 100
                                water = "🌊" if self.constraints.is_water(sx, sy) else ""
                                steep = "⛰️" if slope > self.config.hard_slope_limit_percent else ""
                                print(f"[Kinodynamic-Bidir]   +{step}m toward start: elev={se:.1f}m, slope={slope:.1f}% {steep}{water}")
                    
                    # Gap analysis - distance between the two stuck points
                    if best_forward_progress_node and best_backward_progress_node:
                        gap_x = best_backward_progress_node.state.x - best_forward_progress_node.state.x
                        gap_y = best_backward_progress_node.state.y - best_forward_progress_node.state.y
                        gap_dist = math.sqrt(gap_x*gap_x + gap_y*gap_y)
                        print(f"[Kinodynamic-Bidir] Gap between stuck points: {gap_dist/1000:.1f}km")
                        print(f"[Kinodynamic-Bidir] 💡 TIP: Add waypoints/bridges in the gap, or adjust route around water\n")
                
                # If we have a meeting and both searches are making no more progress, use it
                if best_meeting_forward_node is not None:
                    # Check if continuing search is unlikely to improve
                    if forward_open and backward_open:
                        fwd_best_f = forward_open[0][0]
                        bwd_best_f = backward_open[0][0]
                        # If best unexplored f-costs exceed our meeting cost, we're done
                        if fwd_best_f + bwd_best_f > best_meeting_cost * 1.1:
                            break
        
        elapsed = time.time() - start_time
        
        # Check if we found a meeting point
        if best_meeting_forward_node is not None and best_meeting_backward_node is not None:
            # Reconstruct path: forward path + reversed backward path
            forward_path = self._reconstruct_path(best_meeting_forward_node)
            backward_path = self._reconstruct_path_reversed(best_meeting_backward_node)
            
            # Combine paths (skip duplicate meeting point)
            combined_path = forward_path + backward_path[1:] if backward_path else forward_path
            
            return PathResult(
                success=True,
                path=combined_path,
                total_cost=best_meeting_cost,
                iterations=iterations,
                nodes_expanded=nodes_expanded,
                elapsed_time=elapsed,
                message=f"Segment found via bidirectional meeting in {iterations} iterations"
            )
        
        # No path found - include partial paths from both directions
        failure_x = start_x
        failure_y = start_y
        best_dist = initial_distance
        partial_path: List[PathSegment] = []
        
        # Use best forward progress node for failure location and partial path
        if best_forward_progress_node is not None:
            failure_x = best_forward_progress_node.state.x
            failure_y = best_forward_progress_node.state.y
            best_dist = best_forward_progress_dist
            # Reconstruct partial path from start to best forward progress
            partial_path = self._reconstruct_path(best_forward_progress_node)
            print(f"[Kinodynamic-Bidir] Forward partial path: {len(partial_path)} segments to ({failure_x:.0f}, {failure_y:.0f})")
        
        # Also include backward partial path (from goal toward start)
        backward_partial_path: List[PathSegment] = []
        if best_backward_progress_node is not None and best_backward_progress_node.parent is not None:
            # Reconstruct backward path - this goes from goal toward where backward search reached
            backward_partial_path = self._reconstruct_path_reversed(best_backward_progress_node)
            print(f"[Kinodynamic-Bidir] Backward partial path: {len(backward_partial_path)} segments from goal")
            
            # Combine: forward path + gap + backward path
            # The backward path starts from goal and goes toward start
            # We append it to show both explored regions
            if partial_path and backward_partial_path:
                # Add a gap marker (empty segment list is fine, frontend will see the discontinuity)
                partial_path.extend(backward_partial_path)
                print(f"[Kinodynamic-Bidir] Combined partial path: {len(partial_path)} total segments")
        
        if iterations >= max_iterations:
            message = f"Bidirectional search: max iterations ({max_iterations}) reached"
        else:
            message = "Bidirectional search: no path exists"
        
        return PathResult(
            success=False,
            path=partial_path,  # Include combined partial paths from both directions!
            total_cost=float('inf'),
            iterations=iterations,
            nodes_expanded=nodes_expanded,
            elapsed_time=elapsed,
            message=message,
            failure_x=failure_x,
            failure_y=failure_y,
            best_distance_remaining=best_dist,
            # Also include backward failure location for visualization
            failure_x_backward=best_backward_progress_node.state.x if best_backward_progress_node else None,
            failure_y_backward=best_backward_progress_node.state.y if best_backward_progress_node else None
        )
    
    def _reconstruct_path_reversed(self, goal_node: PathNode) -> List[PathSegment]:
        """
        Reconstruct path from a backward search node.
        
        Since backward search goes goal→start, we need to:
        1. Collect nodes from the backward goal to backward start
        2. Reverse to get start→goal order
        3. Flip headings (add 180°) since backward search faces opposite direction
        """
        # Collect nodes
        nodes: List[PathNode] = []
        current = goal_node
        while current is not None:
            nodes.append(current)
            current = current.parent
        
        # nodes is now [backward_leaf, ..., backward_root (at goal)]
        # We want [goal, ..., backward_leaf] with flipped headings
        # But since backward started at GOAL, we just reverse to get goal→leaf
        # which is still backward. We need leaf→goal for the forward direction.
        # Actually: nodes[0] = where backward search reached
        #           nodes[-1] = goal (where backward started)
        # For output path, we want: nodes[-1] → nodes[0], which is nodes reversed
        # No wait, we DON'T reverse - the backward path goes from meeting point toward goal
        
        path_segments: List[PathSegment] = []
        
        # Don't reverse - nodes[0] is meeting point, nodes[-1] is goal
        # We want meeting_point → goal, so iterate forward through nodes
        for i, node in enumerate(nodes):
            elev = self.elevation_grid.get_elevation(node.state.x, node.state.y)
            if elev == float('inf'):
                elev = 0.0
            
            # Flip heading by 180° since backward search faces opposite direction
            flipped_heading = (math.degrees(node.state.heading) + 180) % 360
            
            curvature = 0.0
            if node.primitive is not None:
                curvature = node.primitive.curvature
            
            path_segments.append(PathSegment(
                x=node.state.x,
                y=node.state.y,
                heading=flipped_heading,
                elevation=elev,
                curvature=curvature,
                direction_gear=node.state.direction_gear,
                structure_type=node.structure_type,
                is_switchback=node.is_switchback
            ))
        
        # Now reverse so it goes from meeting point toward goal
        path_segments.reverse()
        
        return path_segments
    
    def _find_path_segment(
        self,
        start_x: float, start_y: float,
        goal_x: float, goal_y: float,
        initial_heading: float,
        initial_direction_gear: DirectionGear,
        last_switchback_x: Optional[float],
        last_switchback_y: Optional[float],
        max_iterations: int,
        goal_tolerance: float
    ) -> PathResult:
        """
        Find a path for a single segment with explicit initial state.
        
        This is the internal method used by find_path_through_waypoints.
        Unlike find_path, this takes the full initial state including
        direction_gear and last_switchback position.
        """
        # Use bidirectional search if enabled
        if self.config.bidirectional_search:
            return self._find_path_segment_bidirectional(
                start_x=start_x, start_y=start_y,
                goal_x=goal_x, goal_y=goal_y,
                initial_heading=initial_heading,
                initial_direction_gear=initial_direction_gear,
                last_switchback_x=last_switchback_x,
                last_switchback_y=last_switchback_y,
                max_iterations=max_iterations,
                goal_tolerance=goal_tolerance
            )
        
        start_time = time.time()
        
        # Convert heading to radians for internal State
        heading_rad = math.radians(initial_heading)
        
        # Create start state with full continuity info
        start_state = State(
            x=start_x,
            y=start_y,
            heading=heading_rad,
            direction_gear=initial_direction_gear,
            last_switchback_x=last_switchback_x,
            last_switchback_y=last_switchback_y
        )
        
        # Initialize start node
        h_start = self.heuristic(start_state, goal_x, goal_y)
        start_node = PathNode(
            state=start_state,
            g_cost=0.0,
            f_cost=h_start
        )
        
        # Priority queue
        open_set: List[Tuple[float, int, PathNode]] = []
        counter = 0
        heapq.heappush(open_set, (start_node.f_cost, counter, start_node))
        counter += 1
        
        # Closed set
        closed_set: Set[Tuple[int, int, int, int]] = set()
        best_g: Dict[Tuple[int, int, int, int], float] = {}
        
        iterations = 0
        nodes_expanded = 0
        last_progress_time = start_time
        progress_interval = 10000  # Log every N iterations
        
        # Calculate initial distance for progress tracking
        initial_distance = math.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
        stall_counter = 0  # Track how many intervals without progress
        last_best_distance = initial_distance
        current_heuristic_weight = self.config.heuristic_weight  # Start with normal weight
        
        # Track best progress node for partial path on failure
        best_progress_node: Optional[PathNode] = start_node
        best_progress_dist: float = initial_distance
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Progress logging
            if iterations % progress_interval == 0:
                elapsed = time.time() - start_time
                # Use best_progress_dist (updated during expansion) for accurate tracking
                progress_pct = (1 - best_progress_dist / initial_distance) * 100
                
                # Detect stall
                if best_progress_dist >= last_best_distance - 1:  # No meaningful progress
                    stall_counter += 1
                else:
                    stall_counter = 0
                last_best_distance = best_progress_dist
                
                stall_info = f" (stalled {stall_counter}x)" if stall_counter >= 3 else ""
                weight_info = f" [weight={current_heuristic_weight:.2f}]" if current_heuristic_weight != self.config.heuristic_weight else ""
                print(f"[Kinodynamic] {iterations:,} iterations, {nodes_expanded:,} expanded, "
                      f"{len(open_set):,} queued, {elapsed:.1f}s, "
                      f"best dist: {best_progress_dist:.0f}m ({progress_pct:.1f}%){stall_info}{weight_info}")
                
                # If stalled for too long, perform detailed diagnostics
                if stall_counter == 5 and best_progress_node is not None:
                    self._diagnose_stall(
                        best_progress_node, goal_x, goal_y, initial_distance,
                        best_progress_dist
                    )
                    
                    # === STALL RECOVERY: Reduce heuristic weight ===
                    # When search is stuck, reduce greediness to explore more broadly
                    if current_heuristic_weight > 1.0:
                        new_weight = max(1.0, current_heuristic_weight - 0.25)
                        print(f"[Kinodynamic] 🔧 STALL RECOVERY: Reducing heuristic weight "
                              f"from {current_heuristic_weight:.2f} to {new_weight:.2f}")
                        print(f"[Kinodynamic]    This makes A* less greedy and more exploratory")
                        current_heuristic_weight = new_weight
                        
                        # Recompute f-costs for nodes in open set with new weight
                        # This allows nodes that were "behind" to become competitive
                        new_open_set: List[Tuple[float, int, PathNode]] = []
                        for _, seq, node in open_set:
                            new_h = self.heuristic(node.state, goal_x, goal_y, current_heuristic_weight)
                            new_f = node.g_cost + new_h
                            heapq.heappush(new_open_set, (new_f, seq, node))
                        open_set = new_open_set
                        print(f"[Kinodynamic]    Recomputed f-costs for {len(open_set)} queued nodes")
                        
                        # Reset stall counter after recovery attempt
                        stall_counter = 0
                    else:
                        print(f"[Kinodynamic] ⚠️ Already at minimum weight (1.0), can't reduce further")
                        print(f"[Kinodynamic]    This path may be truly blocked by terrain/water")
                        print(f"[Kinodynamic]    Consider adding waypoints or tunnel/bridge markers")
            
            _, _, current = heapq.heappop(open_set)
            
            # Check if we've reached the goal
            if self.is_goal(current.state, goal_x, goal_y, goal_tolerance):
                path_segments = self._reconstruct_path(current)
                elapsed = time.time() - start_time
                
                return PathResult(
                    success=True,
                    path=path_segments,
                    total_cost=current.g_cost,
                    iterations=iterations,
                    nodes_expanded=nodes_expanded,
                    elapsed_time=elapsed,
                    message=f"Segment found in {iterations} iterations"
                )
            
            # === HYBRID A* ANALYTIC EXPANSION ===
            # Try direct shot-to-goal when close enough
            goal_node = self.try_analytic_expansion(
                current, goal_x, goal_y, self.config.step_distance_m
            )
            if goal_node is not None:
                path_segments = self._reconstruct_path(goal_node)
                elapsed = time.time() - start_time
                
                return PathResult(
                    success=True,
                    path=path_segments,
                    total_cost=goal_node.g_cost,
                    iterations=iterations,
                    nodes_expanded=nodes_expanded,
                    elapsed_time=elapsed,
                    message=f"Segment found via analytic expansion in {iterations} iterations"
                )
            
            state_key = current.state._discretize()
            
            if state_key in closed_set:
                continue
            
            if state_key in best_g and best_g[state_key] < current.g_cost:
                continue
            
            closed_set.add(state_key)
            nodes_expanded += 1
            
            # Track best progress node for partial path on failure
            current_dist = math.sqrt(
                (goal_x - current.state.x)**2 + (goal_y - current.state.y)**2
            )
            if current_dist < best_progress_dist:
                best_progress_dist = current_dist
                best_progress_node = current
            
            neighbors = self.neighbor_generator.get_neighbors(current.state, goal_x, goal_y)
            
            for neighbor in neighbors:
                new_g = current.g_cost + neighbor.cost
                new_state_key = neighbor.state._discretize()
                
                if new_state_key in closed_set:
                    continue
                
                if new_state_key in best_g and best_g[new_state_key] <= new_g:
                    continue
                
                best_g[new_state_key] = new_g
                
                h = self.heuristic(neighbor.state, goal_x, goal_y)
                new_f = new_g + h
                
                new_node = PathNode(
                    state=neighbor.state,
                    g_cost=new_g,
                    f_cost=new_f,
                    parent=current,
                    primitive=neighbor.primitive,
                    structure_type=neighbor.structure_type,
                    is_switchback=neighbor.is_switchback,
                    is_portal=neighbor.is_portal,
                    is_auto_structure=neighbor.is_auto_structure
                )
                
                heapq.heappush(open_set, (new_f, counter, new_node))
                counter += 1
        
        elapsed = time.time() - start_time
        
        # Use best progress node for failure location and partial path
        failure_x = start_x
        failure_y = start_y
        partial_path: List[PathSegment] = []
        
        if best_progress_node is not None:
            failure_x = best_progress_node.state.x
            failure_y = best_progress_node.state.y
            # Reconstruct partial path to best progress point
            partial_path = self._reconstruct_path(best_progress_node)
            print(f"[Kinodynamic] Partial path: {len(partial_path)} segments to ({failure_x:.0f}, {failure_y:.0f})")
        
        if iterations >= max_iterations:
            message = f"Segment terminated: max iterations ({max_iterations}) reached"
        else:
            message = "Segment terminated: no path exists"
        
        return PathResult(
            success=False,
            path=partial_path,  # Include partial path!
            total_cost=float('inf'),
            iterations=iterations,
            nodes_expanded=nodes_expanded,
            elapsed_time=elapsed,
            message=message,
            failure_x=failure_x,
            failure_y=failure_y,
            best_distance_remaining=best_progress_dist
        )
