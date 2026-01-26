"""Test the A* pathfinding in kinodynamic.py."""
import sys
sys.path.insert(0, r'c:\Users\HP\Documents\KERJA\Side Projects\nimby-route-finder\backend')

import numpy as np
import math
from app.kinodynamic import (
    State, DirectionGear, KinodynamicConfig, 
    MotionPrimitiveSet, ElevationGrid, ConstraintMasks, 
    PortalRegistry, KinodynamicPathfinder
)

# Create config
cfg = KinodynamicConfig(
    step_distance_m=30.0,
    min_curve_radius_m=200.0,
    num_curvature_samples=5,
    allow_switchbacks=True,
    min_switchback_distance_m=100.0,
    bounds_min_x=0, bounds_min_y=0,
    bounds_max_x=1000, bounds_max_y=1000
)

# Create elevation grid (flat terrain at 100m)
elev_data = np.ones((100, 100), dtype=np.float32) * 100
bounds = (-7.0, 107.0, -6.9, 107.1)  # Dummy bounds for geo transform
elev_grid = ElevationGrid(data=elev_data, bounds=bounds)

# Create constraint masks (no water)
constraints = ConstraintMasks(
    water_mask=np.zeros((100, 100), dtype=bool),
    tunnel_mask=np.zeros((100, 100), dtype=bool),
    bridge_mask=np.zeros((100, 100), dtype=bool),
    elevation_grid=elev_grid
)

# Create portal registry
portals = PortalRegistry()

print("=== Test: A* Pathfinding ===")

# Create pathfinder
pathfinder = KinodynamicPathfinder(
    config=cfg,
    elevation_grid=elev_grid,
    constraints=constraints,
    portal_registry=portals
)

# Test 1: Simple straight path going North
print("\nTest 1: Straight path (500, 100) -> (500, 900)")
result = pathfinder.find_path(
    start_x=500, start_y=100,
    goal_x=500, goal_y=900,
    max_iterations=10000,
    goal_tolerance=30.0
)

print(f"  Success: {result.success}")
print(f"  Iterations: {result.iterations}")
print(f"  Nodes expanded: {result.nodes_expanded}")
print(f"  Path length: {len(result.path)} segments")
print(f"  Time: {result.elapsed_time:.3f}s")
print(f"  Message: {result.message}")

if result.path:
    print(f"  Start: ({result.path[0].x:.1f}, {result.path[0].y:.1f})")
    print(f"  End: ({result.path[-1].x:.1f}, {result.path[-1].y:.1f})")
    switchbacks = [s for s in result.path if s.is_switchback]
    print(f"  Switchbacks: {len(switchbacks)}")

# Test 2: Diagonal path (requires turns)
print("\nTest 2: Diagonal path (100, 100) -> (800, 800)")
result2 = pathfinder.find_path(
    start_x=100, start_y=100,
    goal_x=800, goal_y=800,
    max_iterations=10000,
    goal_tolerance=30.0
)

print(f"  Success: {result2.success}")
print(f"  Iterations: {result2.iterations}")
print(f"  Nodes expanded: {result2.nodes_expanded}")
print(f"  Path length: {len(result2.path)} segments")
print(f"  Time: {result2.elapsed_time:.3f}s")

if result2.path:
    print(f"  First 5 points:")
    for i, seg in enumerate(result2.path[:5]):
        print(f"    [{i}] ({seg.x:.1f}, {seg.y:.1f}) heading={seg.heading:.1f}° curv={seg.curvature:.4f}")

# Test 3: Short path (within single step)
print("\nTest 3: Short path (500, 500) -> (520, 520)")
result3 = pathfinder.find_path(
    start_x=500, start_y=500,
    goal_x=520, goal_y=520,
    max_iterations=1000,
    goal_tolerance=30.0
)

print(f"  Success: {result3.success}")
print(f"  Iterations: {result3.iterations}")
print(f"  Path length: {len(result3.path)} segments")

# Test 4: Multiple waypoints
print("\nTest 4: Waypoint path (100,100) -> (500,300) -> (800,700)")
waypoints = [
    (100, 100),   # Start
    (500, 300),   # Waypoint 1
    (800, 700)    # End
]

result4 = pathfinder.find_path_through_waypoints(
    waypoints=waypoints,
    max_iterations_per_segment=10000,
    goal_tolerance=30.0
)

print(f"  Success: {result4.success}")
print(f"  Total iterations: {result4.iterations}")
print(f"  Nodes expanded: {result4.nodes_expanded}")
print(f"  Path length: {len(result4.path)} segments")
print(f"  Total cost: {result4.total_cost:.1f}")
print(f"  Time: {result4.elapsed_time:.3f}s")
print(f"  Message: {result4.message}")

if result4.path:
    # Check continuity at waypoints
    print(f"  Start: ({result4.path[0].x:.1f}, {result4.path[0].y:.1f})")
    print(f"  End: ({result4.path[-1].x:.1f}, {result4.path[-1].y:.1f})")
    
    # Find segments near waypoint 1 (500, 300)
    wp1_nearby = [(i, s) for i, s in enumerate(result4.path) 
                  if abs(s.x - 500) < 50 and abs(s.y - 300) < 50]
    if wp1_nearby:
        idx, seg = wp1_nearby[0]
        print(f"  Near WP1 (segment {idx}): ({seg.x:.1f}, {seg.y:.1f}) heading={seg.heading:.1f}°")
        if idx + 1 < len(result4.path):
            next_seg = result4.path[idx + 1]
            print(f"  Next segment: ({next_seg.x:.1f}, {next_seg.y:.1f}) heading={next_seg.heading:.1f}°")

# Test 5: Three waypoints with sharp turn
print("\nTest 5: Sharp turn path (100,500) -> (500,500) -> (500,100)")
waypoints5 = [
    (100, 500),   # Start (going East)
    (500, 500),   # Turn point
    (500, 100)    # End (going South)
]

result5 = pathfinder.find_path_through_waypoints(
    waypoints=waypoints5,
    max_iterations_per_segment=10000,
    goal_tolerance=30.0
)

print(f"  Success: {result5.success}")
print(f"  Total iterations: {result5.iterations}")
print(f"  Path length: {len(result5.path)} segments")

if result5.path:
    print(f"  Start heading: {result5.path[0].heading:.1f}°")
    print(f"  End heading: {result5.path[-1].heading:.1f}°")

print("\n=== All A* tests complete! ===")
