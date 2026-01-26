"""Test script for kinodynamic neighbor generation."""

from app.kinodynamic import *
import math
import numpy as np

print("=== Test: Motion Primitives ===")

cfg = KinodynamicConfig(
    step_distance_m=30.0,
    min_curve_radius_m=200.0,
    num_curvature_samples=5,
    allow_switchbacks=True,
    min_switchback_distance_m=100.0,
    bounds_min_x=0, bounds_min_y=0,
    bounds_max_x=1000, bounds_max_y=1000
)

# Test primitives
state = State(x=500, y=500, heading=0)
print(f"Start: ({state.x}, {state.y}), heading={math.degrees(state.heading):.1f}° (North)")

curvatures = cfg.get_curvatures()
for kappa in curvatures:
    prim = MotionPrimitive(curvature=kappa, arc_length=30.0)
    result = prim.apply(state, cfg)
    if result:
        dx = result.x - state.x
        dy = result.y - state.y
        turn = "Left" if kappa < 0 else ("Right" if kappa > 0 else "Straight")
        print(f"  {turn:8s} (κ={kappa:+.4f}): -> ({result.x:.1f}, {result.y:.1f}), Δ=({dx:+.1f}, {dy:+.1f}), heading={math.degrees(result.heading):.1f}°")

print("\n=== Test: Neighbor Generator ===")

# Create elevation grid (flat terrain at 100m)
elev_data = np.ones((100, 100), dtype=np.float32) * 100
bounds = (-7.0, 107.0, -6.9, 107.1)
elev_grid = ElevationGrid(data=elev_data, bounds=bounds)

# Create constraint masks
constraints = ConstraintMasks(
    water_mask=np.zeros((100, 100), dtype=bool),
    tunnel_mask=np.zeros((100, 100), dtype=bool),
    bridge_mask=np.zeros((100, 100), dtype=bool),
    elevation_grid=elev_grid
)

# Create portal registry
portals = PortalRegistry()

# Create neighbor generator
gen = NeighborGenerator(cfg, elev_grid, constraints, portals)

# Test standard moves
state = State(x=500, y=500, heading=0)
neighbors = gen.get_neighbors(state)
print(f"From (500, 500) heading North:")
print(f"  Total neighbors: {len(neighbors)}")

standard = [n for n in neighbors if not n.is_switchback and not n.is_portal]
switchbacks = [n for n in neighbors if n.is_switchback]
print(f"  Standard moves: {len(standard)}")
print(f"  Switchbacks: {len(switchbacks)}")

for n in standard:
    print(f"    -> ({n.state.x:.1f}, {n.state.y:.1f}), heading={math.degrees(n.state.heading):.1f}°, cost={n.cost:.1f}")

print("\n=== Test: Switchback ===")
# Fresh state (no previous switchback) - should allow switchback
state1 = State(x=500, y=500, heading=0)
neighbors1 = gen.get_neighbors(state1)
sb1 = [n for n in neighbors1 if n.is_switchback]
print(f"Fresh state: switchback available = {len(sb1) > 0}")
if sb1:
    n = sb1[0]
    print(f"  -> gear={n.state.direction_gear.name}, heading={math.degrees(n.state.heading):.1f}°, cost={n.cost:.1f}")

# State with recent switchback (< min_distance)
state2 = State(x=500, y=550, heading=0, last_switchback_x=500, last_switchback_y=500)
neighbors2 = gen.get_neighbors(state2)
sb2 = [n for n in neighbors2 if n.is_switchback]
print(f"50m from last switchback: switchback available = {len(sb2) > 0}")

# State with old switchback (> min_distance)
state3 = State(x=500, y=700, heading=0, last_switchback_x=500, last_switchback_y=500)
neighbors3 = gen.get_neighbors(state3)
sb3 = [n for n in neighbors3 if n.is_switchback]
print(f"200m from last switchback: switchback available = {len(sb3) > 0}")

print("\n=== Test: Portal ===")
portals.add_tunnel(entry_x=500, entry_y=500, exit_x=700, exit_y=500, entry_tolerance=50.0)
gen2 = NeighborGenerator(cfg, elev_grid, constraints, portals)

state_at_portal = State(x=510, y=510, heading=0)
neighbors_portal = gen2.get_neighbors(state_at_portal)
portal_moves = [n for n in neighbors_portal if n.is_portal]
print(f"Near portal entry (510, 510):")
print(f"  Portal moves available: {len(portal_moves)}")
if portal_moves:
    n = portal_moves[0]
    print(f"  -> exit=({n.state.x:.1f}, {n.state.y:.1f}), type={n.structure_type.name}, cost={n.cost:.1f}")

state_far = State(x=100, y=100, heading=0)
neighbors_far = gen2.get_neighbors(state_far)
portal_moves_far = [n for n in neighbors_far if n.is_portal]
print(f"Far from portal (100, 100): Portal moves = {len(portal_moves_far)}")

print("\n=== All tests complete! ===")
heading = state_with_dist.heading
gear = state_with_dist.direction_gear
print(f"\n  Before switchback logic:")
print(f"    heading = {heading}")
print(f"    gear = {gear}")

# Switchback flips heading
new_heading = (heading + math.pi) % (2 * math.pi)
new_gear = DirectionGear.REVERSE if gear == DirectionGear.FORWARD else DirectionGear.FORWARD
print(f"  After switchback logic:")
print(f"    new_heading = {new_heading} ({math.degrees(new_heading)}deg)")
print(f"    new_gear = {new_gear}")

direction = float(new_gear)
dx = prim.arc_length * math.sin(new_heading) * direction
dy = prim.arc_length * math.cos(new_heading) * direction
print(f"  Movement:")
print(f"    direction = {direction}")
print(f"    dx = {prim.arc_length} * sin({new_heading}) * {direction} = {dx}")
print(f"    dy = {prim.arc_length} * cos({new_heading}) * {direction} = {dy}")

new_x = state_with_dist.x + dx
new_y = state_with_dist.y + dy
print(f"  New position: ({new_x}, {new_y})")
print(f"  In bounds? {cfg2.is_in_bounds(new_x, new_y)}")

result = prim.apply(state_with_dist, cfg2)
print(f"\n  Apply result: {result}")
