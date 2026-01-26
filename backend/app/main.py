"""
NIMBY Route Finder - Backend API
Railway route generation with terrain constraints
"""

import os
from pathlib import Path

# Load .env file from backend folder
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import math

from .pathfinder import ConstrainedAStar, PathfindingConfig
from .kinodynamic import (
    KinodynamicConfig, KinodynamicPathfinder, 
    ElevationGrid, ConstraintMasks, PortalRegistry,
    DirectionGear, StructureType
)
from .elevation import ElevationService
from .roads import RoadService

app = FastAPI(
    title="NIMBY Route Finder",
    description="Automatic railway route generation with slope and curvature constraints",
    version="0.1.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services
elevation_service = ElevationService()
road_service = RoadService()


class LatLng(BaseModel):
    lat: float
    lng: float


class WaypointWithHeading(BaseModel):
    """Waypoint with optional heading direction"""
    lat: float
    lng: float
    heading: Optional[float] = None  # Degrees, 0=North, clockwise. None = auto-compute


class TunnelPortal(BaseModel):
    """A pair of points defining tunnel entry and exit"""
    entry: LatLng
    exit: LatLng


class BridgeMarker(BaseModel):
    """A pair of points defining bridge start and end (allows water crossing)"""
    start: LatLng
    end: LatLng


class RouteRequest(BaseModel):
    start_lat: float
    start_lng: float
    end_lat: float
    end_lng: float
    max_slope_percent: float = 3.0  # Target maximum grade in percent
    min_curve_radius_m: float = 500.0  # Minimum curve radius in meters
    downsampling_factor: int = 1  # 1 = native resolution, 2 = half, etc.
    # Advanced options
    hard_slope_limit_percent: float = 8.0  # Absolute maximum grade (never exceed)
    allow_water_crossing: bool = False  # If True, reduces water penalty
    padding_factor: float = 0.3  # Padding around bounding box (0.3 = 30%)
    # Switchback control
    allow_switchbacks: bool = False  # If True, allows 180° turns (switchbacks)
    switchback_penalty: float = 5000.0  # Penalty for switchbacks
    min_switchback_interval: int = 50  # Minimum cells between switchbacks
    # Auto tunnel/bridge detection
    auto_tunnel_bridge: bool = False  # Enable automatic tunnel/bridge detection
    max_jump_distance_m: float = 500.0  # Maximum distance for auto tunnel/bridge
    elevation_tolerance_m: float = 10.0  # Elevation tolerance for auto detection
    # Road parallelism constraints
    road_parallel_enabled: bool = False  # Enable road parallel constraints
    road_parallel_threshold_deg: float = 30.0  # Angle threshold for "nearly parallel"
    road_min_separation_m: float = 10.0  # Minimum separation from road when parallel
    road_max_separation_m: float = 50.0  # Maximum separation to apply parallel constraint
    # Waypoints, tunnels, bridges
    waypoints: List[WaypointWithHeading] = []  # Intermediate points with optional heading
    tunnels: List[TunnelPortal] = []  # Tunnel entry/exit pairs
    bridges: List[BridgeMarker] = []  # Bridge start/end pairs
    # Start/end heading (optional manual override)
    start_heading: Optional[float] = None  # Degrees, 0=North, clockwise. None = auto
    end_heading: Optional[float] = None  # Degrees (for final approach, not yet implemented)
    # Pathfinder selection
    use_kinodynamic: bool = True  # True = new kinodynamic A*, False = legacy grid A*
    max_iterations: int = 500000  # Maximum iterations for pathfinding (kinodynamic only)


class RouteResponse(BaseModel):
    success: bool
    message: str
    route_geojson: Optional[dict] = None
    stats: Optional[dict] = None


def latlon_to_local(lat: float, lng: float, origin_lat: float, origin_lng: float) -> tuple:
    """Convert lat/lng to local meters coordinate system."""
    meters_per_deg_lat = 110540.0
    meters_per_deg_lng = 111320.0 * math.cos(math.radians(origin_lat))
    
    x = (lng - origin_lng) * meters_per_deg_lng
    y = (lat - origin_lat) * meters_per_deg_lat
    return (x, y)


def local_to_latlon(x: float, y: float, origin_lat: float, origin_lng: float) -> tuple:
    """Convert local meters back to lat/lng."""
    meters_per_deg_lat = 110540.0
    meters_per_deg_lng = 111320.0 * math.cos(math.radians(origin_lat))
    
    lat = origin_lat + (y / meters_per_deg_lat)
    lng = origin_lng + (x / meters_per_deg_lng)
    return (lat, lng)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/route", response_model=RouteResponse)
async def generate_route(request: RouteRequest):
    """Generate a railway route between two points with constraints."""
    try:
        # Validate coordinates
        if not (-90 <= request.start_lat <= 90 and -180 <= request.start_lng <= 180):
            raise HTTPException(status_code=400, detail="Invalid start coordinates")
        if not (-90 <= request.end_lat <= 90 and -180 <= request.end_lng <= 180):
            raise HTTPException(status_code=400, detail="Invalid end coordinates")

        # Build list of all points to include in bounding box
        all_points = [
            (request.start_lat, request.start_lng),
            (request.end_lat, request.end_lng)
        ]
        for wp in request.waypoints:
            all_points.append((wp.lat, wp.lng))
        for tunnel in request.tunnels:
            all_points.append((tunnel.entry.lat, tunnel.entry.lng))
            all_points.append((tunnel.exit.lat, tunnel.exit.lng))
        for bridge in request.bridges:
            all_points.append((bridge.start.lat, bridge.start.lng))
            all_points.append((bridge.end.lat, bridge.end.lng))
        
        # Calculate expanded bounding box that includes all points
        all_lats = [p[0] for p in all_points]
        all_lngs = [p[1] for p in all_points]
        
        # Get elevation data for the expanded bounding box
        elevation_grid, bounds, transform = await elevation_service.get_elevation_grid(
            start=(min(all_lats), min(all_lngs)),
            end=(max(all_lats), max(all_lngs)),
            downsampling_factor=request.downsampling_factor,
            padding_factor=request.padding_factor
        )

        if elevation_grid is None:
            return RouteResponse(
                success=False,
                message="Could not fetch elevation data for this area"
            )

        cell_size_m = transform.get('cell_size_m', 30.0)
        min_lat, min_lng, max_lat, max_lng = bounds
        
        # Origin is at southwest corner of bounding box
        origin_lat = min_lat
        origin_lng = min_lng

        # ========================================================================
        # LEGACY GRID A* PATHFINDER
        # ========================================================================
        if not request.use_kinodynamic:
            print(f"[Legacy] Using grid-based A* pathfinder")
            
            # Configure legacy pathfinder
            legacy_config = PathfindingConfig(
                max_slope_percent=request.max_slope_percent,
                min_curve_radius_m=request.min_curve_radius_m,
                cell_size_m=cell_size_m,
                hard_slope_limit_percent=request.hard_slope_limit_percent,
                allow_switchbacks=request.allow_switchbacks,
                switchback_penalty=request.switchback_penalty,
                min_switchback_interval=request.min_switchback_interval,
                auto_tunnel_bridge=request.auto_tunnel_bridge,
                max_jump_distance_m=request.max_jump_distance_m,
                elevation_tolerance_m=request.elevation_tolerance_m,
                road_parallel_enabled=request.road_parallel_enabled,
                road_parallel_threshold_deg=request.road_parallel_threshold_deg,
                road_min_separation_m=request.road_min_separation_m,
                road_max_separation_m=request.road_max_separation_m,
            )
            
            pathfinder = ConstrainedAStar(
                elevation_grid=elevation_grid,
                bounds=bounds,
                transform=transform,
                config=legacy_config
            )
            
            # Add tunnel zones
            for tunnel in request.tunnels:
                pathfinder.add_tunnel_zone(
                    entry=(tunnel.entry.lat, tunnel.entry.lng),
                    exit=(tunnel.exit.lat, tunnel.exit.lng)
                )
            
            # Add bridge zones
            for bridge in request.bridges:
                pathfinder.add_bridge_zone(
                    start=(bridge.start.lat, bridge.start.lng),
                    end=(bridge.end.lat, bridge.end.lng)
                )
            
            # Build waypoints list with start and end
            all_waypoints = [(request.start_lat, request.start_lng)]
            for wp in request.waypoints:
                all_waypoints.append((wp.lat, wp.lng))
            all_waypoints.append((request.end_lat, request.end_lng))
            
            # Find path through waypoints
            full_path = []
            total_stats = {
                "nodes_expanded": 0,
                "max_queue_size": 0,
                "path_length": 0,
                "max_slope_encountered": 0.0,
                "elevation_gain_m": 0.0,
                "segments": len(all_waypoints) - 1
            }
            
            for i in range(len(all_waypoints) - 1):
                start_wp = all_waypoints[i]
                end_wp = all_waypoints[i + 1]
                
                start_grid = pathfinder.latlon_to_grid(start_wp[0], start_wp[1])
                end_grid = pathfinder.latlon_to_grid(end_wp[0], end_wp[1])
                
                print(f"[Legacy] Segment {i+1}/{len(all_waypoints)-1}: {start_grid} -> {end_grid}")
                
                path, stats = pathfinder.find_path(start_grid, end_grid)
                
                if path is None:
                    return RouteResponse(
                        success=False,
                        message=f"No valid route found for segment {i+1}: {stats.get('error', 'Unknown error')}",
                        stats=total_stats
                    )
                
                # Append path (skip first point for subsequent segments)
                if i == 0:
                    full_path.extend(path)
                else:
                    full_path.extend(path[1:])
                
                # Accumulate stats
                total_stats["nodes_expanded"] += stats.get("nodes_expanded", 0)
                total_stats["max_queue_size"] = max(total_stats["max_queue_size"], stats.get("max_queue_size", 0))
                total_stats["max_slope_encountered"] = max(
                    total_stats["max_slope_encountered"], 
                    stats.get("max_slope_encountered", 0)
                )
                total_stats["elevation_gain_m"] += stats.get("elevation_gain_m", 0)
            
            total_stats["path_length"] = len(full_path)
            
            # Convert grid path to coordinates
            coordinates = []
            elevations = []
            
            for row, col in full_path:
                lat, lng = pathfinder.grid_to_latlon(row, col)
                coordinates.append([lng, lat])
                elevations.append(float(elevation_grid[row, col]))
            
            # Calculate total distance
            total_distance = 0.0
            for i in range(1, len(coordinates)):
                dx = coordinates[i][0] - coordinates[i-1][0]
                dy = coordinates[i][1] - coordinates[i-1][1]
                # Approximate distance in meters
                dx_m = dx * 111320.0 * math.cos(math.radians(coordinates[i][1]))
                dy_m = dy * 110540.0
                total_distance += math.sqrt(dx_m*dx_m + dy_m*dy_m)
            
            total_stats["total_distance_m"] = total_distance
            
            # Build GeoJSON
            geojson = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                },
                "properties": {
                    "elevations": elevations,
                    "pathfinder": "legacy"
                }
            }
            
            if total_stats["max_slope_encountered"] > request.max_slope_percent:
                total_stats["warning"] = f"Path uses {total_stats['max_slope_encountered']:.1f}% grade (target: {request.max_slope_percent}%)"
            
            return RouteResponse(
                success=True,
                message=f"Route found with {len(full_path)} points",
                route_geojson=geojson,
                stats=total_stats
            )

        # ========================================================================
        # KINODYNAMIC A* PATHFINDER
        # ========================================================================
        print(f"[Kinodynamic] Using kinodynamic A* pathfinder")
        
        # Calculate bounds in local meters
        _, bounds_max_y = latlon_to_local(max_lat, min_lng, origin_lat, origin_lng)
        bounds_max_x, _ = latlon_to_local(min_lat, max_lng, origin_lat, origin_lng)
        
        # Calculate approximate path distance for adaptive step sizing
        start_local = latlon_to_local(request.start_lat, request.start_lng, origin_lat, origin_lng)
        end_local = latlon_to_local(request.end_lat, request.end_lng, origin_lat, origin_lng)
        approx_distance = math.sqrt(
            (end_local[0] - start_local[0])**2 + 
            (end_local[1] - start_local[1])**2
        )
        
        # Adaptive step size: increase for very long paths to speed up search
        # Target ~200-500 steps for any path length
        target_steps = 300
        adaptive_step = max(cell_size_m, approx_distance / target_steps)
        # But cap at a reasonable maximum (don't exceed min curve radius)
        adaptive_step = min(adaptive_step, request.min_curve_radius_m * 0.5)
        
        # Use fewer curvature samples for speed (5 instead of 7)
        num_curvatures = 5
        
        print(f"[Kinodynamic] Local bounds: (0,0) to ({bounds_max_x:.1f}, {bounds_max_y:.1f}) meters")
        print(f"[Kinodynamic] Path distance: {approx_distance/1000:.1f}km, Step: {adaptive_step:.1f}m, Curvatures: {num_curvatures}")
        
        config = KinodynamicConfig(
            step_distance_m=adaptive_step,
            bounds_min_x=0.0,
            bounds_min_y=0.0,
            bounds_max_x=bounds_max_x,
            bounds_max_y=bounds_max_y,
            padding_factor=request.padding_factor,
            # Slope constraints
            max_slope_percent=request.max_slope_percent,
            hard_slope_limit_percent=request.hard_slope_limit_percent,
            slope_penalty_multiplier=20.0,
            # Curvature
            min_curve_radius_m=request.min_curve_radius_m,
            num_curvature_samples=num_curvatures,
            # Costs
            distance_weight=1.0,
            elevation_gain_weight=1.5,
            curvature_weight=0.3,
            # Switchbacks
            allow_switchbacks=request.allow_switchbacks,
            switchback_penalty=request.switchback_penalty,
            min_switchback_distance_m=request.min_switchback_interval * adaptive_step,
            # Water
            water_penalty=100.0 if request.allow_water_crossing else 10000.0,
            # Structures
            tunnel_cost_per_m=50.0,
            bridge_cost_per_m=100.0,
            # Auto detection
            auto_tunnel_bridge=request.auto_tunnel_bridge,
            max_jump_distance_m=request.max_jump_distance_m,
            elevation_tolerance_m=request.elevation_tolerance_m,
            # Road parallelism
            road_parallel_enabled=request.road_parallel_enabled,
            road_parallel_threshold_deg=request.road_parallel_threshold_deg,
            road_min_separation_m=request.road_min_separation_m,
            road_max_separation_m=request.road_max_separation_m,
            road_parallel_penalty=500.0,
        )

        # Create elevation grid wrapper for kinodynamic pathfinder
        kinodynamic_elevation = ElevationGrid(
            data=elevation_grid,
            bounds=bounds
        )
        
        # Create water mask from elevation data (low elevation areas)
        water_threshold = 1.0  # meters
        water_mask = elevation_grid < water_threshold
        
        # Create constraint masks
        constraints = ConstraintMasks(
            water_mask=water_mask,
            tunnel_mask=np.zeros_like(elevation_grid, dtype=bool),
            bridge_mask=np.zeros_like(elevation_grid, dtype=bool),
            elevation_grid=kinodynamic_elevation
        )
        
        # Create portal registry for tunnels and bridges
        portal_registry = PortalRegistry()
        
        # Register tunnel portals
        for tunnel in request.tunnels:
            entry_x, entry_y = latlon_to_local(
                tunnel.entry.lat, tunnel.entry.lng, origin_lat, origin_lng
            )
            exit_x, exit_y = latlon_to_local(
                tunnel.exit.lat, tunnel.exit.lng, origin_lat, origin_lng
            )
            portal_registry.add_tunnel(
                entry_x=entry_x, entry_y=entry_y,
                exit_x=exit_x, exit_y=exit_y,
                entry_tolerance=50.0
            )
            print(f"[Kinodynamic] Added tunnel: ({entry_x:.1f},{entry_y:.1f}) -> ({exit_x:.1f},{exit_y:.1f})")
        
        # Register bridge portals
        for bridge in request.bridges:
            start_x, start_y = latlon_to_local(
                bridge.start.lat, bridge.start.lng, origin_lat, origin_lng
            )
            end_x, end_y = latlon_to_local(
                bridge.end.lat, bridge.end.lng, origin_lat, origin_lng
            )
            portal_registry.add_bridge(
                entry_x=start_x, entry_y=start_y,
                exit_x=end_x, exit_y=end_y,
                entry_tolerance=50.0
            )
            print(f"[Kinodynamic] Added bridge: ({start_x:.1f},{start_y:.1f}) -> ({end_x:.1f},{end_y:.1f})")

        # Create kinodynamic pathfinder
        pathfinder = KinodynamicPathfinder(
            config=config,
            elevation_grid=kinodynamic_elevation,
            constraints=constraints,
            portal_registry=portal_registry
        )
        
        # Build waypoint list in local coordinates with optional headings
        # Format: List of (x, y, optional_heading) tuples
        waypoints_local = []
        headings_local = []  # Parallel list of headings (None for auto-compute)
        
        # Start point
        start_x, start_y = latlon_to_local(
            request.start_lat, request.start_lng, origin_lat, origin_lng
        )
        waypoints_local.append((start_x, start_y))
        headings_local.append(request.start_heading)  # May be None
        
        # Intermediate waypoints
        for wp in request.waypoints:
            wp_x, wp_y = latlon_to_local(wp.lat, wp.lng, origin_lat, origin_lng)
            waypoints_local.append((wp_x, wp_y))
            headings_local.append(wp.heading)  # May be None
        
        # End point
        end_x, end_y = latlon_to_local(
            request.end_lat, request.end_lng, origin_lat, origin_lng
        )
        waypoints_local.append((end_x, end_y))
        headings_local.append(request.end_heading)  # End heading (for future use)
        
        # Calculate path distance for iteration limit
        total_path_distance = 0.0
        for i in range(1, len(waypoints_local)):
            dx = waypoints_local[i][0] - waypoints_local[i-1][0]
            dy = waypoints_local[i][1] - waypoints_local[i-1][1]
            total_path_distance += math.sqrt(dx*dx + dy*dy)
        
        # Use user-provided max_iterations or calculate adaptive limit
        if request.max_iterations > 0:
            max_iters = request.max_iterations
        else:
            # Set reasonable iteration limit based on path length
            min_steps = total_path_distance / cell_size_m
            max_iters = max(100000, int(min_steps * 100))
            max_iters = min(max_iters, 2000000)
        
        print(f"[Kinodynamic] Finding path through {len(waypoints_local)} waypoints")
        print(f"[Kinodynamic] Path distance: {total_path_distance/1000:.1f}km, max iterations: {max_iters:,}")
        for i, (wx, wy) in enumerate(waypoints_local):
            heading_info = f", heading={headings_local[i]:.1f}°" if headings_local[i] is not None else ""
            print(f"[Kinodynamic] Waypoint {i}: ({wx:.1f}, {wy:.1f}){heading_info}")
        
        # Find path through all waypoints
        result = pathfinder.find_path_through_waypoints(
            waypoints=waypoints_local,
            waypoint_headings=headings_local,  # Optional headings per waypoint
            max_iterations_per_segment=max_iters,
            goal_tolerance=cell_size_m * 1.5  # Slightly larger than one step
        )
        
        print(f"[Kinodynamic] Result: success={result.success}, {result.iterations} iterations, {len(result.path)} segments")
        print(f"[Kinodynamic] {result.message}")
        
        # Helper function to build GeoJSON from path
        def build_geojson_and_stats(path_segments, is_partial=False):
            coordinates = []
            elevations = []
            curvatures = []
            structures = []
            
            for seg in path_segments:
                lat, lng = local_to_latlon(seg.x, seg.y, origin_lat, origin_lng)
                coordinates.append([lng, lat])
                elevations.append(seg.elevation)
                curvatures.append(seg.curvature)
                if seg.structure_type and seg.structure_type != StructureType.NORMAL:
                    structures.append({
                        "type": seg.structure_type.value,
                        "position": [lng, lat]
                    })
            
            total_distance = 0.0
            elevation_gain = 0.0
            max_slope = 0.0
            
            for i in range(1, len(path_segments)):
                prev = path_segments[i-1]
                curr = path_segments[i]
                
                dx = curr.x - prev.x
                dy = curr.y - prev.y
                dist = math.sqrt(dx*dx + dy*dy)
                total_distance += dist
                
                elev_diff = curr.elevation - prev.elevation
                if elev_diff > 0:
                    elevation_gain += elev_diff
                
                if dist > 0:
                    slope = abs(elev_diff / dist) * 100
                    max_slope = max(max_slope, slope)
            
            geojson = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                },
                "properties": {
                    "elevations": elevations,
                    "curvatures": curvatures,
                    "structures": structures,
                    "is_partial": is_partial
                }
            }
            
            stats = {
                "path_length": len(path_segments),
                "total_distance_m": total_distance,
                "elevation_gain_m": elevation_gain,
                "max_slope_percent": max_slope,
            }
            
            return geojson, stats
        
        if not result.success:
            # Build partial path and failure info for visualization
            stats = {
                "iterations": result.iterations,
                "nodes_expanded": result.nodes_expanded,
                "elapsed_time": result.elapsed_time,
                "error": result.message,
            }
            
            # Add failure location if available
            failure_lat, failure_lng = None, None
            if result.failure_x is not None and result.failure_y is not None:
                failure_lat, failure_lng = local_to_latlon(
                    result.failure_x, result.failure_y, origin_lat, origin_lng
                )
                stats["failure_location"] = [failure_lng, failure_lat]
                stats["failure_segment"] = result.failure_segment
                stats["best_distance_remaining"] = result.best_distance_remaining
                print(f"[Kinodynamic] Failure location: ({failure_lat:.6f}, {failure_lng:.6f})")
            
            # Build GeoJSON from partial path if any, or create minimal geojson for failure point
            route_geojson = None
            if result.path:
                geojson, path_stats = build_geojson_and_stats(result.path, is_partial=True)
                # Add failure marker to the path
                if failure_lat is not None:
                    geojson["properties"]["failure_point"] = [failure_lng, failure_lat]
                route_geojson = geojson
                stats.update(path_stats)
            elif failure_lat is not None:
                # No path but we have failure location - create minimal geojson
                # Include start point and failure point so we can visualize where it got stuck
                start_lat, start_lng = local_to_latlon(
                    waypoints_local[0][0], waypoints_local[0][1], origin_lat, origin_lng
                )
                route_geojson = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [start_lng, start_lat],
                            [failure_lng, failure_lat]
                        ]
                    },
                    "properties": {
                        "elevations": [],
                        "is_partial": True,
                        "failure_point": [failure_lng, failure_lat],
                        "is_failure_line": True  # Flag that this is just start->failure, not actual path
                    }
                }
                stats["path_length"] = 0
                stats["total_distance_m"] = 0
            
            return RouteResponse(
                success=False,
                message=f"No valid route found: {result.message}",
                route_geojson=route_geojson,
                stats=stats
            )
        
        # Convert path segments to GeoJSON
        geojson, stats = build_geojson_and_stats(result.path)
        
        # Add additional properties
        geojson["properties"]["tunnels"] = [
            {"entry": [t.entry.lng, t.entry.lat], "exit": [t.exit.lng, t.exit.lat]} 
            for t in request.tunnels
        ]
        geojson["properties"]["bridges"] = [
            {"start": [b.start.lng, b.start.lat], "end": [b.end.lng, b.end.lat]} 
            for b in request.bridges
        ]
        geojson["properties"]["waypoints"] = [
            [wp.lng, wp.lat] for wp in request.waypoints
        ]
        
        # Add search stats
        stats["iterations"] = result.iterations
        stats["nodes_expanded"] = result.nodes_expanded
        stats["elapsed_time_s"] = result.elapsed_time
        stats["total_cost"] = result.total_cost
        stats["segments"] = len(waypoints_local) - 1
        
        if stats["max_slope_percent"] > request.max_slope_percent:
            stats["warning"] = f"Best path has {stats['max_slope_percent']:.1f}% max grade (target: {request.max_slope_percent}%)"
            print(f"[Kinodynamic] Warning: {stats['warning']}")

        return RouteResponse(
            success=True,
            message=f"Route found with {len(result.path)} waypoints in {result.elapsed_time:.2f}s",
            route_geojson=geojson,
            stats=stats
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/elevation")
async def get_elevation(lat: float, lng: float):
    """Get elevation at a single point."""
    elevation = await elevation_service.get_elevation_at_point(lat, lng)
    return {"lat": lat, "lng": lng, "elevation_m": elevation}
