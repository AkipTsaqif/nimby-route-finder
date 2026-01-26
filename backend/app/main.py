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

from .pathfinder import ConstrainedAStar, PathfindingConfig
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
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
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
    waypoints: List[LatLng] = []  # Intermediate points the route must pass through
    tunnels: List[TunnelPortal] = []  # Tunnel entry/exit pairs
    bridges: List[BridgeMarker] = []  # Bridge start/end pairs


class RouteResponse(BaseModel):
    success: bool
    message: str
    route_geojson: Optional[dict] = None
    stats: Optional[dict] = None


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

        # Configure pathfinder
        config = PathfindingConfig(
            max_slope_percent=request.max_slope_percent,
            min_curve_radius_m=request.min_curve_radius_m,
            cell_size_m=transform.get('cell_size_m', 30.0),
            hard_slope_limit_percent=request.hard_slope_limit_percent,
            water_penalty=100.0 if request.allow_water_crossing else 10000.0,
            # Switchback control
            allow_switchbacks=request.allow_switchbacks,
            switchback_penalty=request.switchback_penalty,
            min_switchback_interval=request.min_switchback_interval,
            # Auto tunnel/bridge detection
            auto_tunnel_bridge=request.auto_tunnel_bridge,
            max_jump_distance_m=request.max_jump_distance_m,
            elevation_tolerance_m=request.elevation_tolerance_m,
            # Road parallelism constraints
            road_parallel_enabled=request.road_parallel_enabled,
            road_parallel_threshold_deg=request.road_parallel_threshold_deg,
            road_min_separation_m=request.road_min_separation_m,
            road_max_separation_m=request.road_max_separation_m,
        )

        # Fetch road data if road parallel constraints are enabled
        road_mask = None
        road_direction = None
        road_distance = None
        
        if request.road_parallel_enabled:
            min_lat, min_lng, max_lat, max_lng = bounds
            road_segments = await road_service.get_roads_in_bounds(
                min_lat, min_lng, max_lat, max_lng
            )
            
            if road_segments:
                road_mask, road_direction, road_distance = road_service.rasterize_roads(
                    road_segments,
                    bounds,
                    elevation_grid.shape[0],
                    elevation_grid.shape[1],
                    transform.get('cell_size_m', 30.0)
                )

        # Create pathfinder
        pathfinder = ConstrainedAStar(
            elevation_grid, bounds, transform, config,
            road_mask=road_mask,
            road_direction=road_direction,
            road_distance=road_distance
        )
        
        # Register tunnel zones (areas where slope constraints are relaxed)
        for tunnel in request.tunnels:
            pathfinder.add_tunnel_zone(
                entry=(tunnel.entry.lat, tunnel.entry.lng),
                exit=(tunnel.exit.lat, tunnel.exit.lng)
            )
        
        # Register bridge zones (areas where water crossing is allowed)
        for bridge in request.bridges:
            pathfinder.add_bridge_zone(
                start=(bridge.start.lat, bridge.start.lng),
                end=(bridge.end.lat, bridge.end.lng)
            )
        
        # Build route through all waypoints
        route_segments = []
        segment_points = [(request.start_lat, request.start_lng)]
        for wp in request.waypoints:
            segment_points.append((wp.lat, wp.lng))
        segment_points.append((request.end_lat, request.end_lng))
        
        full_path = []
        total_stats = {
            "nodes_expanded": 0,
            "max_queue_size": 0,
            "path_length": 0,
            "total_distance_m": 0,
            "max_slope_encountered": 0,
            "elevation_gain_m": 0,
            "water_crossings": 0,
            "segments": len(segment_points) - 1,
        }
        
        # Import heading helper
        from .pathfinder import heading_from_points
        
        # Find path for each segment with direction constraints
        last_heading = None  # Track heading from previous segment
        
        for i in range(len(segment_points) - 1):
            start_pt = segment_points[i]
            end_pt = segment_points[i + 1]
            
            start_grid = pathfinder.latlon_to_grid(start_pt[0], start_pt[1])
            end_grid = pathfinder.latlon_to_grid(end_pt[0], end_pt[1])
            
            # Calculate direction constraints
            # Start heading: use last_heading from previous segment (if any) for continuity
            start_heading = last_heading
            
            # Goal heading: calculate direction toward next waypoint (if exists)
            goal_heading = None
            if i < len(segment_points) - 2:
                # There's another segment after this one
                next_pt = segment_points[i + 2]
                # Calculate heading from current goal to next waypoint
                goal_heading = heading_from_points(end_pt, next_pt)
            
            print(f"[Route] Finding segment {i+1}/{len(segment_points)-1}: {start_pt} -> {end_pt}")
            if start_heading:
                print(f"[Route] Start heading constraint: {start_heading.name}")
            if goal_heading:
                print(f"[Route] Goal heading constraint: {goal_heading.name}")
            
            path, stats = pathfinder.find_path(
                start_grid, end_grid,
                start_heading=start_heading,
                goal_heading=goal_heading
            )
            
            if path is None:
                # Retry without heading constraints if constrained search failed
                print(f"[Route] Segment {i+1} failed with heading constraints, retrying without...")
                path, stats = pathfinder.find_path(start_grid, end_grid)
                
            if path is None:
                return RouteResponse(
                    success=False,
                    message=f"No valid route found for segment {i+1} (waypoint {i} to {i+1}). Try adding tunnels or bridges, or relaxing constraints.",
                    stats=stats
                )
            
            # Calculate exit heading from this segment for next segment
            if len(path) >= 2:
                # Get direction of last step in path
                last_row, last_col = path[-1]
                prev_row, prev_col = path[-2]
                d_row = last_row - prev_row
                d_col = last_col - prev_col
                # Normalize to unit step
                if d_row != 0:
                    d_row = d_row // abs(d_row)
                if d_col != 0:
                    d_col = d_col // abs(d_col)
                from .pathfinder import heading_from_delta
                last_heading = heading_from_delta(d_row, d_col)
            
            # Append path (skip first point if not first segment to avoid duplicates)
            if i > 0 and full_path:
                path = path[1:]
            full_path.extend(path)
            
            # Aggregate stats
            total_stats["nodes_expanded"] += stats.get("nodes_expanded", 0)
            total_stats["max_queue_size"] = max(total_stats["max_queue_size"], stats.get("max_queue_size", 0))
            total_stats["path_length"] += stats.get("path_length", 0)
            total_stats["total_distance_m"] += stats.get("total_distance_m", 0)
            total_stats["max_slope_encountered"] = max(
                total_stats["max_slope_encountered"], 
                stats.get("max_slope_encountered", 0)
            )
            total_stats["elevation_gain_m"] += stats.get("elevation_gain_m", 0)
            total_stats["water_crossings"] += stats.get("water_crossings", 0)
            
            if "warning" in stats:
                total_stats["warning"] = stats["warning"]

        # Apply curve smoothing post-processing to remove zig-zags
        # This now returns (smoothed_float_path, original_nodes)
        from .pathfinder import smooth_curves_post_process
        original_length = len(full_path)
        smoothed_path, original_nodes = smooth_curves_post_process(
            full_path,
            elevation_grid,
            config,
            config.min_curve_radius_m
        )
        if len(smoothed_path) != original_length:
            print(f"[Route] Curve smoothing: {original_length} -> {len(smoothed_path)} points")
            total_stats["path_length"] = len(smoothed_path)

        # Convert to GeoJSON with smoothed coordinates and original node shadows
        geojson = pathfinder.path_to_geojson(
            full_path,  # Original grid path for tunnel/bridge detection
            smooth=False,  # Don't apply Douglas-Peucker, we have spline-smoothed
            smoothed_path=smoothed_path,  # Pre-smoothed float coordinates
            original_nodes=original_nodes  # Original grid nodes for shadow visualization
        )
        
        # Add tunnel and bridge info to GeoJSON properties
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

        return RouteResponse(
            success=True,
            message=f"Route found with {len(smoothed_path)} waypoints across {total_stats['segments']} segment(s)",
            route_geojson=geojson,
            stats=total_stats
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
