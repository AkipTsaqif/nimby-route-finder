# 🚂 NIMBY Route Finder - Backend

FastAPI backend for automatic railway route generation with terrain constraints.

## Overview

This backend provides the pathfinding engine and elevation data services for generating realistic railway routes. It features two pathfinding algorithms and supports multiple elevation data sources.

## Features

### Pathfinding Engines

#### Kinodynamic A\* (Default)

- Continuous state-space planning with motion primitives
- Arc-based movement for smooth, realistic curves
- Adaptive step sizing based on route length
- Built-in curvature constraints via motion primitives
- Direction gear support for switchbacks

#### Legacy Grid A\*

- Discrete 8-directional movement on elevation grid
- Faster for short routes or low-resolution needs
- Heading-aware state with curvature constraints

### Constraints & Features

- **Slope constraints**: Soft limit with penalty + hard limit never exceeded
- **Curvature constraints**: Minimum curve radius enforcement
- **Water avoidance**: Automatic detection via elevation and flatness analysis
- **Tunnel zones**: Ignore slope constraints in defined tunnel areas
- **Bridge zones**: Allow water crossing with reduced penalty
- **Auto tunnel/bridge**: Automatic shortcut detection when blocked
- **Road parallelism**: Optional constraint to follow existing roads
- **Switchbacks**: Optional 180° turns with configurable penalty

### Elevation Data Sources (Priority Order)

1. **Local GeoTIFF files** - Highest quality, fastest access
2. **Mapbox Terrain-RGB** - Global coverage with API key
3. **Synthetic demo terrain** - Fallback for testing

## Installation

### Prerequisites

- Python 3.10 or higher
- (Optional) Conda for rasterio on Windows

### Setup

1. Create and activate virtual environment:

    ```bash
    python -m venv venv

    # Windows
    venv\Scripts\activate

    # Linux/macOS
    source venv/bin/activate
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Configure environment:

    ```bash
    cp .env.example .env
    ```

    Edit `.env`:

    ```dotenv
    # Set to "false" to use real elevation data
    DEMO_MODE=true

    # Optional: Mapbox token for Terrain-RGB tiles
    MAPBOX_ACCESS_TOKEN=your_token_here
    ```

4. (Optional) Add DEM files to `data/dem/` - see [DEM README](data/dem/README.md)

## Running

### Development Server

```bash
python run.py
```

The API will be available at `http://localhost:8000`

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Reference

### Generate Route

**POST** `/api/route`

Generate a railway route between two points with constraints.

#### Request Body

```json
{
	"start_lat": 35.6762,
	"start_lng": 139.6503,
	"end_lat": 35.6895,
	"end_lng": 139.6917,
	"max_slope_percent": 3.0,
	"min_curve_radius_m": 500.0,
	"hard_slope_limit_percent": 8.0,
	"downsampling_factor": 1,
	"padding_factor": 0.3,
	"allow_water_crossing": false,
	"allow_switchbacks": false,
	"switchback_penalty": 5000.0,
	"min_switchback_interval": 50,
	"auto_tunnel_bridge": false,
	"max_jump_distance_m": 500.0,
	"elevation_tolerance_m": 10.0,
	"road_parallel_enabled": false,
	"road_parallel_threshold_deg": 30.0,
	"road_min_separation_m": 10.0,
	"road_max_separation_m": 50.0,
	"waypoints": [{ "lat": 35.68, "lng": 139.67, "heading": null }],
	"tunnels": [
		{
			"entry": { "lat": 35.68, "lng": 139.66 },
			"exit": { "lat": 35.69, "lng": 139.67 }
		}
	],
	"bridges": [
		{
			"start": { "lat": 35.69, "lng": 139.68 },
			"end": { "lat": 35.7, "lng": 139.69 }
		}
	],
	"start_heading": null,
	"end_heading": null,
	"use_kinodynamic": true,
	"max_iterations": 500000
}
```

#### Response

```json
{
  "success": true,
  "message": "Route found with 150 waypoints in 2.34s",
  "route_geojson": {
    "type": "Feature",
    "geometry": {
      "type": "LineString",
      "coordinates": [[lng, lat, elevation], ...]
    },
    "properties": {
      "elevations": [...],
      "curvatures": [...],
      "structures": [...],
      "tunnels": [...],
      "bridges": [...],
      "waypoints": [...]
    }
  },
  "stats": {
    "path_length": 150,
    "total_distance_m": 5432.1,
    "elevation_gain_m": 45.2,
    "max_slope_percent": 2.8,
    "iterations": 12345,
    "nodes_expanded": 8765,
    "elapsed_time_s": 2.34
  }
}
```

### Get Elevation

**GET** `/api/elevation?lat={lat}&lng={lng}`

Get elevation at a single point.

#### Response

```json
{
	"lat": 35.6762,
	"lng": 139.6503,
	"elevation_m": 42.5
}
```

### Health Check

**GET** `/health`

```json
{
	"status": "healthy"
}
```

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app, routes, request handling
│   ├── pathfinder.py     # Legacy grid-based A* implementation
│   ├── kinodynamic.py    # Kinodynamic A* with motion primitives
│   ├── elevation.py      # Elevation data service (GeoTIFF, Mapbox, demo)
│   └── roads.py          # Road data from OpenStreetMap Overpass API
├── data/
│   └── dem/
│       └── README.md     # Guide for adding DEM files
├── requirements.txt
├── run.py                # Development server runner
├── .env.example          # Environment variable template
├── test_astar.py         # Pathfinder tests
└── test_debug.py         # Debug utilities
```

## Configuration

### Environment Variables

| Variable              | Default      | Description                                |
| --------------------- | ------------ | ------------------------------------------ |
| `DEMO_MODE`           | `true`       | Use synthetic terrain instead of real data |
| `MAPBOX_ACCESS_TOKEN` | -            | Mapbox API token for Terrain-RGB           |
| `DEM_FOLDER`          | `./data/dem` | Path to local DEM files                    |

### Kinodynamic Config Parameters

| Parameter                   | Default  | Description                          |
| --------------------------- | -------- | ------------------------------------ |
| `step_distance_m`           | adaptive | Length of motion primitive arcs      |
| `num_curvature_samples`     | 5-7      | Number of discrete curvature options |
| `position_bucket_size`      | 5.0m     | Spatial hash resolution              |
| `heading_bucket_degrees`    | 10.0°    | Heading discretization               |
| `heuristic_weight`          | 1.5      | A\* heuristic multiplier             |
| `goal_tolerance_multiplier` | 1.5      | Goal proximity tolerance             |

## Dependencies

See `requirements.txt` for full list:

- **fastapi** - Web framework
- **uvicorn** - ASGI server
- **numpy** - Numerical computations
- **scipy** - Scientific algorithms
- **rasterio** - GeoTIFF processing
- **Pillow** - Image processing
- **httpx** - Async HTTP client
- **python-dotenv** - Environment configuration

## Troubleshooting

### "rasterio not installed"

On Windows, rasterio may require conda:

```bash
conda install -c conda-forge rasterio
```

### "Could not fetch elevation data"

1. Set `DEMO_MODE=true` for testing
2. Or configure Mapbox token
3. Or add local DEM files

### Slow performance

- Increase `downsampling_factor` for coarser search
- Reduce `max_iterations`
- Use kinodynamic pathfinder (adapts automatically)

## License

MIT License
