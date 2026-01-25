# NIMBY Route Finder

Automatic railway route generation with real-world elevation data and physical constraints.

## Features

-   🗺️ **Interactive Map Interface** - Click to set start/end points on any location worldwide
-   🏔️ **Elevation-Aware Routing** - Uses terrain data to find feasible railway alignments
-   ⚙️ **Configurable Constraints**:
    -   Maximum slope (grade) percentage
    -   Minimum curve radius
    -   Grid resolution for pathfinding
-   📊 **Route Statistics** - Distance, elevation gain, max slope encountered
-   📈 **Elevation Profile Chart** - Visualize the route's terrain

## Architecture

```
nimby-route-finder/
├── backend/                 # Python FastAPI server
│   ├── app/
│   │   ├── main.py         # API endpoints
│   │   ├── pathfinder.py   # Constrained A* algorithm
│   │   └── elevation.py    # Elevation data service
│   ├── requirements.txt
│   └── run.py
│
└── frontend/               # React + TypeScript + Vite
    ├── src/
    │   ├── App.tsx         # Main application component
    │   ├── components/
    │   │   └── ElevationChart.tsx
    │   └── types.ts
    └── package.json
```

## Quick Start

### Prerequisites

-   Python 3.10+
-   Node.js 18+
-   npm or yarn

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python run.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The app will be available at `http://localhost:5173`

## Configuration

### Environment Variables

| Variable              | Description                  | Default |
| --------------------- | ---------------------------- | ------- |
| `DEMO_MODE`           | Use synthetic elevation data | `true`  |
| `MAPBOX_ACCESS_TOKEN` | For real elevation data      | -       |

### Using Real Elevation Data

1. Get a free Mapbox access token from [mapbox.com](https://www.mapbox.com/)
2. Set the environment variable:
    ```bash
    set MAPBOX_ACCESS_TOKEN=your_token_here  # Windows
    export MAPBOX_ACCESS_TOKEN=your_token_here  # Linux/Mac
    ```
3. Set `DEMO_MODE=false`

## Algorithm Overview

The pathfinding uses a **state-lattice A\*** approach:

1. **State Space**: Each state is `(row, col, heading)` where heading is one of 8 directions (N, NE, E, SE, S, SW, W, NW)

2. **Constraints**:

    - **Slope**: Edges where `|elevation_change / distance| > max_slope` are rejected
    - **Curvature**: Heading can only change by a limited amount per step, derived from `min_radius`

3. **Cost Function**:

    ```
    cost = distance_weight × distance
         + elevation_weight × |Δelevation|
         + curvature_weight × heading_change
    ```

4. **Heuristic**: Euclidean distance (admissible)

## API Reference

### `POST /api/route`

Generate a route between two points.

**Request Body:**

```json
{
	"start_lat": 35.6762,
	"start_lng": 139.6503,
	"end_lat": 35.709,
	"end_lng": 139.7319,
	"max_slope_percent": 3.0,
	"min_curve_radius_m": 500,
	"grid_resolution_m": 50
}
```

**Response:**

```json
{
  "success": true,
  "message": "Route found with 127 waypoints",
  "route_geojson": {
    "type": "Feature",
    "geometry": {
      "type": "LineString",
      "coordinates": [[lng, lat, elevation], ...]
    },
    "properties": {
      "elevations": [...]
    }
  },
  "stats": {
    "total_distance_m": 8450.5,
    "max_slope_encountered": 2.8,
    "elevation_gain_m": 145.2,
    "nodes_expanded": 12543
  }
}
```

## Next Steps / Improvements

-   [ ] Add real elevation data caching
-   [ ] Support for waypoints (intermediate points)
-   [ ] Tunnel/bridge cost modeling
-   [ ] Export to GPX/KML
-   [ ] Integration with OpenRailwayMap
-   [ ] Multi-objective optimization (Pareto frontier)
-   [ ] GPU-accelerated pathfinding for larger grids

## License

MIT
