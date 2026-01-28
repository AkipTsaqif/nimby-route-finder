# 🚂 NIMBY Route Finder - Frontend

React-based interactive web interface for railway route planning with terrain visualization.

## Overview

This frontend provides an interactive map interface for planning railway routes. Users can set start/end points, add waypoints, define tunnels and bridges, configure route constraints, and visualize the generated route with elevation profiles.

## Features

### Interactive Map

- **Leaflet-based mapping** with multiple tile layers:
    - OpenStreetMap (default)
    - OpenRailwayMap (railway overlay)

- **Context menu** (right-click) for:
    - Setting start/end points
    - Adding waypoints
    - Defining tunnel entry/exit
    - Defining bridge start/end

- **Marker interactions**:
    - Click to select and set approach direction
    - Drag to move points
    - Visual feedback for pending operations

### Route Visualization

- **Route path** displayed as colored polyline
- **Partial path display** when route generation fails
- **Failure point indicator** showing where pathfinding got stuck
- **Grid node shadows** (optional) for debugging search grid
- **Hover synchronization** between map and chart

### Elevation Profile

- **Interactive chart** using Recharts
- **Hover to highlight** position on map
- **Statistics display**:
    - Total distance
    - Elevation gain/loss
    - Min/max elevation

### Configuration Panel

- **Basic constraints**:
    - Maximum slope (% grade)
    - Minimum curve radius
    - Downsampling factor

- **Advanced options**:
    - Hard slope limit
    - Search area padding
    - Pathfinder selection (Kinodynamic vs Legacy)
    - Max iterations

- **Switchback control**:
    - Enable/disable 180° turns
    - Penalty and interval settings

- **Auto tunnel/bridge**:
    - Jump distance threshold
    - Elevation tolerance

- **Road parallelism**:
    - Angle threshold
    - Min/max separation distances

### State Persistence

- All configuration saved to localStorage
- Settings persist across browser sessions

## Installation

### Prerequisites

- Node.js 18+ or Bun
- Backend server running (see backend README)

### Setup with Bun (Recommended)

```bash
# Install dependencies
bun install

# Start development server
bun dev
```

### Setup with npm

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:5173`

## Usage

### Basic Workflow

1. **Set start point**: Right-click → "Set Start Point"
2. **Set end point**: Right-click → "Set End Point"
3. **Configure constraints** in sidebar
4. **Click "Generate Route"**

### Adding Waypoints

- Right-click → "Add Waypoint"
- Route will pass through all waypoints in order
- Remove via sidebar list

### Setting Direction

1. Click on a start, end, or waypoint marker
2. Click elsewhere on map to set approach direction
3. Direction arrow appears on marker
4. Clear via marker popup

### Adding Tunnels

1. Right-click → "Add Tunnel Entry"
2. Status indicator appears in sidebar
3. Right-click → "Add Tunnel Exit"
4. Tunnel ignores slope constraints between entry/exit

### Adding Bridges

1. Right-click → "Add Bridge Start"
2. Status indicator appears in sidebar
3. Right-click → "Add Bridge End"
4. Bridge allows water crossing with reduced penalty

### Understanding Results

- **Green route**: Successful path
- **Red marker**: Failure point (if partial path)
- **Stats panel**: Distance, elevation, iterations, warnings
- **Elevation chart**: Interactive profile view

## Project Structure

```
frontend/
├── src/
│   ├── App.tsx               # Main application component
│   ├── main.tsx              # React entry point
│   ├── index.css             # Global styles
│   ├── types.ts              # TypeScript type definitions
│   ├── vite-env.d.ts         # Vite environment types
│   └── components/
│       └── ElevationChart.tsx # Elevation profile component
├── index.html                # HTML entry point
├── package.json              # Dependencies and scripts
├── tsconfig.json             # TypeScript configuration
├── tsconfig.node.json        # Node TypeScript config
├── vite.config.ts            # Vite configuration
└── bun.lockb                 # Bun lock file
```

## Scripts

```bash
# Development server with hot reload
bun dev       # or npm run dev

# Production build
bun run build # or npm run build

# Preview production build
bun run preview # or npm run preview
```

## Configuration

### Vite Proxy

The development server proxies API requests to the backend. Configure in `vite.config.ts`:

```typescript
export default defineConfig({
	server: {
		proxy: {
			"/api": "http://localhost:8000",
		},
	},
});
```

### Environment

No environment variables required for frontend. Backend URL is configured via Vite proxy.

## Dependencies

### Production

- **react** / **react-dom** - UI framework
- **leaflet** / **react-leaflet** - Interactive mapping
- **recharts** - Charts and data visualization

### Development

- **typescript** - Type safety
- **vite** - Build tool and dev server
- **@vitejs/plugin-react** - React integration for Vite
- **@types/leaflet** / **@types/react** - TypeScript definitions

## Type Definitions

Key types defined in `types.ts`:

```typescript
interface LatLng {
	lat: number;
	lng: number;
}

interface WaypointWithHeading extends LatLng {
	heading?: number; // Degrees, 0=North, clockwise
}

interface RouteResponse {
	success: boolean;
	message: string;
	route_geojson?: RouteGeoJSON;
	stats?: RouteStats;
}

interface RouteStats {
	total_distance_m: number;
	elevation_gain_m?: number;
	max_slope_percent?: number;
	iterations?: number;
	elapsed_time_s?: number;
	// ... and more
}
```

## Browser Support

- Chrome/Edge (recommended)
- Firefox
- Safari

Requires modern JavaScript features (ES2020+).

## Troubleshooting

### "Failed to connect to server"

- Ensure backend is running at `http://localhost:8000`
- Check browser console for CORS errors
- Verify Vite proxy configuration

### Map not loading

- Check internet connection (tile servers)
- Verify Leaflet CSS is imported
- Check for JavaScript errors in console

### Route generation slow

- Reduce max iterations
- Increase downsampling factor
- Use kinodynamic pathfinder (adapts automatically)

## License

MIT License
