export interface LatLng {
	lat: number;
	lng: number;
}

// Waypoint with optional heading direction
export interface WaypointWithHeading {
	lat: number;
	lng: number;
	heading?: number; // Degrees, 0=North, clockwise. undefined = auto-compute
}

export interface TunnelPortal {
	entry: LatLng;
	exit: LatLng;
}

export interface BridgeMarker {
	start: LatLng;
	end: LatLng;
}

export interface RouteStats {
	nodes_expanded: number;
	max_queue_size: number;
	path_length: number;
	total_distance_m: number;
	max_slope_encountered?: number;
	max_slope_percent?: number;
	elevation_gain_m?: number;
	water_crossings?: number;
	segments?: number;
	warning?: string;
	error?: string;
	iterations?: number;
	elapsed_time?: number;
	elapsed_time_s?: number;
	// Failure info (when success=false but partial path exists)
	failure_location?: [number, number]; // [lng, lat]
	failure_location_backward?: [number, number]; // [lng, lat] - for bidirectional search
	failure_segment?: number;
	best_distance_remaining?: number;
}

export interface ElevationProfilePoint {
	distance: number; // Cumulative distance in meters
	elevation: number; // Elevation in meters
	lat: number;
	lng: number;
}

export interface GridNode {
	lat: number;
	lng: number;
	elevation: number;
	row: number;
	col: number;
}

export interface RouteGeoJSONProperties {
	elevations: number[];
	elevation_profile?: ElevationProfilePoint[]; // New: includes distance and position
	distances?: number[]; // Cumulative distances in meters
	total_distance_m?: number;
	waypoint_count: number;
	original_waypoints?: number;
	smoothed?: boolean;
	min_curve_radius_m?: number;
	structure_types?: string[]; // 'normal', 'tunnel', or 'bridge' for each point
	tunnels?: Array<{ entry: number[]; exit: number[] }>;
	bridges?: Array<{ start: number[]; end: number[] }>;
	waypoints?: number[][];
	auto_tunnels?: Array<{ start: number[]; end: number[] }>;
	auto_bridges?: Array<{ start: number[]; end: number[] }>;
	grid_nodes?: GridNode[]; // Original grid nodes for shadow visualization
	// Partial path info
	is_partial?: boolean;
	failure_point?: [number, number]; // [lng, lat]
	failure_point_backward?: [number, number]; // [lng, lat] - for bidirectional search
	has_gap?: boolean; // True if path has discontinuity (forward + backward partial paths)
	pathfinder?: "legacy" | "kinodynamic";
}

export interface RouteGeometry {
	type: "LineString" | "MultiLineString";
	coordinates: number[][] | number[][][]; // LineString: [lng, lat, elevation][] or MultiLineString: [lng, lat][][]
}

export interface RouteGeoJSON {
	type: "Feature";
	properties: RouteGeoJSONProperties;
	geometry: RouteGeometry;
}

export interface RouteResponse {
	success: boolean;
	message: string;
	route_geojson?: RouteGeoJSON;
	stats?: RouteStats;
}
