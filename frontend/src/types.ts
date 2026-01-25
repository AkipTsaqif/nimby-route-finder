export interface LatLng {
	lat: number;
	lng: number;
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
	elevation_gain_m?: number;
	water_crossings?: number;
	segments?: number;
	warning?: string;
	error?: string;
}

export interface ElevationProfilePoint {
	distance: number; // Cumulative distance in meters
	elevation: number; // Elevation in meters
	lat: number;
	lng: number;
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
}

export interface RouteGeometry {
	type: "LineString";
	coordinates: number[][]; // [lng, lat, elevation]
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
