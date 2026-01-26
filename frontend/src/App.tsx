import { useState, useCallback, useEffect, useMemo } from "react";
import {
    MapContainer,
    TileLayer,
    Marker,
    Polyline,
    useMapEvents,
    Popup,
    CircleMarker,
    Tooltip,
} from "react-leaflet";
import L from "leaflet";
import ElevationChart from "./components/ElevationChart";
import {
    RouteResponse,
    LatLng,
    TunnelPortal,
    BridgeMarker,
    ElevationProfilePoint,
    GridNode,
} from "./types";

// Fix for default marker icons in React-Leaflet
import icon from "leaflet/dist/images/marker-icon.png";
import iconShadow from "leaflet/dist/images/marker-shadow.png";

const DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41],
});
L.Marker.prototype.options.icon = DefaultIcon;

// Custom marker icons
const startIcon = L.divIcon({
    className: "custom-marker",
    html: '<div style="background:#4ecca3;width:20px;height:20px;border-radius:50%;border:3px solid white;box-shadow:0 2px 5px rgba(0,0,0,0.3);"></div>',
    iconSize: [20, 20],
    iconAnchor: [10, 10],
});

const endIcon = L.divIcon({
    className: "custom-marker",
    html: '<div style="background:#e74c3c;width:20px;height:20px;border-radius:50%;border:3px solid white;box-shadow:0 2px 5px rgba(0,0,0,0.3);"></div>',
    iconSize: [20, 20],
    iconAnchor: [10, 10],
});

const waypointIcon = L.divIcon({
    className: "custom-marker",
    html: '<div style="background:#f39c12;width:16px;height:16px;border-radius:50%;border:2px solid white;box-shadow:0 2px 5px rgba(0,0,0,0.3);"></div>',
    iconSize: [16, 16],
    iconAnchor: [8, 8],
});

const tunnelIcon = L.divIcon({
    className: "custom-marker",
    html: '<div style="background:#9b59b6;width:18px;height:18px;border-radius:3px;border:2px solid white;box-shadow:0 2px 5px rgba(0,0,0,0.3);"></div>',
    iconSize: [18, 18],
    iconAnchor: [9, 9],
});

const bridgeIcon = L.divIcon({
    className: "custom-marker",
    html: '<div style="background:#3498db;width:18px;height:18px;border-radius:3px;border:2px solid white;box-shadow:0 2px 5px rgba(0,0,0,0.3);"></div>',
    iconSize: [18, 18],
    iconAnchor: [9, 9],
});

type ContextMenuAction =
    | "set-start"
    | "set-end"
    | "add-waypoint"
    | "add-tunnel-entry"
    | "add-tunnel-exit"
    | "add-bridge-start"
    | "add-bridge-end";

interface ContextMenuState {
    visible: boolean;
    x: number;
    y: number;
    latlng: LatLng | null;
}

function MapClickHandler({
    onContextMenu,
    onLeftClick,
}: {
    onContextMenu: (latlng: LatLng, x: number, y: number) => void;
    onLeftClick: () => void;
}) {
    useMapEvents({
        contextmenu(e) {
            e.originalEvent.preventDefault();
            onContextMenu(
                { lat: e.latlng.lat, lng: e.latlng.lng },
                e.originalEvent.clientX,
                e.originalEvent.clientY
            );
        },
        click() {
            onLeftClick();
        },
    });
    return null;
}

// Interactive polyline component that supports hover events
function InteractivePolyline({
    positions,
    onHover,
    onMouseLeave,
}: {
    positions: [number, number][];
    onHover: (lat: number, lng: number) => void;
    onMouseLeave: () => void;
}) {
    const polylineRef = useCallback(
        (polyline: L.Polyline | null) => {
            if (polyline) {
                polyline.on("mousemove", (e: L.LeafletMouseEvent) => {
                    onHover(e.latlng.lat, e.latlng.lng);
                });
                polyline.on("mouseout", () => {
                    onMouseLeave();
                });
            }
        },
        [onHover, onMouseLeave]
    );

    return (
        <Polyline
            ref={polylineRef}
            positions={positions}
            pathOptions={{
                color: "#4ecca3",
                weight: 6,
                opacity: 0.9,
            }}
            interactive={true}
        />
    );
}

function App() {
    const [startPoint, setStartPoint] = useState<LatLng | null>(null);
    const [endPoint, setEndPoint] = useState<LatLng | null>(null);
    const [waypoints, setWaypoints] = useState<LatLng[]>([]);
    const [tunnels, setTunnels] = useState<TunnelPortal[]>([]);
    const [bridges, setBridges] = useState<BridgeMarker[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [routeData, setRouteData] = useState<RouteResponse | null>(null);

    // Partial tunnel/bridge being created
    const [pendingTunnelEntry, setPendingTunnelEntry] = useState<LatLng | null>(
        null
    );
    const [pendingBridgeStart, setPendingBridgeStart] = useState<LatLng | null>(
        null
    );

    // Context menu state
    const [contextMenu, setContextMenu] = useState<ContextMenuState>({
        visible: false,
        x: 0,
        y: 0,
        latlng: null,
    });

    // Constraints
    const [maxSlope, setMaxSlope] = useState(3.0);
    const [minRadius, setMinRadius] = useState(500);
    const [downsamplingFactor, setDownsamplingFactor] = useState(1);
    // Advanced options
    const [hardSlopeLimit, setHardSlopeLimit] = useState(8.0);
    const [paddingFactor, setPaddingFactor] = useState(0.3);
    const [showAdvanced, setShowAdvanced] = useState(false);

    // Switchback control
    const [allowSwitchbacks, setAllowSwitchbacks] = useState(false);
    const [switchbackPenalty, setSwitchbackPenalty] = useState(5000);
    const [minSwitchbackInterval, setMinSwitchbackInterval] = useState(50);

    // Auto tunnel/bridge detection
    const [autoTunnelBridge, setAutoTunnelBridge] = useState(false);
    const [maxJumpDistance, setMaxJumpDistance] = useState(500);
    const [elevationTolerance, setElevationTolerance] = useState(10);

    // Road parallelism constraints
    const [roadParallelEnabled, setRoadParallelEnabled] = useState(false);
    const [roadParallelThreshold, setRoadParallelThreshold] = useState(30);
    const [roadMinSeparation, setRoadMinSeparation] = useState(10);
    const [roadMaxSeparation, setRoadMaxSeparation] = useState(50);

    // Map tile layer
    type MapTileType = "osm" | "openrailwaymap";
    const [mapTileType, setMapTileType] = useState<MapTileType>("osm");

    // Elevation chart hover position
    const [hoverPosition, setHoverPosition] =
        useState<ElevationProfilePoint | null>(null);

    // Route hover position (from map interaction)
    const [routeHoverPosition, setRouteHoverPosition] =
        useState<ElevationProfilePoint | null>(null);

    // Show grid node shadows
    const [showGridNodes, setShowGridNodes] = useState(false);

    const handleContextMenu = useCallback(
        (latlng: LatLng, x: number, y: number) => {
            // Adjust position to prevent menu from being cut off at screen edges
            const menuWidth = 200;
            const menuHeight = 220;
            const viewportWidth = window.innerWidth;
            const viewportHeight = window.innerHeight;

            let adjustedX = x;
            let adjustedY = y;

            // Check right edge
            if (x + menuWidth > viewportWidth) {
                adjustedX = viewportWidth - menuWidth - 10;
            }

            // Check bottom edge
            if (y + menuHeight > viewportHeight) {
                adjustedY = viewportHeight - menuHeight - 10;
            }

            // Ensure not negative
            adjustedX = Math.max(10, adjustedX);
            adjustedY = Math.max(10, adjustedY);

            setContextMenu({
                visible: true,
                x: adjustedX,
                y: adjustedY,
                latlng,
            });
        },
        []
    );

    const hideContextMenu = useCallback(() => {
        setContextMenu((prev) => ({ ...prev, visible: false }));
    }, []);

    const handleContextMenuAction = useCallback(
        (action: ContextMenuAction) => {
            if (!contextMenu.latlng) return;
            const latlng = contextMenu.latlng;

            switch (action) {
                case "set-start":
                    setStartPoint(latlng);
                    setRouteData(null);
                    setError(null);
                    break;
                case "set-end":
                    setEndPoint(latlng);
                    setRouteData(null);
                    setError(null);
                    break;
                case "add-waypoint":
                    setWaypoints((prev) => [...prev, latlng]);
                    setRouteData(null);
                    break;
                case "add-tunnel-entry":
                    setPendingTunnelEntry(latlng);
                    break;
                case "add-tunnel-exit":
                    if (pendingTunnelEntry) {
                        setTunnels((prev) => [
                            ...prev,
                            { entry: pendingTunnelEntry, exit: latlng },
                        ]);
                        setPendingTunnelEntry(null);
                        setRouteData(null);
                    }
                    break;
                case "add-bridge-start":
                    setPendingBridgeStart(latlng);
                    break;
                case "add-bridge-end":
                    if (pendingBridgeStart) {
                        setBridges((prev) => [
                            ...prev,
                            { start: pendingBridgeStart, end: latlng },
                        ]);
                        setPendingBridgeStart(null);
                        setRouteData(null);
                    }
                    break;
            }
            hideContextMenu();
        },
        [
            contextMenu.latlng,
            pendingTunnelEntry,
            pendingBridgeStart,
            hideContextMenu,
        ]
    );

    const generateRoute = async () => {
        if (!startPoint || !endPoint) return;

        setLoading(true);
        setError(null);
        setRouteData(null);

        try {
            const response = await fetch("/api/route", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    start_lat: startPoint.lat,
                    start_lng: startPoint.lng,
                    end_lat: endPoint.lat,
                    end_lng: endPoint.lng,
                    max_slope_percent: maxSlope,
                    min_curve_radius_m: minRadius,
                    downsampling_factor: downsamplingFactor,
                    hard_slope_limit_percent: hardSlopeLimit,
                    padding_factor: paddingFactor,
                    allow_water_crossing: false,
                    // Switchback control
                    allow_switchbacks: allowSwitchbacks,
                    switchback_penalty: switchbackPenalty,
                    min_switchback_interval: minSwitchbackInterval,
                    // Auto tunnel/bridge detection
                    auto_tunnel_bridge: autoTunnelBridge,
                    max_jump_distance_m: maxJumpDistance,
                    elevation_tolerance_m: elevationTolerance,
                    // Road parallelism constraints
                    road_parallel_enabled: roadParallelEnabled,
                    road_parallel_threshold_deg: roadParallelThreshold,
                    road_min_separation_m: roadMinSeparation,
                    road_max_separation_m: roadMaxSeparation,
                    // Waypoints, tunnels, bridges
                    waypoints: waypoints,
                    tunnels: tunnels,
                    bridges: bridges,
                }),
            });

            const data: RouteResponse = await response.json();

            if (data.success) {
                setRouteData(data);
            } else {
                setError(data.message);
            }
        } catch (err) {
            setError(
                "Failed to connect to the server. Make sure the backend is running."
            );
        } finally {
            setLoading(false);
        }
    };

    const resetAll = () => {
        setStartPoint(null);
        setEndPoint(null);
        setWaypoints([]);
        setTunnels([]);
        setBridges([]);
        setPendingTunnelEntry(null);
        setPendingBridgeStart(null);
        setRouteData(null);
        setError(null);
    };

    const removeWaypoint = (index: number) => {
        setWaypoints((prev) => prev.filter((_, i) => i !== index));
        setRouteData(null);
    };

    const removeTunnel = (index: number) => {
        setTunnels((prev) => prev.filter((_, i) => i !== index));
        setRouteData(null);
    };

    const removeBridge = (index: number) => {
        setBridges((prev) => prev.filter((_, i) => i !== index));
        setRouteData(null);
    };

    // Extract route coordinates for Polyline
    const routeCoordinates: [number, number][] =
        routeData?.route_geojson?.geometry?.coordinates?.map(
            (coord: number[]) => [coord[1], coord[0]] as [number, number]
        ) || [];

    // Extract elevations for chart (legacy)
    const elevations: number[] =
        routeData?.route_geojson?.properties?.elevations || [];

    // Extract elevation profile with distance and position data
    const elevationProfile: ElevationProfilePoint[] =
        routeData?.route_geojson?.properties?.elevation_profile || [];

    // Extract grid nodes for shadow visualization
    const gridNodes: GridNode[] =
        routeData?.route_geojson?.properties?.grid_nodes || [];

    // Handle elevation chart hover
    const handleElevationHover = useCallback(
        (point: ElevationProfilePoint | null) => {
            setHoverPosition(point);
        },
        []
    );

    // Find closest point on route to a given lat/lng (interpolates along segments)
    const findClosestRoutePoint = useCallback(
        (lat: number, lng: number): ElevationProfilePoint | null => {
            if (elevationProfile.length === 0) return null;
            if (elevationProfile.length === 1) return elevationProfile[0];

            let closestPoint: ElevationProfilePoint | null = null;
            let minDist = Infinity;

            // Check each segment of the route
            for (let i = 0; i < elevationProfile.length - 1; i++) {
                const p1 = elevationProfile[i];
                const p2 = elevationProfile[i + 1];

                // Find closest point on line segment p1-p2 to (lat, lng)
                const dx = p2.lng - p1.lng;
                const dy = p2.lat - p1.lat;
                const segLenSq = dx * dx + dy * dy;

                let t = 0;
                if (segLenSq > 0) {
                    // Project point onto line segment
                    t = ((lng - p1.lng) * dx + (lat - p1.lat) * dy) / segLenSq;
                    t = Math.max(0, Math.min(1, t)); // Clamp to segment
                }

                // Interpolated point on segment
                const interpLng = p1.lng + t * dx;
                const interpLat = p1.lat + t * dy;
                const interpDist =
                    p1.distance + t * (p2.distance - p1.distance);
                const interpElev =
                    p1.elevation + t * (p2.elevation - p1.elevation);

                // Distance from mouse to interpolated point
                const dist = Math.sqrt(
                    Math.pow(interpLat - lat, 2) + Math.pow(interpLng - lng, 2)
                );

                if (dist < minDist) {
                    minDist = dist;
                    closestPoint = {
                        lat: interpLat,
                        lng: interpLng,
                        distance: interpDist,
                        elevation: interpElev,
                    };
                }
            }

            return closestPoint;
        },
        [elevationProfile]
    );

    // Handle route hover from map
    const handleRouteHover = useCallback(
        (lat: number, lng: number) => {
            const point = findClosestRoutePoint(lat, lng);
            setRouteHoverPosition(point);
            // Also update chart hover
            setHoverPosition(point);
        },
        [findClosestRoutePoint]
    );

    const handleRouteMouseLeave = useCallback(() => {
        setRouteHoverPosition(null);
    }, []);

    // Close context menu when clicking outside
    useEffect(() => {
        const handleClick = () => hideContextMenu();
        if (contextMenu.visible) {
            document.addEventListener("click", handleClick);
            return () => document.removeEventListener("click", handleClick);
        }
    }, [contextMenu.visible, hideContextMenu]);

    return (
        <div className="app-container">
            <div className="sidebar">
                <h1>🚂 NIMBY Route Finder</h1>
                <p className="info-text">
                    Right-click on the map to add start/end points, waypoints,
                    tunnels, and bridges.
                </p>

                <div className="coordinates">
                    <p>
                        <span className="label">Start: </span>
                        {startPoint ? (
                            <span className="coord">
                                {startPoint.lat.toFixed(5)},{" "}
                                {startPoint.lng.toFixed(5)}
                            </span>
                        ) : (
                            <span className="coord" style={{ color: "#666" }}>
                                Right-click map → Set Start
                            </span>
                        )}
                    </p>
                    <p>
                        <span className="label">End: </span>
                        {endPoint ? (
                            <span className="coord">
                                {endPoint.lat.toFixed(5)},{" "}
                                {endPoint.lng.toFixed(5)}
                            </span>
                        ) : (
                            <span className="coord" style={{ color: "#666" }}>
                                Right-click map → Set End
                            </span>
                        )}
                    </p>
                </div>

                {/* Show pending tunnel/bridge status */}
                {pendingTunnelEntry && (
                    <div
                        style={{
                            background: "#9b59b6",
                            color: "white",
                            padding: "0.5rem",
                            borderRadius: "4px",
                            marginBottom: "0.5rem",
                        }}
                    >
                        🚇 Tunnel entry set. Right-click to set exit point.
                    </div>
                )}
                {pendingBridgeStart && (
                    <div
                        style={{
                            background: "#3498db",
                            color: "white",
                            padding: "0.5rem",
                            borderRadius: "4px",
                            marginBottom: "0.5rem",
                        }}
                    >
                        🌉 Bridge start set. Right-click to set end point.
                    </div>
                )}

                {/* Show waypoints, tunnels, bridges */}
                {(waypoints.length > 0 ||
                    tunnels.length > 0 ||
                    bridges.length > 0) && (
                    <div
                        className="markers-list"
                        style={{ marginBottom: "1rem" }}
                    >
                        {waypoints.map((wp, i) => (
                            <div
                                key={`wp-${i}`}
                                className="marker-item"
                                style={{
                                    display: "flex",
                                    justifyContent: "space-between",
                                    alignItems: "center",
                                    padding: "0.25rem 0",
                                    borderBottom: "1px solid #333",
                                }}
                            >
                                <span style={{ color: "#f39c12" }}>
                                    📍 Waypoint {i + 1}: {wp.lat.toFixed(4)},{" "}
                                    {wp.lng.toFixed(4)}
                                </span>
                                <button
                                    onClick={() => removeWaypoint(i)}
                                    style={{
                                        background: "transparent",
                                        border: "none",
                                        color: "#e74c3c",
                                        cursor: "pointer",
                                    }}
                                >
                                    ✕
                                </button>
                            </div>
                        ))}
                        {tunnels.map((_t, i) => (
                            <div
                                key={`tunnel-${i}`}
                                className="marker-item"
                                style={{
                                    display: "flex",
                                    justifyContent: "space-between",
                                    alignItems: "center",
                                    padding: "0.25rem 0",
                                    borderBottom: "1px solid #333",
                                }}
                            >
                                <span style={{ color: "#9b59b6" }}>
                                    🚇 Tunnel {i + 1}
                                </span>
                                <button
                                    onClick={() => removeTunnel(i)}
                                    style={{
                                        background: "transparent",
                                        border: "none",
                                        color: "#e74c3c",
                                        cursor: "pointer",
                                    }}
                                >
                                    ✕
                                </button>
                            </div>
                        ))}
                        {bridges.map((_b, i) => (
                            <div
                                key={`bridge-${i}`}
                                className="marker-item"
                                style={{
                                    display: "flex",
                                    justifyContent: "space-between",
                                    alignItems: "center",
                                    padding: "0.25rem 0",
                                    borderBottom: "1px solid #333",
                                }}
                            >
                                <span style={{ color: "#3498db" }}>
                                    🌉 Bridge {i + 1}
                                </span>
                                <button
                                    onClick={() => removeBridge(i)}
                                    style={{
                                        background: "transparent",
                                        border: "none",
                                        color: "#e74c3c",
                                        cursor: "pointer",
                                    }}
                                >
                                    ✕
                                </button>
                            </div>
                        ))}
                    </div>
                )}

                <h2>Constraints</h2>

                <div className="control-group">
                    <label>Target Maximum Slope</label>
                    <input
                        type="range"
                        min="0.5"
                        max="10"
                        step="0.5"
                        value={maxSlope}
                        onChange={(e) =>
                            setMaxSlope(parseFloat(e.target.value))
                        }
                    />
                    <span className="value">{maxSlope.toFixed(1)}%</span>
                </div>

                <div className="control-group">
                    <label>Minimum Curve Radius</label>
                    <input
                        type="range"
                        min="100"
                        max="2000"
                        step="50"
                        value={minRadius}
                        onChange={(e) => setMinRadius(parseInt(e.target.value))}
                    />
                    <span className="value">{minRadius}m</span>
                </div>

                <div className="control-group">
                    <label>Resolution</label>
                    <input
                        type="range"
                        min="1"
                        max="4"
                        step="1"
                        value={downsamplingFactor}
                        onChange={(e) =>
                            setDownsamplingFactor(parseInt(e.target.value))
                        }
                    />
                    <span className="value">
                        {downsamplingFactor}x{" "}
                        {downsamplingFactor === 1
                            ? "(highest)"
                            : `(~${30 * downsamplingFactor}m)`}
                    </span>
                </div>

                <button
                    className="toggle-advanced"
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    style={{
                        background: "transparent",
                        border: "1px solid #4ecca3",
                        color: "#4ecca3",
                        padding: "0.5rem",
                        marginBottom: "1rem",
                        cursor: "pointer",
                        width: "100%",
                        borderRadius: "4px",
                    }}
                >
                    {showAdvanced ? "▲ Hide" : "▼ Show"} Advanced Options
                </button>

                {showAdvanced && (
                    <div
                        className="advanced-options"
                        style={{ marginBottom: "1rem" }}
                    >
                        <div className="control-group">
                            <label>Hard Slope Limit</label>
                            <input
                                type="range"
                                min="3"
                                max="15"
                                step="0.5"
                                value={hardSlopeLimit}
                                onChange={(e) =>
                                    setHardSlopeLimit(
                                        parseFloat(e.target.value)
                                    )
                                }
                            />
                            <span className="value">
                                {hardSlopeLimit.toFixed(1)}%
                            </span>
                        </div>
                        <div className="control-group">
                            <label>Search Area Padding</label>
                            <input
                                type="range"
                                min="0.1"
                                max="1.0"
                                step="0.1"
                                value={paddingFactor}
                                onChange={(e) =>
                                    setPaddingFactor(parseFloat(e.target.value))
                                }
                            />
                            <span className="value">
                                {(paddingFactor * 100).toFixed(0)}%
                            </span>
                        </div>

                        {/* Switchback Control */}
                        <h3
                            style={{
                                marginTop: "1rem",
                                marginBottom: "0.5rem",
                                color: "#f39c12",
                            }}
                        >
                            🔄 Switchback Control
                        </h3>
                        <div
                            className="control-group"
                            style={{
                                display: "flex",
                                alignItems: "center",
                                gap: "0.5rem",
                            }}
                        >
                            <input
                                type="checkbox"
                                id="allowSwitchbacks"
                                checked={allowSwitchbacks}
                                onChange={(e) =>
                                    setAllowSwitchbacks(e.target.checked)
                                }
                                style={{ width: "auto" }}
                            />
                            <label
                                htmlFor="allowSwitchbacks"
                                style={{ cursor: "pointer" }}
                            >
                                Allow Switchbacks (180° turns)
                            </label>
                        </div>
                        {allowSwitchbacks && (
                            <>
                                <div className="control-group">
                                    <label>Switchback Penalty</label>
                                    <input
                                        type="range"
                                        min="1000"
                                        max="10000"
                                        step="500"
                                        value={switchbackPenalty}
                                        onChange={(e) =>
                                            setSwitchbackPenalty(
                                                parseInt(e.target.value)
                                            )
                                        }
                                    />
                                    <span className="value">
                                        {switchbackPenalty}
                                    </span>
                                </div>
                                <div className="control-group">
                                    <label>Min Interval (cells)</label>
                                    <input
                                        type="range"
                                        min="10"
                                        max="200"
                                        step="10"
                                        value={minSwitchbackInterval}
                                        onChange={(e) =>
                                            setMinSwitchbackInterval(
                                                parseInt(e.target.value)
                                            )
                                        }
                                    />
                                    <span className="value">
                                        {minSwitchbackInterval}
                                    </span>
                                </div>
                            </>
                        )}

                        {/* Auto Tunnel/Bridge Detection */}
                        <h3
                            style={{
                                marginTop: "1rem",
                                marginBottom: "0.5rem",
                                color: "#9b59b6",
                            }}
                        >
                            🚇 Auto Tunnel/Bridge
                        </h3>
                        <div
                            className="control-group"
                            style={{
                                display: "flex",
                                alignItems: "center",
                                gap: "0.5rem",
                            }}
                        >
                            <input
                                type="checkbox"
                                id="autoTunnelBridge"
                                checked={autoTunnelBridge}
                                onChange={(e) =>
                                    setAutoTunnelBridge(e.target.checked)
                                }
                                style={{ width: "auto" }}
                            />
                            <label
                                htmlFor="autoTunnelBridge"
                                style={{ cursor: "pointer" }}
                            >
                                Auto-detect tunnels/bridges
                            </label>
                        </div>
                        {autoTunnelBridge && (
                            <>
                                <div className="control-group">
                                    <label>Max Jump Distance</label>
                                    <input
                                        type="range"
                                        min="100"
                                        max="2000"
                                        step="100"
                                        value={maxJumpDistance}
                                        onChange={(e) =>
                                            setMaxJumpDistance(
                                                parseInt(e.target.value)
                                            )
                                        }
                                    />
                                    <span className="value">
                                        {maxJumpDistance}m
                                    </span>
                                </div>
                                <div className="control-group">
                                    <label>Elevation Tolerance</label>
                                    <input
                                        type="range"
                                        min="5"
                                        max="50"
                                        step="5"
                                        value={elevationTolerance}
                                        onChange={(e) =>
                                            setElevationTolerance(
                                                parseInt(e.target.value)
                                            )
                                        }
                                    />
                                    <span className="value">
                                        {elevationTolerance}m
                                    </span>
                                </div>
                                <p
                                    style={{
                                        fontSize: "0.8rem",
                                        color: "#888",
                                        marginTop: "0.25rem",
                                    }}
                                >
                                    When blocked, search for similar elevation
                                    within max distance.
                                </p>
                            </>
                        )}

                        {/* Road Parallelism Constraints */}
                        <h3
                            style={{
                                marginTop: "1rem",
                                marginBottom: "0.5rem",
                                color: "#3498db",
                            }}
                        >
                            🛤️ Road Parallelism
                        </h3>
                        <div
                            className="control-group"
                            style={{
                                display: "flex",
                                alignItems: "center",
                                gap: "0.5rem",
                            }}
                        >
                            <input
                                type="checkbox"
                                id="roadParallelEnabled"
                                checked={roadParallelEnabled}
                                onChange={(e) =>
                                    setRoadParallelEnabled(e.target.checked)
                                }
                                style={{ width: "auto" }}
                            />
                            <label
                                htmlFor="roadParallelEnabled"
                                style={{ cursor: "pointer" }}
                            >
                                Enable road parallel constraints
                            </label>
                        </div>
                        {roadParallelEnabled && (
                            <>
                                <div className="control-group">
                                    <label>Parallel Threshold (deg)</label>
                                    <input
                                        type="range"
                                        min="10"
                                        max="45"
                                        step="5"
                                        value={roadParallelThreshold}
                                        onChange={(e) =>
                                            setRoadParallelThreshold(
                                                parseInt(e.target.value)
                                            )
                                        }
                                    />
                                    <span className="value">
                                        ±{roadParallelThreshold}°
                                    </span>
                                </div>
                                <div className="control-group">
                                    <label>Min Separation</label>
                                    <input
                                        type="range"
                                        min="5"
                                        max="30"
                                        step="5"
                                        value={roadMinSeparation}
                                        onChange={(e) =>
                                            setRoadMinSeparation(
                                                parseInt(e.target.value)
                                            )
                                        }
                                    />
                                    <span className="value">
                                        {roadMinSeparation}m
                                    </span>
                                </div>
                                <div className="control-group">
                                    <label>Max Separation</label>
                                    <input
                                        type="range"
                                        min="30"
                                        max="200"
                                        step="10"
                                        value={roadMaxSeparation}
                                        onChange={(e) =>
                                            setRoadMaxSeparation(
                                                parseInt(e.target.value)
                                            )
                                        }
                                    />
                                    <span className="value">
                                        {roadMaxSeparation}m
                                    </span>
                                </div>
                                <p
                                    style={{
                                        fontSize: "0.8rem",
                                        color: "#888",
                                        marginTop: "0.25rem",
                                    }}
                                >
                                    When near and parallel to roads, maintain
                                    minimum separation distance.
                                </p>
                            </>
                        )}

                        {/* Grid Node Visualization Toggle */}
                        <h3
                            style={{
                                marginTop: "1rem",
                                marginBottom: "0.5rem",
                                color: "#999",
                                fontSize: "0.9rem",
                            }}
                        >
                            🔍 Visualization
                        </h3>
                        <div
                            className="control-group"
                            style={{
                                display: "flex",
                                alignItems: "center",
                                gap: "0.5rem",
                            }}
                        >
                            <input
                                type="checkbox"
                                id="showGridNodes"
                                checked={showGridNodes}
                                onChange={(e) =>
                                    setShowGridNodes(e.target.checked)
                                }
                                style={{ width: "auto" }}
                            />
                            <label
                                htmlFor="showGridNodes"
                                style={{ cursor: "pointer" }}
                            >
                                Show grid nodes (shadows)
                            </label>
                        </div>
                        <p
                            style={{
                                fontSize: "0.8rem",
                                color: "#888",
                                marginTop: "0.25rem",
                            }}
                        >
                            Show original pathfinding grid nodes along the
                            route. Hover over the route or chart to see node
                            info.
                        </p>

                        <p
                            style={{
                                fontSize: "0.8rem",
                                color: "#888",
                                marginTop: "0.5rem",
                            }}
                        >
                            Increase padding to search a larger area for routes
                            around obstacles.
                        </p>
                    </div>
                )}

                <button
                    className="generate-btn"
                    onClick={generateRoute}
                    disabled={!startPoint || !endPoint || loading}
                >
                    {loading ? "Generating..." : "Generate Route"}
                </button>

                <button
                    className="generate-btn"
                    onClick={resetAll}
                    style={{ background: "#666" }}
                >
                    Reset All
                </button>

                {error && <div className="error-message">{error}</div>}

                {routeData?.stats && (
                    <div className="stats-panel">
                        <h3>Route Statistics</h3>
                        {routeData.stats.warning && (
                            <div
                                style={{
                                    background: "#ff9800",
                                    color: "#000",
                                    padding: "0.5rem",
                                    borderRadius: "4px",
                                    marginBottom: "0.5rem",
                                    fontSize: "0.85rem",
                                }}
                            >
                                ⚠️ {routeData.stats.warning}
                            </div>
                        )}
                        {(routeData.stats.water_crossings ?? 0) > 0 && (
                            <div
                                style={{
                                    background: "#2196f3",
                                    color: "#fff",
                                    padding: "0.5rem",
                                    borderRadius: "4px",
                                    marginBottom: "0.5rem",
                                    fontSize: "0.85rem",
                                }}
                            >
                                🌊 Route crosses{" "}
                                {routeData.stats.water_crossings} water cell(s)
                            </div>
                        )}
                        <div className="stats-grid">
                            <div className="stat-item">
                                <span className="stat-label">Distance</span>
                                <span className="stat-value">
                                    {(
                                        routeData.stats.total_distance_m / 1000
                                    ).toFixed(2)}{" "}
                                    km
                                </span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Waypoints</span>
                                <span className="stat-value">
                                    {routeData.stats.path_length}
                                </span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Max Slope</span>
                                <span
                                    className="stat-value"
                                    style={{
                                        color:
                                            (routeData.stats
                                                .max_slope_encountered ?? 0) >
                                            maxSlope
                                                ? "#ff9800"
                                                : "#4ecca3",
                                    }}
                                >
                                    {routeData.stats.max_slope_encountered?.toFixed(
                                        2
                                    )}
                                    %
                                </span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">
                                    Elevation Gain
                                </span>
                                <span className="stat-value">
                                    {routeData.stats.elevation_gain_m?.toFixed(
                                        0
                                    )}
                                    m
                                </span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Segments</span>
                                <span className="stat-value">
                                    {routeData.stats.segments ?? 1}
                                </span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">
                                    Nodes Expanded
                                </span>
                                <span className="stat-value">
                                    {routeData.stats.nodes_expanded}
                                </span>
                            </div>
                        </div>
                    </div>
                )}

                {elevationProfile.length > 0 ? (
                    <ElevationChart
                        elevationProfile={elevationProfile}
                        onHover={handleElevationHover}
                    />
                ) : (
                    elevations.length > 0 && (
                        <ElevationChart
                            elevationProfile={elevations.map((elev, i) => ({
                                distance: i * 100,
                                elevation: elev,
                                lat: 0,
                                lng: 0,
                            }))}
                        />
                    )
                )}
            </div>

            <div className="map-container">
                {/* Map Tile Switcher */}
                <div
                    style={{
                        position: "absolute",
                        top: "10px",
                        right: "10px",
                        zIndex: 1000,
                        background: "#1e1e1e",
                        padding: "0.5rem",
                        borderRadius: "4px",
                        border: "1px solid #444",
                        display: "flex",
                        gap: "0.5rem",
                    }}
                >
                    <button
                        onClick={() => setMapTileType("osm")}
                        style={{
                            padding: "0.4rem 0.8rem",
                            background:
                                mapTileType === "osm" ? "#4ecca3" : "#333",
                            color: mapTileType === "osm" ? "#000" : "#fff",
                            border: "none",
                            borderRadius: "4px",
                            cursor: "pointer",
                            fontSize: "0.8rem",
                        }}
                    >
                        🗺️ OSM
                    </button>
                    <button
                        onClick={() => setMapTileType("openrailwaymap")}
                        style={{
                            padding: "0.4rem 0.8rem",
                            background:
                                mapTileType === "openrailwaymap"
                                    ? "#4ecca3"
                                    : "#333",
                            color:
                                mapTileType === "openrailwaymap"
                                    ? "#000"
                                    : "#fff",
                            border: "none",
                            borderRadius: "4px",
                            cursor: "pointer",
                            fontSize: "0.8rem",
                        }}
                    >
                        🚂 Railway
                    </button>
                </div>

                <MapContainer
                    center={[0, 0]}
                    zoom={3}
                    style={{ height: "100%", width: "100%" }}
                >
                    {/* Base OSM layer - always shown */}
                    <TileLayer
                        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    />

                    {/* OpenRailwayMap overlay layer */}
                    {mapTileType === "openrailwaymap" && (
                        <TileLayer
                            attribution='&copy; <a href="https://www.openrailwaymap.org/">OpenRailwayMap</a>'
                            url="https://{s}.tiles.openrailwaymap.org/standard/{z}/{x}/{y}.png"
                        />
                    )}

                    <MapClickHandler
                        onContextMenu={handleContextMenu}
                        onLeftClick={hideContextMenu}
                    />

                    {/* Start marker */}
                    {startPoint && (
                        <Marker
                            position={[startPoint.lat, startPoint.lng]}
                            icon={startIcon}
                        >
                            <Popup>Start Point</Popup>
                        </Marker>
                    )}

                    {/* End marker */}
                    {endPoint && (
                        <Marker
                            position={[endPoint.lat, endPoint.lng]}
                            icon={endIcon}
                        >
                            <Popup>End Point</Popup>
                        </Marker>
                    )}

                    {/* Waypoint markers */}
                    {waypoints.map((wp, i) => (
                        <Marker
                            key={`wp-${i}`}
                            position={[wp.lat, wp.lng]}
                            icon={waypointIcon}
                        >
                            <Popup>Waypoint {i + 1}</Popup>
                        </Marker>
                    ))}

                    {/* Tunnel markers and lines */}
                    {tunnels.map((t, i) => (
                        <div key={`tunnel-group-${i}`}>
                            <Marker
                                position={[t.entry.lat, t.entry.lng]}
                                icon={tunnelIcon}
                            >
                                <Popup>Tunnel {i + 1} Entry</Popup>
                            </Marker>
                            <Marker
                                position={[t.exit.lat, t.exit.lng]}
                                icon={tunnelIcon}
                            >
                                <Popup>Tunnel {i + 1} Exit</Popup>
                            </Marker>
                            <Polyline
                                positions={[
                                    [t.entry.lat, t.entry.lng],
                                    [t.exit.lat, t.exit.lng],
                                ]}
                                pathOptions={{
                                    color: "#9b59b6",
                                    weight: 3,
                                    dashArray: "10, 10",
                                    opacity: 0.7,
                                }}
                            />
                        </div>
                    ))}

                    {/* Pending tunnel entry */}
                    {pendingTunnelEntry && (
                        <CircleMarker
                            center={[
                                pendingTunnelEntry.lat,
                                pendingTunnelEntry.lng,
                            ]}
                            radius={10}
                            pathOptions={{
                                color: "#9b59b6",
                                fillColor: "#9b59b6",
                                fillOpacity: 0.5,
                            }}
                        />
                    )}

                    {/* Bridge markers and lines */}
                    {bridges.map((b, i) => (
                        <div key={`bridge-group-${i}`}>
                            <Marker
                                position={[b.start.lat, b.start.lng]}
                                icon={bridgeIcon}
                            >
                                <Popup>Bridge {i + 1} Start</Popup>
                            </Marker>
                            <Marker
                                position={[b.end.lat, b.end.lng]}
                                icon={bridgeIcon}
                            >
                                <Popup>Bridge {i + 1} End</Popup>
                            </Marker>
                            <Polyline
                                positions={[
                                    [b.start.lat, b.start.lng],
                                    [b.end.lat, b.end.lng],
                                ]}
                                pathOptions={{
                                    color: "#3498db",
                                    weight: 3,
                                    dashArray: "5, 5",
                                    opacity: 0.7,
                                }}
                            />
                        </div>
                    ))}

                    {/* Pending bridge start */}
                    {pendingBridgeStart && (
                        <CircleMarker
                            center={[
                                pendingBridgeStart.lat,
                                pendingBridgeStart.lng,
                            ]}
                            radius={10}
                            pathOptions={{
                                color: "#3498db",
                                fillColor: "#3498db",
                                fillOpacity: 0.5,
                            }}
                        />
                    )}

                    {/* Route polyline - interactive for hover */}
                    {routeCoordinates.length > 0 && (
                        <InteractivePolyline
                            positions={routeCoordinates}
                            onHover={handleRouteHover}
                            onMouseLeave={handleRouteMouseLeave}
                        />
                    )}

                    {/* Grid node shadows - visible on hover or when toggled */}
                    {(showGridNodes || hoverPosition) &&
                        gridNodes.map((node, i) => (
                            <CircleMarker
                                key={`grid-node-${i}`}
                                center={[node.lat, node.lng]}
                                radius={4}
                                pathOptions={{
                                    color: "rgba(255, 255, 255, 0.6)",
                                    fillColor: "rgba(100, 100, 100, 0.4)",
                                    fillOpacity: 0.4,
                                    weight: 1,
                                }}
                            >
                                <Tooltip
                                    direction="top"
                                    offset={[0, -5]}
                                    opacity={0.9}
                                >
                                    <div style={{ fontSize: "11px" }}>
                                        <strong>Node {i + 1}</strong>
                                        <br />
                                        Elevation: {node.elevation}m<br />
                                        Grid: [{node.row}, {node.col}]
                                    </div>
                                </Tooltip>
                            </CircleMarker>
                        ))}

                    {/* Elevation chart hover position marker */}
                    {hoverPosition && (
                        <CircleMarker
                            center={[hoverPosition.lat, hoverPosition.lng]}
                            radius={8}
                            pathOptions={{
                                color: "#ffffff",
                                fillColor: "#ff6b6b",
                                fillOpacity: 1,
                                weight: 3,
                            }}
                        >
                            <Tooltip
                                direction="top"
                                offset={[0, -10]}
                                opacity={0.95}
                                permanent
                            >
                                <div
                                    style={{
                                        textAlign: "center",
                                        fontSize: "12px",
                                    }}
                                >
                                    <strong>
                                        {Math.round(hoverPosition.elevation)}m
                                    </strong>
                                    <br />
                                    {(hoverPosition.distance / 1000).toFixed(
                                        2
                                    )}{" "}
                                    km
                                </div>
                            </Tooltip>
                        </CircleMarker>
                    )}

                    {/* Auto-detected tunnel segments */}
                    {routeData?.route_geojson?.properties?.auto_tunnels?.map(
                        (tunnel, i) => (
                            <Polyline
                                key={`auto-tunnel-${i}`}
                                positions={[
                                    [tunnel.start[1], tunnel.start[0]], // [lat, lng]
                                    [tunnel.end[1], tunnel.end[0]],
                                ]}
                                pathOptions={{
                                    color: "#9b59b6", // Purple for tunnels
                                    weight: 6,
                                    opacity: 0.9,
                                    dashArray: "10, 5",
                                }}
                            />
                        )
                    )}

                    {/* Auto-detected bridge segments */}
                    {routeData?.route_geojson?.properties?.auto_bridges?.map(
                        (bridge, i) => (
                            <Polyline
                                key={`auto-bridge-${i}`}
                                positions={[
                                    [bridge.start[1], bridge.start[0]], // [lat, lng]
                                    [bridge.end[1], bridge.end[0]],
                                ]}
                                pathOptions={{
                                    color: "#3498db", // Blue for bridges
                                    weight: 6,
                                    opacity: 0.9,
                                    dashArray: "5, 5",
                                }}
                            />
                        )
                    )}
                </MapContainer>

                {/* Context Menu */}
                {contextMenu.visible && (
                    <div
                        className="context-menu"
                        style={{
                            position: "fixed",
                            top: contextMenu.y,
                            left: contextMenu.x,
                            background: "#1e1e1e",
                            border: "1px solid #444",
                            borderRadius: "4px",
                            boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
                            zIndex: 10000,
                            minWidth: "180px",
                        }}
                    >
                        <div
                            className="context-menu-item"
                            onClick={() => handleContextMenuAction("set-start")}
                            style={{
                                padding: "0.6rem 1rem",
                                cursor: "pointer",
                                borderBottom: "1px solid #333",
                                color: "#4ecca3",
                            }}
                            onMouseEnter={(e) =>
                                (e.currentTarget.style.background = "#333")
                            }
                            onMouseLeave={(e) =>
                                (e.currentTarget.style.background =
                                    "transparent")
                            }
                        >
                            🟢 Set Start Point
                        </div>
                        <div
                            className="context-menu-item"
                            onClick={() => handleContextMenuAction("set-end")}
                            style={{
                                padding: "0.6rem 1rem",
                                cursor: "pointer",
                                borderBottom: "1px solid #333",
                                color: "#e74c3c",
                            }}
                            onMouseEnter={(e) =>
                                (e.currentTarget.style.background = "#333")
                            }
                            onMouseLeave={(e) =>
                                (e.currentTarget.style.background =
                                    "transparent")
                            }
                        >
                            🔴 Set End Point
                        </div>
                        <div
                            className="context-menu-item"
                            onClick={() =>
                                handleContextMenuAction("add-waypoint")
                            }
                            style={{
                                padding: "0.6rem 1rem",
                                cursor: "pointer",
                                borderBottom: "1px solid #333",
                                color: "#f39c12",
                            }}
                            onMouseEnter={(e) =>
                                (e.currentTarget.style.background = "#333")
                            }
                            onMouseLeave={(e) =>
                                (e.currentTarget.style.background =
                                    "transparent")
                            }
                        >
                            📍 Add Waypoint
                        </div>
                        <div
                            className="context-menu-item"
                            onClick={() =>
                                handleContextMenuAction(
                                    pendingTunnelEntry
                                        ? "add-tunnel-exit"
                                        : "add-tunnel-entry"
                                )
                            }
                            style={{
                                padding: "0.6rem 1rem",
                                cursor: "pointer",
                                borderBottom: "1px solid #333",
                                color: "#9b59b6",
                            }}
                            onMouseEnter={(e) =>
                                (e.currentTarget.style.background = "#333")
                            }
                            onMouseLeave={(e) =>
                                (e.currentTarget.style.background =
                                    "transparent")
                            }
                        >
                            🚇{" "}
                            {pendingTunnelEntry
                                ? "Set Tunnel Exit"
                                : "Add Tunnel Entry"}
                        </div>
                        <div
                            className="context-menu-item"
                            onClick={() =>
                                handleContextMenuAction(
                                    pendingBridgeStart
                                        ? "add-bridge-end"
                                        : "add-bridge-start"
                                )
                            }
                            style={{
                                padding: "0.6rem 1rem",
                                cursor: "pointer",
                                color: "#3498db",
                            }}
                            onMouseEnter={(e) =>
                                (e.currentTarget.style.background = "#333")
                            }
                            onMouseLeave={(e) =>
                                (e.currentTarget.style.background =
                                    "transparent")
                            }
                        >
                            🌉{" "}
                            {pendingBridgeStart
                                ? "Set Bridge End"
                                : "Add Bridge Start"}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;
