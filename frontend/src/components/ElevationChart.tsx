import { useCallback } from "react";
import {
	AreaChart,
	Area,
	XAxis,
	YAxis,
	Tooltip,
	ResponsiveContainer,
	ReferenceLine,
} from "recharts";

interface ElevationProfilePoint {
	distance: number;
	elevation: number;
	lat: number;
	lng: number;
}

interface ElevationChartProps {
	elevationProfile: ElevationProfilePoint[];
	onHover?: (point: ElevationProfilePoint | null) => void;
}

export default function ElevationChart({
	elevationProfile,
	onHover,
}: ElevationChartProps) {
	// Format distance for display
	const formatDistance = (meters: number): string => {
		if (meters >= 1000) {
			return `${(meters / 1000).toFixed(1)} km`;
		}
		return `${Math.round(meters)} m`;
	};

	const data = elevationProfile.map((point) => ({
		...point,
		distanceKm: point.distance / 1000,
	}));

	const minElev = Math.min(...elevationProfile.map((p) => p.elevation));
	const maxElev = Math.max(...elevationProfile.map((p) => p.elevation));
	const elevRange = maxElev - minElev;
	const totalDistance =
		elevationProfile[elevationProfile.length - 1]?.distance || 0;

	// Calculate elevation gain/loss
	let gain = 0;
	let loss = 0;
	for (let i = 1; i < elevationProfile.length; i++) {
		const diff =
			elevationProfile[i].elevation - elevationProfile[i - 1].elevation;
		if (diff > 0) gain += diff;
		else loss += Math.abs(diff);
	}

	const handleMouseMove = useCallback(
		(e: { activePayload?: Array<{ payload: ElevationProfilePoint }> }) => {
			if (e.activePayload && e.activePayload.length > 0 && onHover) {
				onHover(e.activePayload[0].payload);
			}
		},
		[onHover]
	);

	const handleMouseLeave = useCallback(() => {
		if (onHover) {
			onHover(null);
		}
	}, [onHover]);

	return (
		<div className="elevation-chart">
			<div
				style={{
					display: "flex",
					justifyContent: "space-between",
					alignItems: "center",
					marginBottom: "0.5rem",
				}}
			>
				<h3 style={{ margin: 0 }}>Elevation Profile</h3>
				<div style={{ fontSize: "0.75rem", color: "#888" }}>
					<span style={{ color: "#4ecca3", marginRight: "1rem" }}>
						↑ {Math.round(gain)}m
					</span>
					<span style={{ color: "#e74c3c" }}>
						↓ {Math.round(loss)}m
					</span>
				</div>
			</div>
			<div
				style={{
					display: "flex",
					justifyContent: "space-between",
					fontSize: "0.7rem",
					color: "#666",
					marginBottom: "0.25rem",
				}}
			>
				<span>Total: {formatDistance(totalDistance)}</span>
				<span>
					Min: {Math.round(minElev)}m | Max: {Math.round(maxElev)}m
				</span>
			</div>
			<ResponsiveContainer width="100%" height={140}>
				<AreaChart
					data={data}
					margin={{ top: 5, right: 10, left: -10, bottom: 5 }}
					onMouseMove={handleMouseMove}
					onMouseLeave={handleMouseLeave}
				>
					<defs>
						<linearGradient
							id="elevGradient"
							x1="0"
							y1="0"
							x2="0"
							y2="1"
						>
							<stop
								offset="5%"
								stopColor="#4ecca3"
								stopOpacity={0.3}
							/>
							<stop
								offset="95%"
								stopColor="#4ecca3"
								stopOpacity={0}
							/>
						</linearGradient>
					</defs>
					<XAxis
						dataKey="distanceKm"
						tick={{ fontSize: 9, fill: "#666" }}
						axisLine={{ stroke: "#333" }}
						tickFormatter={(v) => `${v.toFixed(1)}`}
						label={{
							value: "km",
							position: "insideBottomRight",
							offset: -5,
							fontSize: 9,
							fill: "#666",
						}}
					/>
					<YAxis
						domain={[
							Math.floor(minElev - elevRange * 0.1),
							Math.ceil(maxElev + elevRange * 0.1),
						]}
						tick={{ fontSize: 9, fill: "#666" }}
						axisLine={{ stroke: "#333" }}
						tickFormatter={(v) => `${v}m`}
						width={45}
					/>
					<Tooltip
						contentStyle={{
							background: "#16213e",
							border: "1px solid #4ecca3",
							borderRadius: "4px",
							fontSize: "0.8rem",
						}}
						labelStyle={{ color: "#888" }}
						formatter={(value: number) => [
							`${Math.round(value)}m`,
							"Elevation",
						]}
						labelFormatter={(label) =>
							`Distance: ${formatDistance(label * 1000)}`
						}
					/>
					<ReferenceLine y={0} stroke="#444" strokeDasharray="3 3" />
					<Area
						type="monotone"
						dataKey="elevation"
						stroke="#4ecca3"
						strokeWidth={2}
						fill="url(#elevGradient)"
						dot={false}
						activeDot={{
							r: 5,
							fill: "#4ecca3",
							stroke: "#fff",
							strokeWidth: 2,
						}}
					/>
				</AreaChart>
			</ResponsiveContainer>
			<div
				style={{
					textAlign: "center",
					fontSize: "0.65rem",
					color: "#555",
					marginTop: "0.25rem",
				}}
			>
				Hover over the chart to see position on map
			</div>
		</div>
	);
}
