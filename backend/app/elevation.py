"""
Elevation Data Service

Fetches elevation data from various sources (in priority order):
1. Local GeoTIFF files (fastest, highest quality)
2. Mapbox Terrain-RGB tiles (requires API key)
3. Synthetic demo terrain (fallback)
"""

import os
import math
import asyncio
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import httpx

# Try to import rasterio for local GeoTIFF support
try:
    import rasterio
    from rasterio.merge import merge
    from rasterio.windows import from_bounds
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[Elevation] rasterio not installed - local GeoTIFF support disabled")
    print("[Elevation] Install with: pip install rasterio")


class ElevationService:
    """
    Service for fetching and managing elevation data.
    
    Priority order:
    1. Local GeoTIFF files in ./data/dem/ folder
    2. Mapbox Terrain-RGB tiles (if MAPBOX_ACCESS_TOKEN is set)
    3. Synthetic demo terrain (if DEMO_MODE=true or no other source available)
    
    To use local DEMs:
    1. Create folder: backend/data/dem/
    2. Add .tif files (SRTM, DEMNAS, ALOS, etc.)
    3. Set DEMO_MODE=false
    """
    
    def __init__(self, dem_folder: str = None):
        self.mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN")
        self.use_demo_mode = os.getenv("DEMO_MODE", "true").lower() == "true"
        
        # Set up DEM folder path
        if dem_folder:
            self.dem_folder = Path(dem_folder)
        else:
            # Default: ./data/dem relative to backend folder
            backend_dir = Path(__file__).parent.parent
            self.dem_folder = backend_dir / "data" / "dem"
        
        # Create folder if it doesn't exist
        self.dem_folder.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded rasters and tiles
        self.tile_cache: Dict[str, np.ndarray] = {}
        self.raster_index: List[dict] = []  # Index of available local DEMs
        
        # Build index of local DEM files
        self._index_local_dems()
        
        # Print status
        self._print_status()
    
    def _print_status(self):
        """Print current configuration status."""
        print(f"[Elevation] DEM folder: {self.dem_folder}")
        print(f"[Elevation] Demo mode: {self.use_demo_mode}")
        print(f"[Elevation] Mapbox token: {'set' if self.mapbox_token else 'not set'}")
        print(f"[Elevation] Rasterio available: {HAS_RASTERIO}")
        print(f"[Elevation] Local DEMs indexed: {len(self.raster_index)}")
    
    def _index_local_dems(self):
        """Scan the DEM folder and build an index of available tiles."""
        self.raster_index = []
        
        if not HAS_RASTERIO:
            return
        
        if not self.dem_folder.exists():
            return
        
        # Find all .tif files
        tif_patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
        tif_files = []
        for pattern in tif_patterns:
            tif_files.extend(self.dem_folder.glob(pattern))
            tif_files.extend(self.dem_folder.glob(f"**/{pattern}"))  # Recursive
        
        for tif_path in tif_files:
            try:
                with rasterio.open(tif_path) as src:
                    # Get bounds in WGS84 (EPSG:4326)
                    if src.crs and src.crs != CRS.from_epsg(4326):
                        # Transform bounds to WGS84
                        from rasterio.warp import transform_bounds
                        bounds = transform_bounds(src.crs, CRS.from_epsg(4326), *src.bounds)
                    else:
                        bounds = src.bounds
                    
                    # bounds = (left, bottom, right, top) = (min_lng, min_lat, max_lng, max_lat)
                    self.raster_index.append({
                        "path": str(tif_path),
                        "bounds": bounds,
                        "resolution": src.res,  # (x_res, y_res) in CRS units
                        "crs": src.crs,
                        "shape": src.shape,
                        "nodata": src.nodata
                    })
                    print(f"[Elevation] Indexed: {tif_path.name} "
                          f"lat: {bounds[1]:.4f} to {bounds[3]:.4f}, "
                          f"lng: {bounds[0]:.4f} to {bounds[2]:.4f}")
            except Exception as e:
                print(f"[Elevation] Warning: Could not read {tif_path}: {e}")
        
        if self.raster_index:
            print(f"[Elevation] Found {len(self.raster_index)} local DEM file(s)")
    
    def _find_covering_dems(
        self, 
        min_lat: float, min_lng: float, 
        max_lat: float, max_lng: float
    ) -> List[dict]:
        """Find local DEM files that cover (or partially cover) the given bounds."""
        covering = []
        for dem in self.raster_index:
            dem_min_lng, dem_min_lat, dem_max_lng, dem_max_lat = dem["bounds"]
            
            # Check for overlap
            if (min_lng <= dem_max_lng and max_lng >= dem_min_lng and
                min_lat <= dem_max_lat and max_lat >= dem_min_lat):
                covering.append(dem)
        
        return covering
    
    async def get_elevation_grid(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        downsampling_factor: int = 1,
        padding_factor: float = 0.2
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float, float]], Optional[Dict]]:
        """
        Get an elevation grid covering the area between start and end points.
        
        Args:
            start: (lat, lng) of start point
            end: (lat, lng) of end point
            downsampling_factor: 1 = native (~30m), 2 = half res (~60m), etc.
            padding_factor: Extra padding around the bounding box (0.2 = 20%)
        
        Returns:
            elevation_grid: 2D numpy array of elevations in meters
            bounds: (min_lat, min_lng, max_lat, max_lng)
            transform: Grid transform parameters including 'source' used
        """
        # Calculate bounding box with padding
        lat1, lng1 = start
        lat2, lng2 = end
        
        min_lat = min(lat1, lat2)
        max_lat = max(lat1, lat2)
        min_lng = min(lng1, lng2)
        max_lng = max(lng1, lng2)
        
        lat_range = max_lat - min_lat
        lng_range = max_lng - min_lng
        
        # Add padding
        min_lat -= lat_range * padding_factor
        max_lat += lat_range * padding_factor
        min_lng -= lng_range * padding_factor
        max_lng += lng_range * padding_factor
        
        # Calculate dimensions in meters
        lat_center = (min_lat + max_lat) / 2
        meters_per_deg_lat = 111320
        meters_per_deg_lng = 111320 * math.cos(math.radians(lat_center))
        
        height_m = (max_lat - min_lat) * meters_per_deg_lat
        width_m = (max_lng - min_lng) * meters_per_deg_lng
        
        # Ensure minimum area for very close points
        min_dimension_m = 1000
        if height_m < min_dimension_m:
            extra_lat = (min_dimension_m - height_m) / meters_per_deg_lat / 2
            min_lat -= extra_lat
            max_lat += extra_lat
            height_m = min_dimension_m
        
        if width_m < min_dimension_m:
            extra_lng = (min_dimension_m - width_m) / meters_per_deg_lng / 2
            min_lng -= extra_lng
            max_lng += extra_lng
            width_m = min_dimension_m
        
        bounds = (min_lat, min_lng, max_lat, max_lng)
        
        # Try sources in priority order
        elevation_grid = None
        source_used = "none"
        native_resolution = 30.0
        
        # 1. Try local GeoTIFF files (highest priority)
        if HAS_RASTERIO and self.raster_index and not self.use_demo_mode:
            covering_dems = self._find_covering_dems(min_lat, min_lng, max_lat, max_lng)
            if covering_dems:
                print(f"[Elevation] Found {len(covering_dems)} local DEM(s) covering area")
                elevation_grid, native_resolution = self._load_from_geotiff(
                    bounds, covering_dems, downsampling_factor
                )
                if elevation_grid is not None:
                    source_used = "local_geotiff"
        
        # 2. Try Mapbox Terrain-RGB
        if elevation_grid is None and self.mapbox_token and not self.use_demo_mode:
            print("[Elevation] Trying Mapbox Terrain-RGB...")
            elevation_grid = await self._fetch_mapbox_elevation(bounds, downsampling_factor)
            if elevation_grid is not None:
                source_used = "mapbox"
                native_resolution = 30.0
        
        # 3. Fall back to demo terrain
        if elevation_grid is None:
            if not self.use_demo_mode:
                print("[Elevation] No elevation source available, using demo terrain")
            else:
                print("[Elevation] Using synthetic demo terrain")
            
            base_resolution = 30.0
            cell_size_m = base_resolution * downsampling_factor
            rows = max(20, int(height_m / cell_size_m))
            cols = max(20, int(width_m / cell_size_m))
            rows = min(rows, 500)
            cols = min(cols, 500)
            
            elevation_grid = self._generate_demo_elevation(bounds, rows, cols)
            source_used = "demo"
            native_resolution = base_resolution
        
        rows, cols = elevation_grid.shape
        actual_cell_size_m = max(height_m / rows, width_m / cols)
        
        transform = {
            "rows": rows,
            "cols": cols,
            "cell_size_m": actual_cell_size_m,
            "downsampling_factor": downsampling_factor,
            "source": source_used,
            "native_resolution_m": native_resolution
        }
        
        print(f"[Elevation] Grid: {rows}x{cols}, cell ~{actual_cell_size_m:.1f}m, source={source_used}")
        print(f"[Elevation] Elevation range: {elevation_grid.min():.1f}m - {elevation_grid.max():.1f}m")
        
        return elevation_grid, bounds, transform
    
    def _load_from_geotiff(
        self,
        bounds: Tuple[float, float, float, float],
        dems: List[dict],
        downsampling_factor: int
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Load elevation data from local GeoTIFF file(s).
        
        Returns:
            (elevation_grid, native_resolution_m) or (None, 0) on failure
        """
        min_lat, min_lng, max_lat, max_lng = bounds
        
        try:
            if len(dems) == 1:
                # Single file - read directly with windowing
                dem = dems[0]
                with rasterio.open(dem["path"]) as src:
                    # Calculate window in pixel coordinates
                    # Note: bounds for from_bounds are (left, bottom, right, top) = (min_lng, min_lat, max_lng, max_lat)
                    try:
                        window = from_bounds(
                            min_lng, min_lat, max_lng, max_lat,
                            src.transform
                        )
                    except Exception as e:
                        print(f"[Elevation] Window calculation failed: {e}")
                        return None, 0
                    
                    # Read the data
                    data = src.read(
                        1, 
                        window=window,
                        out_shape=(
                            max(1, int(window.height) // downsampling_factor),
                            max(1, int(window.width) // downsampling_factor)
                        ),
                        resampling=Resampling.bilinear
                    )
                    
                    # Handle nodata values
                    if src.nodata is not None:
                        data = np.where(data == src.nodata, np.nan, data)
                    
                    # Also handle common nodata values
                    data = np.where(data < -1000, np.nan, data)  # Common nodata sentinel
                    data = np.where(data > 9000, np.nan, data)   # Higher than Everest
                    
                    # Fill NaN with interpolation or nearest valid
                    if np.any(np.isnan(data)):
                        from scipy.ndimage import generic_filter
                        def nanmean_filter(values):
                            valid = values[~np.isnan(values)]
                            return np.mean(valid) if len(valid) > 0 else 0
                        data = generic_filter(data, nanmean_filter, size=3)
                    
                    # Calculate native resolution in meters
                    # src.res gives (x_res, y_res) in CRS units (degrees for EPSG:4326)
                    lat_center = (min_lat + max_lat) / 2
                    if src.crs and src.crs.is_geographic:
                        native_res_m = abs(src.res[0]) * 111320 * math.cos(math.radians(lat_center))
                    else:
                        native_res_m = abs(src.res[0])  # Already in meters
                    
                    print(f"[Elevation] Loaded from {Path(dem['path']).name}: {data.shape}")
                    return data.astype(np.float32), native_res_m
            
            else:
                # Multiple files - need to merge them
                datasets = []
                for d in dems:
                    datasets.append(rasterio.open(d["path"]))
                
                # Merge the datasets
                mosaic, out_transform = merge(
                    datasets,
                    bounds=(min_lng, min_lat, max_lng, max_lat),
                    resampling=Resampling.bilinear
                )
                
                # Close datasets
                for ds in datasets:
                    ds.close()
                
                data = mosaic[0]  # First band
                
                # Apply downsampling if needed
                if downsampling_factor > 1 and data.size > 100:
                    try:
                        from scipy.ndimage import zoom
                        data = zoom(data, 1.0 / downsampling_factor, order=1)
                    except ImportError:
                        # Manual downsampling
                        step = downsampling_factor
                        data = data[::step, ::step]
                
                # Handle nodata
                data = np.where(data < -1000, np.nan, data)
                data = np.where(data > 9000, np.nan, data)
                
                native_res_m = abs(dems[0]["resolution"][0]) * 111320
                
                print(f"[Elevation] Merged {len(dems)} DEMs: {data.shape}")
                return data.astype(np.float32), native_res_m
                
        except Exception as e:
            print(f"[Elevation] Error loading GeoTIFF: {e}")
            import traceback
            traceback.print_exc()
            return None, 0
    
    async def _fetch_mapbox_elevation(
        self,
        bounds: Tuple[float, float, float, float],
        downsampling_factor: int
    ) -> Optional[np.ndarray]:
        """
        Fetch elevation from Mapbox Terrain-RGB tiles.
        
        Terrain-RGB encoding: height = -10000 + ((R * 256 * 256 + G * 256 + B) * 0.1)
        """
        if not self.mapbox_token:
            return None
        
        min_lat, min_lng, max_lat, max_lng = bounds
        
        # Calculate grid size
        lat_center = (min_lat + max_lat) / 2
        meters_per_deg_lat = 111320
        meters_per_deg_lng = 111320 * math.cos(math.radians(lat_center))
        
        height_m = (max_lat - min_lat) * meters_per_deg_lat
        width_m = (max_lng - min_lng) * meters_per_deg_lng
        
        # Base resolution ~30m, apply downsampling
        # Allow larger grids for better resolution (up to 400x400)
        cell_size_m = 30.0 * downsampling_factor
        max_grid_size = 400  # Increased from 200 for better resolution
        rows = max(20, min(max_grid_size, int(height_m / cell_size_m)))
        cols = max(20, min(max_grid_size, int(width_m / cell_size_m)))
        
        # Determine appropriate zoom level (higher = more detail)
        zoom = max(8, min(14, 14 - int(math.log2(max(1, downsampling_factor)))))
        
        print(f"[Mapbox] Fetching {rows}x{cols} grid at zoom {zoom}...")
        
        elevation_grid = np.zeros((rows, cols), dtype=np.float32)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for i in range(rows):
                for j in range(cols):
                    lat = max_lat - (i / (rows - 1)) * (max_lat - min_lat) if rows > 1 else lat_center
                    lng = min_lng + (j / (cols - 1)) * (max_lng - min_lng) if cols > 1 else (min_lng + max_lng) / 2
                    
                    elev = await self._get_mapbox_elevation_at_point(client, lat, lng, zoom)
                    elevation_grid[i, j] = elev
                
                # Progress for large grids
                if rows > 50 and i % 25 == 0:
                    print(f"[Mapbox] Progress: {i}/{rows} rows...")
        
        print(f"[Mapbox] Fetched elevation grid")
        return elevation_grid
    
    async def _get_mapbox_elevation_at_point(
        self,
        client: httpx.AsyncClient,
        lat: float,
        lng: float,
        zoom: int
    ) -> float:
        """Get elevation at a single point from Mapbox Terrain-RGB"""
        # Convert lat/lng to tile coordinates
        n = 2 ** zoom
        x_tile = int((lng + 180) / 360 * n)
        lat_rad = math.radians(lat)
        y_tile = int((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * n)
        
        # Calculate pixel position within tile
        x_pixel = int(((lng + 180) / 360 * n - x_tile) * 256)
        y_pixel = int(((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * n - y_tile) * 256)
        
        cache_key = f"mapbox_{zoom}_{x_tile}_{y_tile}"
        
        if cache_key not in self.tile_cache:
            url = f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{zoom}/{x_tile}/{y_tile}.pngraw?access_token={self.mapbox_token}"
            
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(response.content))
                    self.tile_cache[cache_key] = np.array(img)
                else:
                    print(f"[Mapbox] Tile fetch error: {response.status_code}")
                    return 0.0
            except Exception as e:
                print(f"[Mapbox] Error: {e}")
                return 0.0
        
        tile_data = self.tile_cache[cache_key]
        
        # Clamp pixel coordinates
        x_pixel = max(0, min(x_pixel, 255))
        y_pixel = max(0, min(y_pixel, 255))
        
        # Decode Terrain-RGB (convert to int to avoid uint8 overflow)
        r = int(tile_data[y_pixel, x_pixel, 0])
        g = int(tile_data[y_pixel, x_pixel, 1])
        b = int(tile_data[y_pixel, x_pixel, 2])
        height = -10000 + ((r * 256 * 256 + g * 256 + b) * 0.1)
        
        return height
    
    def _generate_demo_elevation(
        self,
        bounds: Tuple[float, float, float, float],
        rows: int,
        cols: int
    ) -> np.ndarray:
        """
        Generate synthetic elevation data for demo/testing.
        Creates GENTLE terrain suitable for railway constraints.
        """
        min_lat, min_lng, max_lat, max_lng = bounds
        
        # Use lat/lng to seed randomness for consistency
        seed = int(abs(min_lat * 1000 + min_lng * 100)) % 100000
        np.random.seed(seed)
        
        # Create coordinate grids
        y = np.linspace(0, 1, rows)
        x = np.linspace(0, 1, cols)
        xx, yy = np.meshgrid(x, y)
        
        # Calculate cell size for slope calibration
        lat_center = (min_lat + max_lat) / 2
        meters_per_deg_lat = 111320
        meters_per_deg_lng = 111320 * math.cos(math.radians(lat_center))
        height_m = (max_lat - min_lat) * meters_per_deg_lat
        width_m = (max_lng - min_lng) * meters_per_deg_lng
        cell_size_m = max(height_m / rows, width_m / cols) if rows > 0 and cols > 0 else 50
        
        # For gentle slopes (~1-2%)
        max_elev_change_per_cell = cell_size_m * 0.015
        total_rise = min(rows, cols) * max_elev_change_per_cell * 0.3
        
        # Base elevation with gentle tilt
        base = 50 + total_rise * yy * 0.5 + total_rise * xx * 0.3
        
        # Gentle rolling hills
        wave_scale = 0.15
        hills = (
            total_rise * wave_scale * np.sin(xx * 2 * np.pi * 1.5) * np.cos(yy * 2 * np.pi * 1.2) +
            total_rise * wave_scale * 0.5 * np.sin(xx * 2 * np.pi * 0.7 + 1) * np.sin(yy * 2 * np.pi * 0.9 + 0.5)
        )
        
        # Small noise
        noise = np.random.randn(rows, cols) * (max_elev_change_per_cell * 0.1)
        try:
            from scipy.ndimage import gaussian_filter
            noise = gaussian_filter(noise, sigma=2)
        except ImportError:
            noise = noise * 0.5
        
        elevation = base + hills + noise
        elevation = np.maximum(elevation, 0)
        
        return elevation.astype(np.float32)
    
    async def get_elevation_at_point(self, lat: float, lng: float) -> Optional[float]:
        """Get elevation at a single point."""
        # Try local DEMs first
        if HAS_RASTERIO and self.raster_index and not self.use_demo_mode:
            covering = self._find_covering_dems(lat, lng, lat, lng)
            if covering:
                try:
                    with rasterio.open(covering[0]["path"]) as src:
                        for val in src.sample([(lng, lat)]):
                            return float(val[0])
                except Exception:
                    pass
        
        # Try Mapbox
        if self.mapbox_token and not self.use_demo_mode:
            async with httpx.AsyncClient(timeout=10.0) as client:
                return await self._get_mapbox_elevation_at_point(client, lat, lng, 14)
        
        # Demo fallback
        return 100 + 50 * math.sin(lat * 10) + 30 * math.cos(lng * 10)
    
    def refresh_dem_index(self):
        """Re-scan the DEM folder for new files."""
        print("[Elevation] Refreshing DEM index...")
        self._index_local_dems()
        print(f"[Elevation] Found {len(self.raster_index)} DEM file(s)")
