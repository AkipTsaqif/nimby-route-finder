# DEM Data Folder

Place your GeoTIFF elevation files here. The system will automatically index and use them.

## Supported Formats

-   `.tif` / `.tiff` files (GeoTIFF format)
-   Any CRS (will be reprojected to WGS84 if needed)
-   Single or multi-tile coverage

## Recommended Data Sources

### Indonesia (DEMNAS - 8m resolution) 🇮🇩

**Best quality for Indonesia!**

1. Go to: https://tanahair.indonesia.go.id/demnas/
2. Register for free account
3. Download tiles for your area of interest
4. Extract and place `.tif` files here

### Global (SRTM 30m)

1. Go to: https://dwtkns.com/srtm30m/
2. Click on the tile covering your area
3. Download the `.tif` file
4. Place it in this folder

### Global (Copernicus GLO-30)

1. Go to: https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model
2. Register for free Copernicus account
3. Download 30m resolution tiles
4. Extract and place here

### Japan/Asia (ALOS World 3D - 30m)

1. Go to: https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm
2. Register for free account
3. Download tiles
4. Extract and place here

## File Naming

You can use any filename. The system reads the geographic bounds from the GeoTIFF metadata.

Example structure:

```
data/dem/
├── DEMNAS_0809-14.tif     (Jakarta area)
├── DEMNAS_0809-15.tif     (Adjacent tile)
├── srtm_60_13.tif         (Alternative SRTM coverage)
└── README.md              (This file)
```

## After Adding Files

1. Restart the backend server, OR
2. The system will auto-detect on next request

## Configuration

Make sure to set in your environment:

```
DEMO_MODE=false
```

This enables real elevation data instead of synthetic terrain.

## Troubleshooting

**"rasterio not installed"**

```bash
pip install rasterio
```

On Windows, you may need to install from conda:

```bash
conda install -c conda-forge rasterio
```

**"No local DEM files found"**

-   Check that files have `.tif` or `.tiff` extension
-   Check file permissions
-   Check the server log for indexing errors

**Wrong elevation values**

-   Ensure the GeoTIFF has proper CRS metadata
-   Check for nodata values (the system handles -9999, etc.)
