# Urban Context Data

## Data Sources

Urban context is derived from the Transport for NSW traffic-station reference table and a local `Sydney.osm.pbf` OpenStreetMap extract. The station table supplies identifiers, metadata, and WGS84 coordinates; the OSM extract supplies POI, building, and land-use geometries.

## Station Selection

Stations are retained when their RMS region is Sydney, they are permanent monitoring stations, they are approved for publication, and their quality rating is at least 4. This produces 295 station records.

## Spatial Processing

Station coordinates are loaded in `EPSG:4326` and reprojected to `EPSG:7856` for metre-based processing. A 500 m buffer around each station is used as the common spatial unit for feature extraction.

## POI Features

POIs are extracted from OSM point and multipolygon layers using amenity, shop, public transport, leisure, tourism, and healthcare tags. Polygon POIs are represented by points, likely duplicate named point/polygon features within 20 m are removed, and category counts are aggregated within each station buffer.

## Building Features

OSM polygons with building tags are matched to the buffers. Unique buildings are counted per station, and building density is calculated from the 500 m buffer area.

## Land-use Features

OSM land-use polygons are mapped to six broader categories, intersected with station buffers, and summarized as percentages of mapped land-use area.

## Final Output

`POIs_output/sydney_station_urban_context_500m.csv` is the modelling-ready output. Each of its 295 rows contains station metadata, seven POI counts, building count and density, six land-use percentages, and missing-data flags. A GeoPackage version preserves station geometry.

## Integration Key

`station_key` is the primary identifier used to join the station-level urban-context features to traffic observations and the benchmark Master Table.

## Original Implementation

See [Ziji-Yuan/Sydney-Urban-Context-Data](https://github.com/Ziji-Yuan/Sydney-Urban-Context-Data) for the notebooks, repository structure, data dictionary, and full spatial-processing details.
