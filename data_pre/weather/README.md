# Weather Data

## Data Source

The weather preparation reads individual regional CSV files from the local Open-Meteo collection. A malformed combined file, `nsw_all_regions_weather.csv`, is excluded. Only observations from 2022 through 2024 are retained.

## Data Collection

Each regional file provides hourly numeric weather observations where available, including temperature, apparent temperature, wet-bulb temperature, precipitation, rain, humidity, wind, cloud, sunshine, radiation, WMO weather code, snowfall, latitude, and longitude.

## Spatial Alignment

Each traffic station is assigned to exactly one nearest available weather region using Haversine distance between station and weather-region coordinates. The mapping joins `station_key` to `weather_location`; the documented mapping contains 295 stations across seven weather regions.

## Temporal Alignment

Weather timestamps are interpreted as local times. Timezone suffixes are removed before deriving local `date`, `hour`, and calendar fields. Weather is joined to traffic using `weather_location + date + hour`, rather than raw datetime, to reduce timezone mismatch risk.

## Processing

The processing retains the numeric observations and derives categorical features for rain intensity, temperature, humidity, wind, cloud, sunshine, and snow. It also creates a priority `weather_context_label` and a compositional `weather_combined_label`. A rain hour is identified when precipitation or rain is at least 0.2 mm.

## Output Used in This Benchmark

The processed output is `weather_hourly_features_2022_2024.csv`. It remains region-level and is integrated through the station-to-weather mapping and local hourly keys.
