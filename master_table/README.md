# Master Table Construction

## Purpose

The Master Table is the shared spatio-temporal foundation of the Urban Context Benchmark. Its observation unit is a traffic monitoring station at a local calendar date and hour:

```text
station_key × date × hour
```

It places observed traffic volume and the contextual variables used by downstream benchmark tasks in one consistent representation.

## Inputs

The integration workflow uses the following source categories:

- **Traffic:** hourly volume and station identifiers, with direction or classification detail retained when present.
- **Station and urban context:** station metadata, coordinates, POI counts, building counts, land-use categories, and missing-data indicators.
- **Weather:** hourly regional temperature, precipitation, rain, humidity, wind, cloud, and derived weather categories.
- **Crashes:** station-hour crash counts, severity, injury, fatal, and wet-condition summaries.
- **Events:** supplementary event timing, type, name, and representative coordinates.
- **Calendar context:** year, month, day, day of week, weekend, public-holiday, and school-holiday fields where available.

## Integration Process

1. Traffic files are standardised to long-form hourly records. Wide `hour_00`–`hour_23` inputs are melted when necessary, and records are restricted to 2022–2024.
2. Crash features already aligned with hourly traffic records are aggregated and joined on `station_key + date + hour`.
3. Station metadata and urban-context features are joined on `station_key`.
4. Each station is assigned a weather region. Weather is then joined on `weather_location + date + hour`.
5. Event timestamps are aligned to local date and hour. Daily events are expanded across their active dates and hours when an hourly timestamp is unavailable.
6. Active events are matched to stations using Haversine distance, with a 3 km radius. Counts and nearest-event attributes are aggregated at station-hour level.
7. Calendar fields are standardised, and rows are filtered for benchmark usability without constructing task-specific traffic labels.

The integration preserves source missingness rather than replacing unavailable land-use categories or ratios with artificial values. Multiple traffic rows at the same station-hour remain distinct when direction or vehicle classification fields distinguish them.

## Output

The integrated table contains core keys, observed traffic volume, station metadata, static urban form, hourly weather, crash summaries, nearby-event summaries, and calendar context. A benchmark-ready subset retains rows with usable traffic, weather, station, and urban-context information.

The full table is not versioned here because the generated CSV is large and derived from external source data.

## Usage

The Master Table is an input to later ground-truth construction and benchmark-task generation. Task-specific code is responsible for defining baselines, pairing observations, assigning labels, sampling examples, and formatting questions; these operations are intentionally separated from the integration stage.
