# Data Preparation

The benchmark combines traffic and crash records, weather conditions, event information, and static urban-context features. These components are prepared at their source granularities and aligned around Sydney traffic monitoring stations and hourly observations before benchmark construction.

## Data Components

| Component | Source | Main Processing | Output Used for Integration |
| --- | --- | --- | --- |
| Traffic / Crash | Transport for NSW traffic counts and NSW road crash data | Convert traffic counts to station-hour records, combine multiple traffic streams, and attach spatially and temporally aggregated crash context | Year-specific aligned traffic/crash tables keyed by `station_key`, `date`, and `hour` |
| Weather | Hourly regional weather files | Retain 2022–2024 local hourly observations, derive weather categories, and assign each station to its nearest available weather region | Hourly weather features joined through `weather_location`, `date`, and `hour` |
| Event | Manually curated supplementary Sydney event records | Expand event schedules to hourly records and match active events to stations by time and distance | Station-hour event counts and nearest-event attributes keyed by `station_key`, `date`, and `hour` |
| Urban Context | Transport for NSW station metadata and a local Sydney OpenStreetMap extract | Select permanent Sydney stations and aggregate POI, building, and land-use features within 500 m | Station-level urban-context table keyed by `station_key` |

## Overall Workflow

```text
Traffic / Crash
      +
Weather
      +
Events
      +
Urban Context
      ->
Spatial and temporal alignment
      ->
Master Table
      ->
Benchmark construction
```

## Spatial and Temporal Alignment

The components do not share a single native resolution. Traffic and processed crash context use station-hour keys. Weather remains region-level and is joined after each station is assigned to its nearest available weather region. Events are aligned to local hours and matched to stations within 3 km using coordinates. Urban-context features are static station-level attributes derived within 500 m buffers.

## Master Table

The processed components are combined into the station-hour Master Table using their documented keys and alignment rules. This documentation describes the inputs to that integration; it does not alter the existing Master Table construction workflow.

## Detailed Documentation

- [Traffic and Crash](./traffic/README.md)
- [Weather](./weather/README.md)
- [Events](./event/README.md)
- [Urban Context](./urban_context/README.md)
