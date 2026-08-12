# Event Data

## Data Source

The event component is a supplementary Sydney events dataset for 2019–2024. Records were manually curated from public event archives and official event websites to complement an existing major-events dataset. Included categories cover conferences, expos, festivals, music festivals, and entertainment events.

## Data Collection

The daily source records event names, start and end dates, representative locations and coordinates, event types, and internal source labels. Locations may use simplified central coordinates such as Sydney CBD, Bondi, or Darling Harbour.

## Processing

Events with explicit start and end times are expanded from the start hour through the end hour. When an exact hourly schedule is unavailable, each active date is expanded across hours 00:00–23:00. For Master Table integration, hourly data is preferred; non-zero minutes are rounded to the nearest hour, and only 2022–2024 rows with valid coordinates are retained.

## Spatial Matching

Active events are matched to traffic stations using Haversine distance between event and station coordinates. An event contributes station-level context when it is within 3 km of a station.

## Temporal Alignment

Event timestamps use day-first date parsing and are reduced to local `date + hour`. Station-event matches require both the same date-hour and the 3 km spatial condition.

## Output Used in This Benchmark

The station-hour event output contains `event_count_3km`, `has_event_3km`, and the distance, type, location, and name of the nearest matched event. It is integrated by `station_key + date + hour`.

## Limitations

The dataset is supplementary and partial, not a complete record of Sydney events. Some locations are representative rather than exact venue boundaries, and full-day expansion can overstate activity when exact schedules are unavailable. The 3 km radius is a consistent proximity rule, not a causal event-impact boundary.
