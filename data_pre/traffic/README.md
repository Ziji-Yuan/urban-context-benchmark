# Traffic and Crash Data

## Data Source

Traffic data comes from the Transport for NSW Traffic Volume Counts API for the Sydney metropolitan area. The preparation covers 2022, 2023, and 2024 and includes permanent-station metadata and hourly traffic counts. Crash context comes from the NSW Road Crash Data 2020–2024 dataset, filtered to crashes from 2022–2024.

## Data Collection

The source workflow queries published traffic-station metadata and retrieves hourly permanent-station counts for each year. Wide hourly columns are converted to long records, and station spatial metadata is attached. The collection output includes `station_key`, local date and hour, traffic volume, and station attributes.

## Processing

Some stations report multiple streams for the same `station_key + date + hour`. Before crash features are attached, these records are combined by summing traffic volume and daily total so that the aligned output has one row per station-date-hour.

The crash source does not provide an exact calendar date. Crashes are therefore expanded from two-hour windows, spatially matched to stations within 5 km, and aggregated by `station_key + month + day_of_week + hour`. The resulting crash counts, severity, injury, fatality, and wet-surface summaries are left-joined to the hourly traffic records. These fields represent statistical crash context rather than confirmation of a crash on a specific date.

## Output Used in This Benchmark

The benchmark uses the aligned yearly files `traffic_hourly_sydney_{2022,2023,2024}_aligned.csv`. They contain station-hour traffic volume and five crash-context features and are integrated through `station_key`, `date`, and `hour`.

## Original Implementation

See [Gyuan-H/NSW_traffic_data](https://github.com/Gyuan-H/NSW_traffic_data) for the collection scripts, crash-alignment implementation, data dictionary, and full execution details.
