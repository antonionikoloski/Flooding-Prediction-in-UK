# Flood Tool

See the [GiHub documentation](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#task-lists) for the task list syntax if you choose to use this.

## Description

Flood prediction tool using machine learning for UK [Environment Agency](https://www.gov.uk/government/organisations/environment-agency) data.

### Preproccessing
  No of datasets: 8
  
  Find more relevant data

  For given dataset:
    Removing duplicates/irrelevant data

    [
    Split into validation / training / test dataset
    Split into numerical / categorical data
    Impute missing values
    ]    (Maybe put this in prediction model pipeline?)

  
  Datasets:
    - [ ] postcode_missing_data.csv
    - [ ] postcodes_unlabelled.csv
    - [ ] typical_day.csv
    - [ ] wet_day.csv
    - [ ] district_data.csv
    - [ ] postcodes_labelled.csv
    - [ ] sector_data.csv
    - [ ] stations.csv
    - [ ] ...

### Flood Risk Tool
  - [X] geo.py
  - [ ] Fit
  - [ ] Lookup Easting/Northing
  - [ ] Lookup Lat/Long
  - [ ] Impute

#### Prediction Models
- [ ] Create models
  - [X] Zero risk
  - [ ] Predict flood class from postcode
  - [ ] Predict flood class from osgb36
  - [ ] Predict flood class from wgs1984
  - [ ] Predict median house price
  - [ ] Predict local authority
  - [ ] Predict historic flooding

  - [ ] Estimate total value
  - [ ] Estimate annual human flood risk
  - [ ] Estimate annual flood economic risk


#### Visualisation
- [ ] Analyse files
- [ ] Display data
- [ ] Access live files
  - [X] Get data for a date


#### Coordinate translation
- [X] Convert from lat/long to OSGB36

functions/methods in the geo.py module:
  dms2rad(deg, min=0, sec=0)
  rad2dms(rad, dms=False)
  dms2deg(deg, min, sec)
  lat_long_to_xyz(phi, lam, rads=False, datum=osgb36)
  xyz_to_lat_long(x, y, z, rads=False, datum=osgb36)
  get_easting_northing_from_gps_lat_long(phi, lam, rads=False, dtype=float64)
  get_gps_lat_long_from_easting_northing(east, north, rads=False, dms=False, dtype=float64)
  WGS84toOSGB36(phi, lam, rads=False)
  OSGB36toWGS84(phi, lam, rads=False)euclidean_distance(easting1, northing1, easting2, northing2)
  haversine_distance(lat1, long1, lat2, long2, deg=True, dms=False)