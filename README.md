# Flood Risk Prediction tool

## Deadlines
-  **Code: 12pm GMT Friday 24th November 2023**
-  **Presentation and one-page report: 4pm GMT Friday 24th November 2023**

You should update this document in the course of your work to reflect the scope and abilities of your project, as well as to provide appropriate instuctions to potential users (and markers) of your code.

### Key Requirements

Your project must provide the following:

 1. at least one analysis method to estimate a number of attributes for unlabelled postcodes extrapolated from sample data which is provided to you:
    - Flood risk from rivers & seas (on a 10 point scale).
    - Flood risk from surface water (on a 10 point scale).
    - Median house price.
 2. at least one analysis method to estimate the Local Authority & flood risks of arbitrary locations. 
 3. a process to find the rainfall and water level near a given postcode from provided rainfall, river and tide level data, or by looking online.
 4. visualization and analysis tools for the postcode, rainfall, river & tide data provided to you, ideally in a way which will identify potential areas at immediate risk of flooding by combining the data you have been given.
 
 Your code should have installation instructions and basic documentation, as docstrings for functions & class methods, as a full manual or ideally both.

![London postcode density](images/LondonPostcodeDensity.png)
![England Flood Risk](images/EnglandFloodRisk.png)
![UK soil types](images/UKSoilTypes.png)

This README file *should be updated* over the course of your group's work to represent the scope and abilities of your project.

### Assessment

 - Your code will be assessed for its speed (both at training and prediction) & predictive accuracy.
 - Your code should include tests of its functionality.
 - Additional marks will be awarded for maintaining good code quality and for a clean, well-organised repository. You should consider the kind of code you would be happy to take over from another group of students.

### AI usage

*To be written by you during the week explaining how your group used AI tools in developing your code*

# Example Module in Template Package

This module provides tools for working with postcode databases, including methods for flood classification, median house price prediction, geographic data imputation, and more. The main class, `Tool`, allows for data preprocessing, model fitting, and prediction based on different methods for flood risk classification, median house price estimation, and geographic data lookup.

---

## **Features**

- **Data Preprocessing**: Handle missing values, encode categorical variables, and scale numerical features.
- **Flood Classification**: Predict flood risk using various methods like Logistic Regression and Random Forest Classifier.
- **House Price Prediction**: Estimate median house prices for given postcodes using pre-defined methods.
- **Geographic Data Lookup**: Retrieve geographic data (easting, northing, latitude, longitude) for specified postcodes or locations.
- **Imputation**: Handle missing data using various strategies such as mean, most frequent, constant, and KNN imputation.
- **Model Training**: Fit and tune models for flood classification, house price prediction, and more.
- **Utility Methods**: Access additional data and perform tasks like historic flooding predictions and local authority predictions.

---

## **Requirements**

- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib

---

## **Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/
   
   ```

2. **Set Up Virtual Deluge Environment**:
   ```bash
   conda env create -f deluge.yml

   conda activate deluge

   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**:
   ```bash
   python -m pytest --doctest-modules flood_tool
   ```

---

## **Usage**

### 1. Initialize the Tool
```python
from flood_tool import Tool

tool = Tool(
    labelled_unit_data,
    unlabelled_unit_data,
    sector_data,
    district_data}
)
```

### 2. Predict Flood Risk
```python
flood_classes = tool.predict_flood_class_from_postcode(
    ["SW1A 1AA", "M1 2AB"], method="Logistic_Regression_flood"
)
print(flood_classes)
```

### 3. Impute Missing Values
```python
imputed_data = tool.impute_missing_values(dataframe, method="mean")
```

### 4. Geographic Data Lookup
```python
easting_northing = tool.lookup_easting_northing(['M34 7QL'])
lat_long = tool.lookup_lat_long(['M34 7QL'])
```

### 5. Predict Median House Prices
```python
house_prices = tool.predict_median_house_price(['M34 7QL'])
```

---

## **Data Files**

The `Tool` class expects the following data files:
- `postcodes_labelled.csv`: Contains labeled postcode data with flood risk labels and other features.
- `postcodes_unlabelled.csv`: Contains unlabeled postcode data.
- `sector_data.csv`: Data related to postcode sectors.
- `district_data.csv`: Data related to postcode districts.

Default files are provided with the package, or you can provide your own.

---

## **Documentation**

### Build HTML Documentation:
```bash
python -m sphinx docs html
```
Open the `index.html` file in the `html` directory with a browser.

### Build PDF Documentation (Requires LaTeX):
```bash
python -m sphinx -b latex docs latex
```
Follow the instructions in the `latex` directory to process the `FloodTool.tex` file.

---

## **Testing**

Run the included tests to verify the tool's functionality:
```bash
python -m pytest --doctest-modules flood_tool
```

---

### Reading list (this can be updated as you go)

 - [A guide to coordinate systems in Great Britain](https://webarchive.nationalarchives.gov.uk/20081023180830/http://www.ordnancesurvey.co.uk/oswebsite/gps/information/coordinatesystemsinfo/guidecontents/index.html)

 - [Information on postcode validity](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/283357/ILRSpecification2013_14Appendix_C_Dec2012_v1.pdf)


---

### visualization
Our visualisation uses notebooks to call wrapped functions for data processing and mapping. You can find the documentation for running our visualisations inside `flood_tool/map.ipynb`.
We plotted risk_map, historical_flood_map, postcode_household_map, houseprice_map, potential_map, and folium interactions about rainfall, river, tides, and risk, respectively.
In details, 
- the rainfall-related data processing functions are encapsulated in `../flood_tool/visualisation/rainfall.py` file.
- tide related data processing functions are encapsulated in `../flood_tool/visualisation/tide.py`.
- river related data processing functions are encapsulated in `../flood_tool/visualisation/visualise.py`.
- rainfall, river and tide all use wet_day and typical_day data, respectively.
- The household and headcount related data processing functions are encapsulated in the `../flood_tool/visualisation/proverty.py file`.

<mark>For details of the functions, see ... /notebooks/funtion_explanation.md file<mark>


## **Contributing**

Contributions are welcome! Submit issues or pull requests to improve the project.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.


