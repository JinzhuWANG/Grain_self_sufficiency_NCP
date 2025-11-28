# Grain Self-Sufficiency Modeling for North China Plain (NCP)

This project models future grain production and self-sufficiency in the North China Plain under various climate change scenarios, considering multiple contributing factors including climate change, urban expansion, land reclamation, and agricultural trends.

## Overview

The modeling pipeline integrates data from multiple sources to project grain production changes from the baseline year through future scenarios:

- **Climate Data**: GAEZ (Global Agro-Ecological Zones) v4 crop yield projections
- **Population & Urban Growth**: SSP (Shared Socioeconomic Pathways) scenarios
- **Agricultural Statistics**: Provincial yearbook records
- **Land Use**: Historical and projected urban expansion and cropland reclamation

## Project Structure

### Main Analysis Pipeline

The analysis follows a sequential numbering scheme (01-17):

1. **01-07: Yield Projections**
   - `01_Extrapolate_GAEZ_attainable_yield.py` - Extrapolate GAEZ attainable yields
   - `02_Convert_GAEZ_attainable_dryWeight_to_yield.py` - Convert dry weight to grain yield
   - `03_Force_GAEZ_attainable_agree_with_GYGA.py` - Calibrate with GYGA data
   - `04_Force_GAEZ_actual_yield_agree_with_yearbook.py` - Align with yearbook statistics
   - `05_Extend_GAEZ_yield_by_attainable_potential.py` - Extend yields by potential
   - `06_Extrapolate_yearbook_yield.py` - Extrapolate historical trends
   - `07_Extend_GAEZ_yield_by_yearbook_trend.py` - Integrate yearbook trends

2. **08-14: Land Use Changes**
   - `08_Get_urban_hist_and_potential_area.py` - Urban area historical analysis
   - `09_1_Inspect_population_and_urban_area.py` - Population-urban relationship analysis
   - `09_2_Predict_urban_population_ratio.py` - Urbanization rate predictions
   - `09_3_Predict_urban_area_with_total_population_and_urban_ratio.py` - Urban area projections
   - `10_get_SSP_Urban_area_GEE.py` - SSP urban scenarios from Google Earth Engine
   - `11_get_trainsition_threshold_for_pred_urban.py` - Urban transition thresholds
   - `12_Reclassify_urban_potential.py` - Reclassify urban potential areas
   - `13_Get_Urban_occupying_cropland_area.py` - Calculate cropland lost to urbanization
   - `14_Get_reclimation_increased_framland_area.py` - Project land reclamation gains

3. **15-17: Production & Factor Analysis**
   - `15_Force_GAEZ_area_agree_with_yearbook.py` - Harmonize area statistics
   - `16_Multi_yield_area_for_production.py` - Calculate total production
   - `17_Calc_factor_contribution_to_production.py` - Decompose factor contributions

### Supporting Directories

- **`data/`** - Raw and processed data files
- **`data_transform/`** - Data preprocessing scripts
  - `01_Harmonize_lucc_tifs.py` - Harmonize land use/cover rasters
  - `02_Get_GAEZ_to_yearbook_multiplier.py` - Calculate calibration factors
  - `03_Create_province_mask.py` - Generate provincial masks
  - `04_Calculate_TIF_area_km2.py` - Calculate raster area statistics
  - `05_Get_GYGA_attainable.py` - Process GYGA yield data
  - `06_Get_GDP_POP_under_SSPs.py` - Extract SSP socioeconomic data
  - `NC2TIFF_clipped.py` - NetCDF to GeoTIFF conversion

- **`helper_func/`** - Utility functions
  - `parameters.py` - Configuration and constants
  - `calculate_GAEZ_stats.py` - GAEZ data statistics
  - `calculate_WGS_pix_area_km2.py` - Pixel area calculations
  - `get_yearbook_records.py` - Yearbook data extraction
  - `GAEZ_1_scrap.py`, `GAEZ_2_download.py`, `GAEZ_3_Clip_TIF.py` - GAEZ data acquisition

## Key Features

### Climate Scenarios
- Multiple RCP (Representative Concentration Pathways) scenarios
- With/without CO2 fertilization effects
- Multiple climate models integrated

### Socioeconomic Scenarios
- SSP1, SSP2, SSP3, SSP5 pathways
- Provincial-level population projections
- Urban-rural migration patterns

### Factor Decomposition
The analysis quantifies contributions from:
- Climate change impacts on crop yields
- Urbanization and cropland loss
- Agricultural land reclamation
- Yield trend improvements (technology, management)

### Uncertainty Quantification
- Monte Carlo simulations for probabilistic projections
- Confidence intervals (5th-95th percentiles)
- Multi-model ensemble statistics

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python 3.8+
- numpy, pandas
- rasterio, rioxarray, xarray
- geopandas, rasterstats
- plotnine (ggplot2-style plotting)
- pymc (Bayesian modeling)
- earthengine-api (Google Earth Engine)

## Usage

### Running the Full Pipeline

Execute scripts in numerical order (01 through 17):

```bash
python 01_Extrapolate_GAEZ_attainable_yield.py
python 02_Convert_GAEZ_attainable_dryWeight_to_yield.py
# ... continue through 17
```

### Key Outputs

Results are saved to `data/results/`:
- CSV files: `step_XX_*.csv` - Statistical outputs
- Plots: `fig_step_XX_*.svg` - Visualizations
- GeoTIFF files: Spatial projections

### Visualization Examples

The final script (`17_Calc_factor_contribution_to_production.py`) generates comprehensive plots showing:
- Temporal trends in production changes
- Factor-specific contributions
- Scenario comparisons
- Provincial-level breakdowns

## Data Sources

- **GAEZ v4**: FAO Global Agro-Ecological Zones
- **GYGA**: Global Yield Gap Atlas
- **SSP Database**: Shared Socioeconomic Pathways
- **Provincial Yearbooks**: Chinese provincial agricultural statistics
- **Land Use Data**: Historical land cover classifications

## Methodology

The modeling approach combines:
1. **Process-based models**: GAEZ crop simulations under climate scenarios
2. **Statistical calibration**: Alignment with historical observations
3. **Trend extrapolation**: Continuation of observed agricultural improvements
4. **Spatial analysis**: Raster-based land use change modeling
5. **Uncertainty propagation**: Monte Carlo sampling through the modeling chain

