# Toolkit for 4D analysis of oceanographic variables

## Overview

This project provides a pipeline for working with CTD profiles (from SNO MEMO and SNO ARGO data for instance)

The code contains different modules: grid definition, data access, and data joining. This makes it scalable, maintainable, and adaptable to large datasets.

---

## Components

### Grid

The grid defines a structured space over time, depth, latitude, and longitude. 
Users can specify ranges and resolutions for each dimension. 
The grid is stored as a netcdf file, allowing easy indexing and compatibility with existing scientific data tools.

### Data Source

The data source layer is responsible for retrieving data from storage. 
It uses a SQL-based engine to query data efficiently from files or databases. 
This allows for rapid filtering and aggregation without loading data into memory.

### Join

The join module adds observational data to the user-defined grid. 
It maps profiles to the appropriate grid cells.

---

## Workflow

1. Define the grid with the desired spatial and temporal resolution.
2. Query observational data using the data source layer.
3. Aggregate data to match grid resolution.
5. Join the processed data into the grid.


---

## Notes

A module is coming soon for functional PCA approaches


## Author

Anatole Gros-Martial
- GitHub: https://github.com/gmanatole
- Email: anatole.gros-martial@cebc.cnrs.fr
