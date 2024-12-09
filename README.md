# :earth_americas: streamlit-geomaps

A choropleth map divides a region into polygons, each outlining a region of
interest such as an administrative border or ZIP code.
This repository contains configuration and source files used to generate
an interactive choropleth map of the United States, with county-
and ZIP code-level resolution.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://zippy-geomaps.streamlit.app/)
<!--
## Analysis

View the report [here](https://bainmatt.github.io/streamlit-geomaps/notebooks/report.html).
 -->

## Getting started

1. Install the requirements

    ```bash
    pip install -r requirements.txt
    ```

2. Run the app

    ```bash
    streamlit run streamlit_app.py
    ```

## Example

A state-level view of Ohio, USA, with choropleths representing its counties:

![ohio_choropleth](output/figures/ohio_choropleth.png)

A county-level view of Athens, Ohio, with choropleths representing its ZIP
codes:

![ohio_athens_choropleth](output/figures/ohio_athens_choropleth.png)

## Tools

Tools used:

- [`Pandas`](https://pandas.pydata.org/docs/) for data wrangling
- [`Plotly`](https://plotly.com/) for interactive plots
- [`Streamlit`](https://streamlit.io/) for user interface
