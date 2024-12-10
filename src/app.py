"""
Load geojson polygons of zip codes and plot/display with plotly/streamlit.

To run the streamlit app:

    $ streamlit run src/app.py

To locally run the GitHub-hosted app:

    $ streamlit run https://raw.githubusercontent.com/bainmatt/streamlit-geomaps/main/src/app.py

"""

import os
import re
import sys
import math
# import time
import json
import geojson
import requests
import urllib.request

import pandas as pd
import streamlit as st
import geopandas as gpd
import plotly.express as px

from pathlib import Path
from fnmatch import fnmatch
from dotenv import load_dotenv
from collections import defaultdict
from plotly.colors import qualitative

# Add src to path to allow app to use absolute imports
src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

from src.paths import get_path_to
from src.colors import get_palette, rgb_to_hex

load_dotenv()
DATA_PATH = Path(os.getenv('DATA_PATH', get_path_to('data')))

lookup_dict_type = dict[str, dict[str, list[str]]]


# -- Get the data ------------------------------------------------------------


def download_gh_files(
    branch_dir_pattern_lookup: dict[str, str],
    output_path: Path = DATA_PATH,
    output_dir: str | None = None,
    save: bool = False,
    asglob: bool = True
) -> None:
    """
    Download files matching a given pattern from a given GitHub repository.

    Parameters
    ----------
    branch_dir_pattern_lookup : dict[str, str]
        Keys are relative branch paths '{git-user}/{repo-name}/{branch-name}'.
        Values are regex patterns (e.g., r".+\\.py$") or filename
        glob patterns (e.g., "*.py") to match files in that branch directory.

    output_path : Path, default=DATA_PATH
        Base output directory for downloaded files.

    output_dir : str, default=None
        Subdirectory within `output_path` to store the downloaded files.

    save : bool, default=False
        Safeguard to ensure no existing files are unintentionally overwritten.

    asglob : bool, default=True
        Whether to treat the pattern as a glob pattern (True) or regular
        expression (False).

    Examples
    --------
    >>> from src.app import download_gh_files

    >>> files_to_download = {
    ...     "HIPS/autograd/master": "autograd/util.py|autograd/tracer.py"
    ... }
    >>> download_gh_files(files_to_download, asglob=False)  # doctest: +ELLIPSIS
    <BLANKLINE>
    Requested file: HIPS/autograd/master/autograd/tracer.py
    Destination: ...
    Skipping download.
    ...

    >>> files_to_download = {
    ...     "HIPS/autograd/master": "*py"
    ... }
    >>> download_gh_files(files_to_download)  # doctest: +ELLIPSIS
    <BLANKLINE>
    Requested file: HIPS/autograd/master/autograd/__init__.py
    Destination: ...
    Skipping download.
    ...

    >>> files_to_download = {
    ...     "HIPS/autograd": "*.py"
    ... }
    >>> download_gh_files(files_to_download)  # doctest: +ELLIPSIS
    Invalid repository path: HIPS/autograd. Skipping.

    >>> files_to_download = {
    ...     "HIPS/autogra/master": "*.py"
    ... }
    >>> download_gh_files(files_to_download)  # doctest: +ELLIPSIS
    Failed to fetch file list from repo: HIPS/autogra/master. Skipping.

    >>> files_to_download = {
    ...     "HIPS/autograd/master": "*.csv"
    ... }
    >>> download_gh_files(files_to_download)  # doctest: +ELLIPSIS
    No files match pattern '*.csv' in HIPS/autograd/master. Skipping.
    """
    if output_dir:
        output_path = output_path / output_dir
    else:
        output_path = output_path / "downloads"

    for repo, pattern in branch_dir_pattern_lookup.items():
        # Construct GitHub API URL to list files recursively in the directory
        user_repo_branch = repo.split("/")
        if len(user_repo_branch) != 3:
            print(
                f"Invalid repository path: {repo}. Skipping."
            )
            continue

        user, repo_name, branch = user_repo_branch
        api_url = (
            "https://api.github.com/repos/"
            f"{user}/{repo_name}/git/trees/{branch}?recursive=1"
        )
        response = requests.get(api_url)
        if response.status_code != 200:
            print(f"Failed to fetch file list from repo: {repo}. Skipping.")
            continue

        # Filter files matching the pattern
        file_tree = response.json().get("tree", [])
        if asglob:
            matching_files = [
                f["path"] for f in file_tree
                if fnmatch(f["path"], pattern)
            ]
        else:
            matching_files = [
                f["path"] for f in file_tree
                if re.search(pattern, f["path"])
            ]

        if not matching_files:
            print(f"No files match pattern '{pattern}' in {repo}. Skipping.")
            continue

        # Download matching files
        for module in matching_files:
            module_url = f"https://raw.githubusercontent.com/{repo}/{module}"
            filepath = os.path.join(output_path, os.path.basename(module))

            print(f"\nRequested file: {repo}/{module}")
            print(f"Destination: {filepath}")

            if os.path.isfile(filepath):
                print(f"File {repo}/{module} already downloaded.")
                continue

            if not save:
                print("Skipping download.")
                continue

            print("Downloading.")
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(url=module_url, filename=filepath)


def download_data(
    url: str,
    output_file: str,
    output_path: Path = DATA_PATH,
    output_dir: str | None = None
) -> None:
    """
    Download and save a GeoJSON file from a given URL.

    Parameters
    ----------
    url : str
        The URL pointing to the GeoJSON data (e.g., Open Data API endpoint).

    output_file : str
        Name (with extension) to give to the downloaded file.

    output_path : Path, default=DATA_PATH
        Base output directory for downloaded files.

    output_dir : str, default=None
        Subdirectory within `output_path` to store the downloaded files.

    Examples
    --------
    # >>> url = "https://..."
    # >>> download_data(url, output_file="georef_united_states.geojson")
    # Data saved to /...geojson
    """
    if output_dir:
        output_path = output_path / output_dir
        output_filepath = output_path / output_file
    else:
        # output_path = output_path / "downloads"
        output_filepath = output_path / output_file

    if os.path.isfile(output_filepath):
        print(f"File already exists at {output_filepath}.")
        return

    response = requests.get(url)

    if response.status_code == 200:
        os.makedirs(output_path, exist_ok=True)

        with open(output_filepath, 'wb') as file:
            file.write(response.content)
        print(f"Data saved to {output_filepath}")
    else:
        raise Exception(
            f"Failed to retrieve data from {url}."
            f"Status code: {response.status_code}"
        )


# -- Process -----------------------------------------------------------------


@st.cache_data
def extract_zip_lookup(
    input_file: str,
    data_path: Path = DATA_PATH
) -> lookup_dict_type:
    """
    Extract a lookup dictionary with the structure
    {ste_name: {coty_name: [zcta5_code]}} from the GeoJSON data.

    Parameters
    ----------
    input_file : str
        Name of the GeoJSON file to parse.

    data_path : Path, default=DATA_PATH
        Path to the directory where the GeoJSON file is located.

    Returns
    -------
    dict
        A nested dictionary with states as keys and counties as values,
        containing lists of ZCTA codes.
    """
    input_filepath = data_path / input_file
    zcta_lookup: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    # Open and read the local GeoJSON file
    with open(input_filepath, 'r') as f:
        geojson_data = json.load(f)

    # Extract states, counties, and ZCTA codes
    for feature in geojson_data.get('features', []):
        properties = feature.get('properties', {})
        state = properties.get('ste_name', [None])[0]
        county = properties.get('coty_name', [None])[0]
        zcta_code = properties.get('zcta5_code', [None])[0]

        # Check for valid state, county, and zcta_code
        if state and county and zcta_code:
            zcta_lookup[state][county].append(zcta_code)

    return dict(zcta_lookup)


def save_lookup_dict(
    lookup_dict: lookup_dict_type,
    output_file: str,
    output_path: Path = DATA_PATH,
    output_dir: str | None = None
) -> None:
    """
    Save the lookup dictionary to a JSON file.
    """
    if output_dir:
        output_filepath = output_path / output_dir / output_file
    else:
        output_filepath = output_path / output_file
    with open(output_filepath, 'w') as f:
        json.dump(lookup_dict, f, indent=4)


@st.cache_data
def load_lookup_dict(
    input_file: str,
    input_path: Path = DATA_PATH,
    input_dir: str | None = None
) -> lookup_dict_type:
    """
    Load the lookup dictionary from a JSON file.
    """
    if input_dir:
        input_filepath = input_path / input_dir / input_file
    else:
        input_filepath = input_path / input_file
    with open(input_filepath, 'r') as f:
        lookup_dict: lookup_dict_type = json.load(f)
    return lookup_dict


def parse_geojson_zips(
    input_file: str,
    zip_lookup: lookup_dict_type,
    target_state: str = 'California',
    data_path: Path = DATA_PATH,
    output_dir: str = "geojson_files"
) -> None:
    """
    Parse a GeoJSON file and save a separate GeoJSON file per county.
    Each file will contain a FeatureCollection of ZCTA codes for the county.

    Parameters
    ----------
    input_file : str
        Name of the GeoJSON file to parse.

    zip_lookup : lookup_dict_type
        A lookup dictionary with structure
        {ste_name: {coty_name: [zcta5_code]}}.

    data_path : Path, default=DATA_PATH
        Path to the directory where the GeoJSON file is located.

    output_dir : str, default='geojson_files'
        Subdirectory to save the parsed county GeoJSON files.

    target_state : str, default='California'
        If provided, limits the processing to the specified state.
    """
    input_filepath = data_path / input_file
    output_path = data_path / output_dir

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Open and read the local GeoJSON file
    with open(input_filepath, 'r') as f:
        geojson_data = json.load(f)

    # Loop through lookup to generate separate GeoJSON files per county
    for state, counties in zip_lookup.items():
        if target_state and state != target_state:
            continue

        for county, zcta_codes in counties.items():
            # Filter features matching ZCTA codes for this state and county
            matching_features = [
                feature
                for feature in geojson_data.get('features', [])
                if feature.get(
                    'properties', {}).get('ste_name', [None])[0] == state
                and feature.get(
                    'properties', {}).get('coty_name', [None])[0] == county
                and feature.get('properties', {}).get(
                    'zcta5_code', [None])[0] in zcta_codes
            ]

            # If there are any matching features, create the GeoJSON file
            if matching_features:
                # Construct the output filename ({state}_{county}.geojson)
                output_filename = (
                    f"{state.replace(' ', '-')}"
                    f"_{county.replace(' ', '-')}.geojson"
                )
                output_filepath = output_path / output_filename

                # Create a FeatureCollection from the matching features
                feature_collection = {
                    "type": "FeatureCollection",
                    "features": matching_features
                }

                # Save the FeatureCollection to the output file
                with open(output_filepath, 'w') as out_file:
                    json.dump(feature_collection, out_file, indent=4)

                print(f"Saved: {output_filepath}")


@st.cache_data
def parse_geojson_states(
    input_file: str,
    zip_lookup: lookup_dict_type,
    target_state: str = 'California',
    data_path: Path = DATA_PATH,
    output_dir: str = "geojson_files"
) -> None:
    """
    Parse a GeoJSON and dissolve zip codes within counties for a target state.

    Save a single GeoJSON for the entire state with county-level boundaries.

    Parameters
    ----------
    input_file : str
        Name of the GeoJSON file to parse.

    zip_lookup : lookup_dict_type
        A lookup dictionary with structure
        {ste_name: {coty_name: [zcta5_code]}}.

    target_state : str
        The state to process.

    data_path : Path, default=DATA_PATH
        Path to the directory where the GeoJSON file is located.

    output_dir : str, default='geojson_files'
        Subdirectory to save the parsed state GeoJSON file.
    """
    input_filepath = data_path / input_file
    output_path = data_path / output_dir

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Open and read the local GeoJSON file
    with open(input_filepath, 'r') as f:
        geojson_data = json.load(f)

    print(f"Processing target state: {target_state}")

    # Check if the target state is in the lookup dictionary
    if target_state not in zip_lookup:
        print(
            f"Target state '{target_state}' not found in lookup dictionary."
        )
        return

    # Initialize an empty list to collect dissolved county geometries
    state_geometries = []

    # Process each county in the target state
    for county, zcta_codes in zip_lookup[target_state].items():
        print(f"Processing county: {county}")

        matching_features = [
            feature
            for feature in geojson_data.get('features', [])
            if feature.get(
                'properties', {}).get('ste_name', [None])[0] == target_state
            and feature.get(
                'properties', {}).get('coty_name', [None])[0] == county
        ]

        # Skip if no matching features for the county
        if not matching_features:
            print(f"No features found for county: {county}")
            continue

        # Convert matching features to a GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(matching_features)

        # Ensure geometries are valid
        gdf['geometry'] = gdf['geometry'].apply(
            lambda geom: geom if geom.is_valid else geom.buffer(0)
        )

        # Dissolve the geometries for the current county
        dissolved = gdf.dissolve(by=lambda x: county)

        # Reduce file size by simplifying geometries
        dissolved['geometry'] = dissolved['geometry'].simplify(
            tolerance=0.01, preserve_topology=True
        )

        # Retain only necessary columns
        # dissolved['county_name'] = county
        # dissolved = dissolved[['geometry', 'county_name']]

        state_geometries.append(dissolved)

    # If no geometries were collected, stop processing
    if not state_geometries:
        print(f"No features to save for state: {target_state}")
        return

    # Combine all dissolved county geometries into a single GeoDataFrame
    state_gdf = gpd.GeoDataFrame(
        pd.concat(state_geometries, ignore_index=True)
    )

    # Convert the state GeoDataFrame to GeoJSON format
    state_geojson = json.loads(state_gdf.to_json())

    # Construct the output file path
    output_filepath = output_path / f"{target_state.replace(' ', '-')}.geojson"

    # Save the state GeoJSON file
    try:
        with open(output_filepath, 'w') as out_file:
            json.dump(state_geojson, out_file, indent=4)
        print(f"Saved: {output_filepath}")
    except Exception as e:
        print(f"Error saving file {output_filepath}: {e}")


def get_available_regions(
    data_path: Path = DATA_PATH,
    data_dir: str = "geojson_files"
) -> list[str]:
    """
    Extract the names of the regions from the geojson properties.

    Example: get_available_regions()
    """
    if data_dir:
        data_path = data_path / data_dir

    return [
        os.path.splitext(region)[0]
        for region in os.listdir(data_path)
        if region.endswith('.geojson')
        and not region.startswith(('.', '_'))
    ]


@st.cache_data
def load_geojson_as_json(
    region: str,
    data_path: Path = DATA_PATH,
    data_dir: str = "geojson_files"
):
    """
    Load GeoJSON file based on region selection.

    Assumes you have files like 'akron.geojson', 'cleveland.geojson', etc.
    Example: load_geojson_as_json('akron')
    """
    if data_dir:
        data_path = data_path / data_dir

    file_path = f"{data_path}/{region}.geojson"
    with open(file_path, 'r') as f:
        return geojson.load(f)


@st.cache_data
def load_geojson_as_gdf(
    region: str,
    data_path: Path = DATA_PATH,
    data_dir: str = "geojson_files"
):
    """
    Load GeoJSON file based on region selection as a GeoDataFrame.

    Example: load_geojson_as_json('akron')
    """
    if data_dir:
        data_path = data_path / data_dir

    file_path = f"{data_path}/{region}.geojson"
    if os.path.exists(file_path):
        return gpd.read_file(file_path)
    else:
        st.error(f"GeoJSON file for region {region} not found.")
        return None


@st.cache_data
def load_csv_data(
    input_file: str,
    input_path: Path = DATA_PATH,
    input_dir: str | None = None
) -> pd.DataFrame | None:
    """
    Load data from a CSV file.
    """
    if input_dir:
        input_filepath = input_path / input_dir / input_file
    else:
        input_filepath = input_path / input_file

    if os.path.exists(input_filepath):
        return pd.read_csv(input_filepath, encoding='ISO-8859-1')
    else:
        st.error(f"File {input_filepath} not found.")
        return None


@st.cache_data
def process_population_data(
    pop_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare the population data.

    Melt wide form (region * yearly population) data into long form data
    with columns 'region', 'year', 'population'.

    References
    ----------
    .. [1] https://github.com/streamlit/gdp-dashboard-template
    """
    pop_df = pop_df.drop(columns='POPESTIMATE042020')

    id_vars = ['STNAME', 'CTYNAME']
    value_vars = pop_df.filter(like='POPESTIMATE').columns.to_list()

    # Pivot population year columns into two: year and population
    pop_df = (
        pop_df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='year',
            value_name='population',
        ).rename(
            columns={'STNAME': 'ste_name', 'CTYNAME': 'coty_name'}
        )
    )
    # Clean and normalize column values
    pop_df['coty_name'] = pop_df['coty_name'].str.replace(
        ' County', '', regex=False
    )
    pop_df['year'] = pd.to_numeric(pop_df['year'].str.replace(
        'POPESTIMATE', '', regex=False
    ))

    return pop_df


# -- Plot --------------------------------------------------------------------


def plot_zips(
    state_list: list[str],
    data_path: Path = DATA_PATH,
    data_dir: str = "geojson_files"
) -> str:
    """
    Generate a choropleth map using Plotly/Matbox and display it in Streamlit.

    Parameters
    ----------
    state_list : list[str]
        A list of valid state names to choose from a dropdown.

    data_path : Path
        Path to the directory containing the data files.

    data_dir : str
        Subdirectory where parsed GeoJSON files are stored.
    """
    # -- Load and configure --------------------------------------------------

    st.subheader('Step 1: Configure map', divider='gray')

    # 1. Select state

    region_selection_container = st.container(border=True)

    # Primary dropdown for state selection
    selected_state = region_selection_container.selectbox(
        label='**Step 1a**: Which state are you interested in?',
        options=sorted(state_list),
        label_visibility="visible"
        # sorted(get_available_regions(data_path, data_dir))
    )
    selected_state_fname = selected_state.replace(" ", "-")

    # 2. Parse

    # Dynamically parse the constituent counties and zip codes
    status = st.status("Running ...", expanded=True)
    status.write("Parsing data ...")
    extract_state_data(selected_state_fname)

    # 3. Select county

    # Secondary dropdown for county selection
    area_list = get_available_regions(data_path, data_dir)
    county_list = [area for area in area_list if area not in state_list]
    filtered_county_list = [
        cty for cty in county_list if selected_state_fname in cty
    ]
    formatted_county_list = sorted([
        county.split("_")[-1].replace("-", " ")
        for county in filtered_county_list
    ])
    selected_county = region_selection_container.selectbox(
        label="**Step 1b**: Which county would you like to view?",
        options=['All'] + formatted_county_list,
        label_visibility="visible"
    )
    if selected_county != 'All':
        selected_county_fname = (
            selected_state_fname + "_" + selected_county.replace(" ", "-")
        )
    if selected_county == 'All':
        selected_area = selected_state_fname
        selected_area_name = selected_state
        id_field = 'coty_name'
        id_label = 'County'
    else:
        selected_area = selected_county_fname
        selected_area_name = selected_county + ', ' + selected_state
        id_field = 'zcta5_code'
        id_label = 'ZIP'

    # Save region selection to session state for timeline plots
    selected_region = (
        'all' if selected_county == 'All' else selected_state
    )
    # st.session_state.region = (
    #     'all' if selected_county == 'All' else selected_county
    # )

    # -- Run -----------------------------------------------------------------

    # 3. Load

    # Load GeoJSON data for the selected region
    status.write("Loading data ...")

    geojson_data = load_geojson_as_json(selected_area, data_path, data_dir)
    gdf = load_geojson_as_gdf(selected_area, data_path, data_dir)

    # 4. Prepare

    status.write("Processing data ...")

    features = geojson_data.get('features', [])
    df = pd.DataFrame([{
        'id': feature['properties'].get(id_field, 'unknown')[0],
        'geometry': feature['geometry'],
    } for feature in features])

    # Sort both data structures indentically
    sorted_indices = df['id'].argsort()
    df = df.iloc[sorted_indices].reset_index(drop=True)
    gdf = gdf.iloc[sorted_indices].reset_index(drop=True)

    # Generate discrete color palette
    discrete_palette = [
        rgb_to_hex(color)
        for color in get_palette(
            "plasma", n_colors=len(df)
        )  # type: ignore[union-attr]
    ]
    discrete_color_scale = {
        df['id'][i]: discrete_palette[i] for i in range(len(df))
    }

    # Compute centroids
    # gdf_projected = gdf.to_crs(epsg=32633)
    # centroids = gdf_projected.geometry.centroid
    centroids = gdf.geometry.centroid
    mean_lat = centroids.y.mean()
    mean_lon = centroids.x.mean()

    # 5. Plot

    status.write("Constructing map ...")

    fig = px.choropleth_mapbox(
        df,
        geojson=gdf.geometry.__geo_interface__,
        locations=df.index,
        color=(
            'id'
            # if selected_county != 'All'
            # else pd.to_numeric(df['id'], errors='coerce')
        ),
        hover_name='id',
        hover_data={
            "id": False
        },
        zoom=5 if selected_county == 'All' else 7,
        center={
            "lat": mean_lat,
            "lon": mean_lon
        },
        mapbox_style="carto-positron",
        color_discrete_map=discrete_color_scale,
        color_continuous_scale="Plasma",
    )

    status.update(label="Data processed", state="complete", expanded=False)

    # 6. Style

    # Create buttons to highlight a given trace
    # traces = fig.data
    trace_buttons = [
        {
            'args': [
                {
                    'marker.opacity': [.5] * len(df),
                    # 'legendwidth': [20] * len(df),
                },
            ],
            'label': 'Select all',
            'method': 'restyle',
        }
    ] + [
        {
            'args': [
                {
                    'marker.opacity': [
                        .5 if j == i else 0.2
                        for j in range(len(df))
                    ],
                    # 'legendwidth': [
                    #     100 if j == i else 20
                    #     for j in range(len(df))
                    # ],
                },
            ],
            'label': (
                df['id'][i]
                # if selected_county != 'All'
                # else pd.to_numeric(df['id'], errors='coerce')[i]
            ),
            'method': 'restyle',
        } for i in range(len(df))
    ]

    # Create buttons to toggle the legend visibility
    legend_button_hide = {
        'args': [{'showlegend': False}],
        'label': 'Hide Legend',
        'method': 'relayout',
        'visible': True
    }
    legend_button_show = {
        'args': [{'showlegend': True}],
        'label': 'Show Legend',
        'method': 'relayout',
        'visible': True
    }

    # Create buttons to toggle street map
    style_button_carto = {
        'args': [{'mapbox.style': 'carto-positron'}],
        'label': 'Hide streets',
        'method': 'relayout',
        'visible': True
    }
    style_button_street = {
        'args': [{'mapbox.style': "open-street-map"}],
        'label': 'Show streets',
        'method': 'relayout',
        'visible': True
    }

    # Display buttons
    fig.update_layout(
        legend=dict(
            title=id_label,
            itemclick="toggleothers",
        ),
        mapbox=dict(
            style="carto-positron"
        ),
        updatemenus=[
            dict(
                type="dropdown",
                buttons=trace_buttons,
                direction="down",
                x=1,
                y=1,
                xanchor="right",
                yanchor="top",
                showactive=True,
                # pad=dict(t=10, r=10),
                # font=dict(size=10),
                borderwidth=0,
                bgcolor='#f5f5f5',
                font=dict(color="#000000"),
                visible=True,
            ),
            dict(
                type="buttons",
                buttons=[legend_button_show, legend_button_hide],
                direction="right",
                x=0,
                y=1,
                xanchor="left",
                yanchor="top",
                showactive=True,
                borderwidth=0,
                bgcolor='#f5f5f5',
                font=dict(color="#000000"),
                visible=True,
            ),
            dict(
                type="buttons",
                buttons=[style_button_carto, style_button_street],
                direction="right",
                x=0,
                y=1,
                xanchor="left",
                yanchor="bottom",
                showactive=True,
                borderwidth=0,
                bgcolor='#f5f5f5',
                font=dict(color="#000000"),
                visible=True,
            ),
        ],
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )

    # Customize hover and choropleth default opacity
    fig.update_traces(
        hovertemplate=(
            "<b>County: </b> %{hovertext}<extra></extra>"
            if selected_county == 'All' else
            "<b>ZIP: </b> %{hovertext}<extra></extra>"
        ),
        marker_opacity=.5,
    )

    ''
    ''

    st.subheader(f"Map of {selected_area_name}", divider='rainbow')
    st.plotly_chart(fig, use_container_width=True)
    return selected_region


def plot_hoods(
    data_path: Path = DATA_PATH,
    data_dir: str = "click_that_hood_files"
) -> None:
    """
    Generate a choropleth map and display it using Streamlit and Plotly.
    """
    # Dropdown for region selection
    region_option = st.selectbox(
        'Select Region',
        sorted(get_available_regions(data_path, data_dir))
    )

    # Get the geojson data for the selected region
    geojson_data = load_geojson_as_json(region_option, data_path, data_dir)
    gdf = load_geojson_as_gdf(region_option, data_path, data_dir)

    # Convert geojson features to a DataFrame
    features = geojson_data['features']
    df = pd.DataFrame([{
        'id': feature['properties']['name'],
        'name': feature['properties']['name'],
        'geometry': feature['geometry']
    } for feature in features])

    # Map categories to palette colors
    palette = qualitative.Plotly
    categories = df['name'].unique()
    custom_colors = {
        category: palette[i % len(palette)]
        for i, category in enumerate(categories)
    }

    # Create choropleth
    fig = px.choropleth(
        gdf,
        geojson=gdf.geometry.__geo_interface__,
        locations=gdf.index,
        color='name',
        color_continuous_scale="Viridis",
        color_discrete_map=custom_colors,
        title=f"Map of {region_option.capitalize()}",
        hover_name='name',
        category_orders={"name": sorted(df['name'].unique())},
    )

    # Customize hover
    fig.update_traces(
        hovertemplate="<b>Area:</b> %{hovertext}<extra></extra>"
    )
    # Adjust background visibility and legend style/toggling
    fig.update_traces(marker_opacity=0.8)
    fig.update_layout(
        legend_title=dict(text="Area"),
        legend_itemclick="toggle",
        legend_itemdoubleclick="toggleothers",
    )
    # Adjust the geo layout to ensure the region is visible
    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig)


def plot_trajectories(
    state_list: list[str],
    wdi_df: pd.DataFrame,
    # region: str = 'all'
):
    """
    Plot trajectories and infocards for a given WDI.

    Analyze regional WDIs (world development indicators) in time.
    """
    # -- Load and configure --------------------------------------------------

    st.subheader('Step 2: Configure timeline', divider='gray')

    # 0. Select state

    wdi_selection_container = st.container(border=True)

    # Primary dropdown for state selection
    selected_region = wdi_selection_container.selectbox(
        label='**Step 2a**: Which state are you interested in?',
        options=['All'] + sorted(state_list),
        label_visibility="visible"
    )

    # 1. Select range

    min_value = wdi_df['year'].min()
    max_value = wdi_df['year'].max()

    from_year, to_year = wdi_selection_container.slider(
        '**Step 2b**: Which years are you interested in?',
        min_value=min_value,
        max_value=max_value,
        value=[min_value, max_value])

    # 2. Filter

    if selected_region == 'All':
        filtered_wdi_df = wdi_df[
            (wdi_df['coty_name'] == wdi_df['ste_name'])
            & (from_year <= wdi_df['year'])
            & (wdi_df['year'] <= to_year)
        ]
        region_label = "the United States"
    else:
        filtered_wdi_df = wdi_df[
            (wdi_df['ste_name'] == selected_region)
            & (wdi_df['coty_name'] != selected_region)
            & (from_year <= wdi_df['year'])
            & (wdi_df['year'] <= to_year)
        ]
        region_label = f"{selected_region}"
    region_list = filtered_wdi_df['coty_name'].unique().tolist()

    sorted_final_wdi_df = (
        filtered_wdi_df[filtered_wdi_df['year'] == to_year]
        .sort_values(by=['population'], ascending=False)
    )
    top_regions = sorted_final_wdi_df.head(6)['coty_name'].tolist()

    # 3. Select region

    selected_regions: list[str] = wdi_selection_container.multiselect(
        label='**Step 2c**: Which regions would you like to view?',
        options=region_list,
        default=top_regions,
    )
    subfiltered_wdi_df = filtered_wdi_df[
        filtered_wdi_df['coty_name'].isin(selected_regions)
    ]

    # In regions with more than one value per year only keep max
    subfiltered_wdi_df = subfiltered_wdi_df.loc[
        subfiltered_wdi_df.groupby(['coty_name', 'year'])
        ['population'].idxmax()
    ].reset_index(drop=True)

    # -- Run -----------------------------------------------------------------

    # 4. Plot

    ''
    ''

    st.subheader(f"Population of {region_label} over time", divider='rainbow')

    ''

    st.line_chart(
        subfiltered_wdi_df.rename(columns={'coty_name': 'County'}),
        x='year',
        y='population',
        color='County',
    )

    # 4.1. Display the data

    pretty_df = (
        subfiltered_wdi_df
        .sort_values(
            by=['coty_name', 'year'],
            ascending=True
        )
        .rename(columns={"coty_name": "region"})
        [['region', 'year', 'population']]
    )
    with st.sidebar:
        st.dataframe(
            pretty_df
            .groupby(['region'])
            .agg(
                population_list=('population', list),
                final_population=('population', lambda x: x.iloc[-1]),
            )
            .reset_index()
            .sort_values(by='final_population', ascending=False),
            column_config={
                "region": "region",
                "population_list": st.column_config.AreaChartColumn(
                    label="population over time",
                    help="The population trajectory over the selected range",
                    y_min=0,
                    y_max=30_000_000
                ),
                "final_population": st.column_config.ProgressColumn(
                    label="final population",
                    help="The final population from the selected range",
                    format="%f",
                    min_value=0,
                    max_value=30_000_000,
                ),
            },
            hide_index=True,
        )
        st.dataframe(
            pretty_df,
            hide_index=True,
        )

    # 5. Display infocards

    first_year = wdi_df[wdi_df['year'] == from_year]
    last_year = wdi_df[wdi_df['year'] == to_year]

    ''

    st.subheader(
        f'Population of {region_label} in {to_year} vs. {from_year}',
        divider='rainbow'
    )

    ''

    cols = st.columns(4)

    for i, region in enumerate(selected_regions):
        col = cols[i % len(cols)]

        with col:
            first_pop = first_year[
                first_year['coty_name'] == region
            ]['population'].iat[0]

            last_pop = last_year[
                last_year['coty_name'] == region
            ]['population'].iat[0]

            if math.isnan(first_pop) or first_pop == 0:
                growth = 'n/a'
                # delta_color = 'off'
            else:
                growth_rate = (last_pop / first_pop) - 1
                growth = f'{growth_rate:,.2f}x'
                # delta_color = 'inverse' if growth_rate < 1 else 'normal'

            st.metric(
                label=f'{region}',
                value=f'{last_pop / 1e6:,.1f}M',
                delta=growth,
                # delta_color=delta_color  # type: ignore[arg-type]
            )


# -- Orchestrate -------------------------------------------------------------


def extract_state_data(
    target_state: str,
    data_path: Path = DATA_PATH,
    output_dir: str = "geojson_files"
) -> None:
    """
    Perform parsing at the state and county level for the state given by name.
    """
    output_path = data_path / output_dir
    os.makedirs(output_path, exist_ok=True)

    existing_states = [
        os.path.splitext(region)[0].lower()
        for region in os.listdir(output_path)
        if region.endswith('.geojson')
        and not region.startswith(('.'))
        and '_' not in region
    ]
    target_state_processed = target_state.replace(' ', '-').lower()
    if target_state_processed in existing_states:
        print(
            f"Target state '{target_state}' already parsed."
        )
        return

    zip_lookup = load_lookup_dict('zip_lookup.json')
    input_file = "georef_united_states.geojson"

    status = st.status("Running ...", expanded=True)
    status.write("Parsing counties ...")
    parse_geojson_zips(input_file, zip_lookup, target_state)

    status.write("Parsing ZIP codes ...")
    parse_geojson_states(input_file, zip_lookup, target_state)
    status.update(label="Data parsed", state="complete", expanded=False)


def main():
    # import doctest
    # doctest.testmod(verbose=True)

    # from src.workflow import doctest_function
    # doctest_function(parse_geojson_zips, globs=globals())
    # exit()

    # -- Orchestrate ---------------------------------------------------------

    # Set up page title and icon
    st.set_page_config(
        page_title="Geography",
        page_icon=":earth_americas:",
        layout="centered",
    )
    st.title(":earth_americas: Zippy Geomaps")

    st.markdown(
        'Analyze country data in space and time '
        'at both the state and county level.'
    )

    with st.expander('About this app'):
        st.markdown('**What can this app do?**')
        st.info(
            '- Explore the geography of state counties and county ZIP codes.\n'
            '- Explore population over time across and within states.'
        )

        st.markdown('**How to use the app?**')
        st.warning(
            'To interact with this app, simply follow the steps below:\n'
            '1. **Configure map**: '
            'Select a state and optional county of interest.\n'
            '2. **Configure timeline**: '
            'Select a year range and regions of interest.'
        )

        st.markdown('**Under the hood**')
        st.markdown('Data sets:')
        st.info(
            '- **Geographical data**: '
            '[Opendatasoft United States ZIP code data](https://public.opendatasoft.com/explore/dataset/georef-united-states-of-america-zcta5/)\n'  # noqa: E501
            '- **Population data**: '
            '[United States Census Bureau county population totals](https://www.census.gov/data/datasets/time-series/demo/popest/2010s-counties-total.html#par_textimage_739801612)',  # noqa: E501
            # language='markdown'
        )

        st.markdown('Libraries used:')
        st.info(
            '''
            - [Pandas](https://pandas.pydata.org/docs/) for data wrangling
            - [Plotly](https://plotly.com/) for interactive plots
            - [Altair](https://altair-viz.github.io/) for chart creation
            - [Streamlit](https://streamlit.io/) for user interface
            '''
        )

    def run_global_app():
        download_gh_files(
            {
                "codeforgermany/click_that_hood/main":
                    "public/data/*.geojson"
            },
            output_dir="click_that_hood_files",
            save=True
        )
        plot_hoods(data_dir='click_that_hood_files')

    def run_local_app():
        # 0. Download data and load

        data_url = (
            "https://public.opendatasoft.com"
            "/api/explore/v2.1/catalog/datasets/"
            "georef-united-states-of-america-zcta5/exports/geojson"
        )
        data_file = "georef_united_states.geojson"
        data_filepath = DATA_PATH / data_file

        status = st.status("Running ...", expanded=True)
        status.write("Getting data ...")

        if not os.path.isfile(data_filepath):
            download_data(url=data_url, output_file=data_file)

            # 1. Extract a {state: county: [zip]} lookup table

            zip_lookup = extract_zip_lookup(input_file=data_file)
            save_lookup_dict(zip_lookup, 'zip_lookup.json')
        else:
            zip_lookup = load_lookup_dict('zip_lookup.json')

        status.update(label="Data loaded", state="complete", expanded=False)

        # 2. Run app

        ''
        ''
        ''

        plot_zips(state_list=list(zip_lookup.keys()))

    def run_wdi_app(region: str = 'all'):
        # 0. Download data and load

        if "pop_df" not in st.session_state:
            pop_data_url = (
                "https://www2.census.gov/programs-surveys/popest/datasets/"
                "2010-2020/counties/totals/co-est2020.csv"
            )
            pop_data_file = "county_populations.csv"
            download_data(url=pop_data_url, output_file=pop_data_file)

            st.session_state["pop_df"] = process_population_data(
                load_csv_data(
                    input_file=pop_data_file
                )  # type: ignore[arg-type]
            )
        pop_df = st.session_state["pop_df"]

        # 1. Run app

        ''
        ''
        ''

        zip_lookup = load_lookup_dict('zip_lookup.json')
        plot_trajectories(state_list=list(zip_lookup.keys()), wdi_df=pop_df)

    with st.sidebar:
        st.subheader('Tabulated data', divider='rainbow')

    # with st.sidebar:
    #     use_global_data = st.toggle('Use global dataset')
    # if use_global_data:
    #     run_global_app()
    # else:
    #     run_local_app()

    run_local_app()

    # if "region" not in st.session_state:
    #     region = 'all'

    run_wdi_app()


if __name__ == "__main__":
    main()
