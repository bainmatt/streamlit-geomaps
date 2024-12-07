"""
Load geojson polygons of zip codes and plot/display with plotly/streamlit.

To run the streamlit app:

    $ streamlit run src/app.py

"""
import os
import re
import json
import gdown
import geojson
import requests
import urllib.request

import pandas as pd
import streamlit as st
import geopandas as gpd
import plotly.express as px

from pathlib import Path
from fnmatch import fnmatch
from collections import defaultdict
from plotly.colors import qualitative

from src.paths import get_path_to
from src.colors import get_palette, rgb_to_hex

from dotenv import load_dotenv

load_dotenv()

GDRIVE_URL = (
    "https://drive.google.com/drive/folders/"
    "1Zdt8bkAWc_iIwSCUNhkXEjXxPLraKkxd?usp=sharing"
)
DATA_PATH = Path(os.getenv('DATA_PATH', get_path_to('data')))
lookup_dict_type = dict[str, dict[str, list[str]]]


def download_gdrive_folder(
    folder_url: str = GDRIVE_URL,
    download_path: Path = get_path_to('data'),
    download_dir: str = "geojson_files"
):
    """
    Download all files from a Google Drive folder.
    """
    download_path = download_path / download_dir
    download_path.mkdir(parents=True, exist_ok=True)

    print(f'Downloading folder from: {folder_url}')
    gdown.download_folder(
        folder_url, output=str(download_path), quiet=False
    )


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
        output_filepath = output_path / output_dir / output_file
    else:
        output_filepath = output_path / "downloads" / output_file

    response = requests.get(url)

    if response.status_code == 200:
        with open(output_filepath, 'wb') as file:
            file.write(response.content)
        print(f"Data saved to {output_filepath}")
    else:
        raise Exception(
            f"Failed to retrieve data from {url}."
            f"Status code: {response.status_code}"
        )


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


def plot_zips(
    data_path: Path = DATA_PATH,
    data_dir: str = "geojson_files"
) -> None:
    """
    Generate a choropleth map using Plotly/Matbox and display it in Streamlit.

    Parameters
    ----------
    data_path : Path
        Path to the directory containing the data files.

    data_dir : str
        Subdirectory where parsed GeoJSON files are stored.
    """
    area_list = get_available_regions(data_path, data_dir)
    state_list = [area for area in area_list if '_' not in area]
    county_list = [area for area in area_list if area not in state_list]

    # Primary dropdown for state selection
    selected_state = st.selectbox(
        'Select a state',
        sorted(state_list)
        # sorted(get_available_regions(data_path, data_dir))
    )
    # Secondary dropdown for county selection
    filtered_county_list = [
        cty for cty in county_list if selected_state in cty
    ]
    selected_county = st.selectbox(
        "Select a county",
        ['All'] + sorted(filtered_county_list)
        # df['id']
    )
    if selected_county == 'All':
        selected_area = selected_state
        id_field = 'coty_name'
        id_label = 'County'
    else:
        selected_area = selected_county
        id_field = 'zcta5_code'
        id_label = 'ZIP'

    # Load GeoJSON data for the selected region
    geojson_data = load_geojson_as_json(selected_area, data_path, data_dir)
    gdf = load_geojson_as_gdf(selected_area, data_path, data_dir)

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
        for color in get_palette("plasma", n_colors=len(df))
    ]
    discrete_color_scale = {
        df['id'][i]: discrete_palette[i] for i in range(len(df))
    }

    # Create a choropleth map using Plotly/Mapbox
    fig = px.choropleth_mapbox(
        df,
        geojson=gdf.geometry.__geo_interface__,
        locations=df.index,
        color='id',
        # color=pd.to_numeric(df['id'], errors='coerce'),  # For continuous
        hover_name='id',
        hover_data={
            "id": False
        },
        title=f"Map of {', '.join(selected_area.split('_')[::-1]).title()}",
        zoom=7,
        center={
            "lat": gdf.geometry.centroid.y.mean(),
            "lon": gdf.geometry.centroid.x.mean()
        },
        mapbox_style="carto-positron",  # {open-street-map, carto-positron}
        color_discrete_map=discrete_color_scale,
        # color_continuous_scale="Plasma",
    )

    # Customize hover/opacity, layout/legend/margins
    fig.update_traces(
        hovertemplate="<b>Area:</b> %{hovertext}<extra></extra>",
        # hovertemplate="<b>ZIP:</b> %{hovertext}<extra></extra>",
        marker_opacity=.5,
    )
    fig.update_layout(
        # mapbox=dict(layers=[{
        #     "source": gdf.geometry.__geo_interface__,
        #     "type": "fill",
        #     "color": "rgba(0,0,0,0)",  # Bg transparency (last value 0-1)
        # }]),
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        legend_title=dict(text=id_label),
        legend_itemclick="toggle",
        legend_itemdoubleclick="toggleothers",
        # coloraxis_colorbar=dict(title="ZIP Codes"),  # For continuous
        # coloraxis_showscale=False,
    )

    # **********

    # selected_idx = df[df['id'] == selected_area].index[0]

    # # # Add a red border to the selected region
    # fig.update_traces(
    #     geojson=gdf.geometry.__geo_interface__,
    #     line=dict(color='red', width=10),
    #     selector=dict(location=gdf.index[selected_idx])
    # )

    # **********

    st.plotly_chart(fig)


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


# -- Orchestration -------------------------------------------------------


def extract_state_data(
    target_state: str,
    data_path: Path = DATA_PATH,
    output_dir: str = "geojson_files"
) -> None:
    """
    Perform parsing at the state and county level for the state given by name.
    """
    output_path = data_path / output_dir
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
    parse_geojson_zips(input_file, zip_lookup, target_state)
    parse_geojson_states(input_file, zip_lookup, target_state)


def main():
    # import doctest
    # doctest.testmod(verbose=True)

    # from src.workflow import doctest_function
    # doctest_function(parse_geojson_zips, globs=globals())

    # exit()

    # -- Run once ------------------------------------------------------------

    # 0. Download data

    # download_gh_files(
    #     {"codeforgermany/click_that_hood/main": "public/data/*.geojson"},
    #     save=True
    # )
    # url = (
    #     "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    #     "georef-united-states-of-america-zcta5/exports/geojson"
    # )
    # download_data(url, output_file="georef_united_states.geojson")

    # 1. Extract a {state: county: [zip]} lookup table

    # zip_lookup = extract_zip_lookup(input_file="georef_united_states.geojson")
    # save_lookup_dict(zip_lookup, 'zip_lookup.json')

    # For use in a cloud-based application
    geojson_folder = Path(DATA_PATH / 'geojson_files')
    if not geojson_folder.exists() or not any(geojson_folder.iterdir()):
        download_gdrive_folder()

    # -- Orchestrate ---------------------------------------------------------

    # 2. Parse GeoJSON files

    # extract_state_data('New Mexico')

    # 4. Run app

    # plot_hoods(data_dir='click_that_hood_files_small')
    plot_zips()


if __name__ == "__main__":
    main()
