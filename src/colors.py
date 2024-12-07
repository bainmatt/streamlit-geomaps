"""
Versatile colour manipulation utilities.
"""

import re
import random
import colorsys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Literal
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


PaletteType = Literal[
    # Sequential
    'Blues', 'BuGn', 'BuPu', 'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
    'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu', 'Reds', 'YlGn', 'YlGnBu',
    'YlOrBr', 'YlOrRd',

    'Blues_r', 'BuGn_r', 'BuPu_r', 'GnBu_r', 'Greens_r', 'Greys_r',
    'Oranges_r', 'OrRd_r', 'PuBu_r', 'PuBuGn_r', 'PuRd_r', 'Purples_r',
    'RdPu_r', 'Reds_r', 'YlGn_r', 'YlGnBu_r', 'YlOrBr_r', 'YlOrRd_r',

    # Diverging
    'BrBG', 'PRGn', 'PiYG', 'PuOr', 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn',
    'Spectral',

    'BrBG_r', 'PRGn_r', 'PiYG_r', 'PuOr_r', 'RdBu_r', 'RdGy_r', 'RdYlBu_r',
    'RdYlGn_r', 'Spectral_r',

    # Qualitative
    'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3',
    'tab10', 'tab20', 'tab20b', 'tab20c',

    # Miscellaneous
    'inferno', 'mako', 'magma', 'plasma', 'rocket', 'turbo', 'viridis',

    'inferno_r', 'mako_r', 'magma_r', 'plasma_r', 'rocket_r', 'turbo_r',
    'viridis_r',

    # Seaborn
    'husl',
]

RGBType = tuple[int, ...] | tuple[float, ...]


def hex_to_rgb(hex_color: str) -> RGBType:
    """
    Convert a HEX colour to RGB.

    Parameters
    ----------
    hex_color : str
        A string representing a colour in HEX format.

    Returns
    -------
    RGBType
        The (scaled/compressed) RGB representation of `hex_color`,
        with values in [0, 255].

    Examples
    --------
    >>> from src.colors import hex_to_rgb
    >>> hex_to_rgb("#ff0000")
    (255, 0, 0)
    >>> try:
    ...     hex_to_rgb("#f0")
    ... except TypeError as e:
    ...     print(f"TypeError: {e}")
    TypeError: invalid hex color: #f0
    """
    is_hex = re.fullmatch(r"^#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$", hex_color)
    if not is_hex:
        raise TypeError(f"invalid hex color: {hex_color}")

    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: RGBType) -> str:
    """
    Convert an RGB color to HEX.

    Parameters
    ----------
    rgb : RGBType
        A 3-tuple representing a color in RGB format,
        with all values in either [0, 255] or [0.0, 1.0].

    Returns
    -------
    str
        The HEX representation of `rgb`.

    Raises
    ------
    ValueError
        If RGB values are out of the expected range.

    Examples
    --------
    >>> from src.colors import rgb_to_hex
    >>> rgb_to_hex(rgb=(255, 0, 0))
    '#ff0000'
    >>> rgb_to_hex(rgb=(1., 0., 0.))
    '#ff0000'
    """
    rgb = _process_rgb(gradient=rgb, scale_mode="stretch")[0]
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def darken_color(rgb: RGBType, factor: float) -> RGBType:
    """
    Darken an RGB color by a given factor.

    Parameters
    ----------
    rgb : RGBType
        A 3-tuple representing a color in RGB format,
        with all values in either [0, 255] or [0.0, 1.0].
    factor : float
        A proportion between 0 and 1 by which to darken `rgb`.

    Returns
    -------
    RGBType
        The darkened RGB color representation, with values in [0.0, 1.0].

    Examples
    --------
    >>> from src.colors import darken_color
    >>> darken_color(rgb=(0.0785, 0.0393, 1.0), factor=0.)
    (0.0785, 0.0393, 1.0)
    >>> darken_color(rgb=(0.0785, 0.0393, 1.0), factor=1.)
    (0.0, 0.0, 0.0)
    >>> darken_color(rgb=(0.0785, 0.0393, 1.0), factor=.5)
    (0.03925, 0.01965, 0.5)
    """
    assert 0. <= factor <= 1., f"factor must be between 0 and 1, got {factor}"
    rgb = _process_rgb(gradient=rgb, scale_mode="compress")[0]

    return tuple(max((c * (1. - factor)), 0.) for c in rgb)


def lighten_color(rgb: RGBType, factor: float) -> RGBType:
    """
    Lighten an RGB color by a given factor.

    Parameters
    ----------
    rgb : RGBType
        A 3-tuple representing a color in RGB format,
        with all values in either [0, 255] or [0.0, 1.0].

    factor : float
        A proportion between 0 and 1 by which to lighten `rgb`.

    Returns
    -------
    RGBType
        The lightened RGB color representation, with values in [0.0, 1.0].

    Examples
    --------
    >>> from src.colors import lighten_color
    >>> lighten_color(rgb=(0.0785, 0.0393, 1.0), factor=0.)
    (0.0785, 0.0393, 1.0)
    >>> lighten_color(rgb=(0.0785, 0.0393, 1.0), factor=1.)
    (1.0, 1.0, 1.0)
    >>> lighten_color(rgb=(0.0785, 0.0393, 1.0), factor=.5)
    (0.53925, 0.51965, 1.0)
    """
    assert 0. <= factor <= 1., f"factor must be between 0 and 1, got {factor}"
    rgb = _process_rgb(gradient=rgb, scale_mode="compress")[0]

    return tuple(min((c + (1. - c) * factor), 1.) for c in rgb)


def random_color() -> str:
    """
    Generate a random HEX color.

    Returns
    -------
    str
        A string representing a colour in HEX format.

    Examples
    --------
    >>> from src.colors import random_color
    >>> import random
    >>> random.seed(0)
    >>> random_color()
    '#c53edf'
    """
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def create_gradient(
    rgb1: RGBType,
    rgb2: RGBType,
    n_colors: int,
    as_hex: bool = False
) -> list[RGBType] | list[str]:
    """
    Generate a gradient between two colors.

    Parameters
    ----------
    rgb1: RGBType
        A 3-tuple representing the first color in RGB format,
        with all values in either [0, 255] or [0.0, 1.0].
    rgb2: RGBType
        A 3-tuple representing the second color in RGB format,
        with all values in either [0, 255] or [0.0, 1.0].
    n_colors : int
        The number of colours to interpolate between `rgb1` and `rgb2`.
    as_hex : bool, default=False
        If True, converts each colour into a HEX representation.

    Returns
    -------
    list[RGBType] | list[str]
        A list of `n_colors` RGB (scale [0.0, 1.0]) or HEX colours
        interpolated between `rgb1` and `rgb2`:
        [rgb1, ..., rgb2].

    Examples
    --------
    >>> from src.colors import create_gradient

    >>> create_gradient(
    ...     rgb1=(255, 0, 0), rgb2=(0, 0, 255), n_colors=3
    ... )
    [(1.0, 0.0, 0.0), (0.5, 0.0, 0.5), (0.0, 0.0, 1.0)]

    >>> pal = create_gradient(
    ...     rgb1=(1., 0., 0.), rgb2=(0., 0., 1.), n_colors=4
    ... )
    >>> print([tuple(round(value, 2) for value in rgb) for rgb in pal])
    [(1.0, 0.0, 0.0), (0.67, 0.0, 0.33), (0.33, 0.0, 0.67), (0.0, 0.0, 1.0)]

    >>> create_gradient(
    ...     rgb1=(1., 0., 0.), rgb2=(0., 0., 1.), n_colors=4, as_hex=True
    ... )
    ['#ff0000', '#aa0055', '#5500aa', '#0000ff']
    """
    gradient: list[RGBType] = []

    rgb1 = _process_rgb(gradient=[rgb1], scale_mode="compress")[0]
    rgb2 = _process_rgb(gradient=[rgb2], scale_mode="compress")[0]

    r = np.linspace(rgb1[0], rgb2[0], n_colors)
    g = np.linspace(rgb1[1], rgb2[1], n_colors)
    b = np.linspace(rgb1[2], rgb2[2], n_colors)
    gradient = list(zip(r, g, b))

    if as_hex:
        return [rgb_to_hex(rgb) for rgb in gradient]
    else:
        return gradient


def get_palette(
    palette: PaletteType | None = None,
    n_colors: int = 10,
    as_cmap: bool = False
) -> list[RGBType] | ListedColormap | LinearSegmentedColormap:
    """
    Create a sequential color palette with n colors.

    Parameters
    ----------
    palette : PaletteType, default=None
        The name of the palette from which to retrieve a sequence of colours.
        If None, will return the current matplotlib color cycle.
    n_colors : int, default=10
        The number of sequential colours to generate from the given palette.
    as_cmap : bool, default=False
        If True, return a matplotlib.colors.ListedColormap.

    Returns
    -------
    list[RGBType] | ListedColormap
        A list of n sequential RGB 3-tuples containing values in [0.0, 1.0]
        interpolated from `palette`:
        [color1, ..., color2].

    Examples
    --------
    >>> from src.colors import get_palette
    >>> pal = get_palette(palette="Blues", n_colors=3)
    >>> pal = [tuple(round(value, 2) for value in tpl) for tpl in pal]
    >>> print(pal)
    [(0.78, 0.86, 0.94), (0.42, 0.68, 0.84), (0.13, 0.44, 0.71)]
    """
    return sns.color_palette(palette, n_colors=n_colors, as_cmap=as_cmap)


def create_monochromatic_palette(
    hue: float | None = None,
    n_colors: int = 5,
    as_hex: bool = False
) -> list[RGBType] | list[str]:
    """
    Generate a monochromatic palette with `n_colors` starting from a specific
    hue.

    If no hue is provided, a random hue is used. The function generates colors
    by varying the lightness and saturation while keeping the hue constant.

    Parameters
    ----------
    hue : float, default=None
        Starting hue (0-1 scale). If None, a random hue is selected.
    n_colors : int, default=5
        The number of colors to generate in the palette.
    as_hex : bool, default=False
        If True, converts each colour into a HEX representation.

    Returns
    -------
    list[str]
        A list of `n_colors` RGB (scale [0.0, 1.0]) or HEX colours.

    Examples
    --------
    >>> from src.colors import display_palette, create_monochromatic_palette
    >>> pal = create_monochromatic_palette(n_colors=5, hue=.6, as_hex=True)
    >>> print(pal)
    ['#1e437a', '#2d64b7', '#5b8cd6', '#98b7e5', '#d6e2f4']

    >>> pal = create_monochromatic_palette(n_colors=3, hue=.6, as_hex=False)
    >>> pal = [tuple(round(value, 2) for value in tpl) for tpl in pal]
    >>> print(pal)
    [(0.12, 0.26, 0.48), (0.36, 0.55, 0.84), (0.84, 0.89, 0.96)]
    """
    if hue is None:
        hue = random.random()

    palette: list[RGBType] = []

    for i in range(n_colors):
        # Vary lightness between 0.3 and 0.9 (to avoid too dark/light colors);
        # keep saturation constant.
        lightness = 0.3 + (i / (n_colors - 1)) * 0.6
        saturation = 0.6

        # Convert HLS (Hue, Lightness, Saturation) to RGB
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        palette.append(rgb)

    if as_hex:
        return [rgb_to_hex(rgb) for rgb in palette]
    else:
        return palette


def spherical_to_rgb(
    shift: np.ndarray,  # type: ignore[type-arg]
    radius: float,
    theta: float,
    psi: float
) -> RGBType:
    """
    Calculate RGB value from spherical coordinates.

    Parameters
    ----------
    shift : np.ndarray
        The center of the sphere in RGB space.
    radius : float
        Radius of the sphere.
    theta : float
        Polar angle in degrees.
    psi : float
        Azimuthal angle in degrees.

    Returns
    -------
    RGBType
        The calculated RGB value, with components rounded to 3 decimal places.

    Examples
    --------
    >>> from src.colors import spherical_to_rgb
    """
    d2r       = np.pi / 180.
    theta_rad = theta * d2r
    psi_rad   = psi * d2r

    r = radius * np.cos(theta_rad) * np.sin(psi_rad)
    g = radius * np.sin(theta_rad) * np.sin(psi_rad)
    b = radius * np.cos(psi_rad)
    rgb = np.array([r, g, b]) + shift

    return tuple(round(value, 3) for value in rgb)


def get_spherical_palette(n_colors: int) -> list[RGBType]:
    """
    Generate a list of distinct RGB colors by sampling from a sphere
    in RGB space.

    Parameters
    ----------
    n_colors : int
        The number of colors to generate.

    Returns
    -------
    List[RGBType]
        List of `n_colors` RGB colors.

    Notes
    -----
    Colors are generated on the surface of a sphere in 3D RGB space,
    and the colors are shuffled to ensure randomness.

    Examples
    --------
    >>> from src.colors import get_spherical_palette
    """
    color_list: list[RGBType] = []
    np.random.seed(1)

    interval = 90.                      # Initial angular step size
    center = np.array([0.5, 0.5, 0.5])  # Center of the sphere in RGB space
    radius = 0.5

    while len(color_list) < n_colors:
        new_colors = []

        for psi in np.arange(0, 360 + interval, interval):
            for theta in np.arange(0, 360 + interval, interval):
                new_color = spherical_to_rgb(
                    shift=center,
                    radius=radius,
                    theta=theta,
                    psi=psi
                )
                if new_color not in color_list:
                    new_colors.append(new_color)

        # Ensure uniqueness
        new_colors = list(set(new_colors))

        np.random.shuffle(new_colors)
        color_list.extend(new_colors)

        # Decrease interval to refine colors
        interval /= 2

    return color_list[:n_colors]


def get_cubic_palette(n_colors: int) -> list[RGBType]:
    """
    Generate a list of distinct RGB colors by sampling from a cubic grid.

    Parameters
    ----------
    n_colors : int
        The number of colors to generate.

    Returns
    -------
    list[RGBType]
        List of `n_colors` RGB colors.

    Notes
    -----
    This function starts by using 8 predefined colors, then adds more colors
    by iterating over a cubic grid and adjusting the grid resolution.

    Examples
    --------
    >>> from src.colors import get_cubic_palette
    """
    color_list: list[RGBType] = [
        (1., 0., 0.),  # Red
        (0., 1., 0.),  # Green
        (0., 0., 1.),  # Blue
        (1., 1., 0.),  # Yellow
        (0., 1., 1.),  # Cyan
        (1., 0., 1.),  # Magenta
        (0., 0., 0.),  # Black
        (1., 1., 1.)   # White
    ]

    # Add new colours as needed
    np.random.seed(1)

    # Initial grid step size
    interval = 0.5

    while len(color_list) < n_colors:
        new_colors = []

        grid = np.arange(0., 1.0001, interval)
        for i in grid:
            for j in grid:
                for k in grid:
                    new_color = (i, j, k)
                    if new_color not in color_list:
                        new_colors.append(new_color)

        # Ensure uniqueness
        new_colors = list(set(new_colors))

        np.random.shuffle(new_colors)
        color_list.extend(new_colors)

        # Decrease interval to refine colors
        interval /= 2

    return color_list[:n_colors]


def display_palette(
    gradient:
        list[str] | list[RGBType] | ListedColormap | LinearSegmentedColormap
) -> None:
    """
    Display a list of colors as a palette.

    Parameters
    ----------
    gradient : list[RGBType] | ListedColormap | LinearSegmentedColormap
        A list of HEX strings, RGB 3-tuples containing
        values in [0.0, 1.0] or [0, 255] ([rgb1, ..., rgb2]),
        or a ListedColormap or LinearSegmentedColormap object.

    Examples
    --------
    >>> from src.colors import display_palette, get_palette
    >>> from src.colors import create_gradient, create_monochromatic_palette
    >>> from matplotlib.colors import ListedColormap

    >>> display_palette(get_palette(
    ...     palette="viridis", n_colors=5, as_cmap=False
    ... ))
    >>> display_palette(get_palette(
    ...     palette="plasma", as_cmap=True
    ... ))

    >>> display_palette(create_gradient(
    ...     rgb1=(10, 200, 50), rgb2=(200, 90, 155),
    ...     n_colors=10,
    ...     as_hex=False
    ... ))
    >>> display_palette(create_monochromatic_palette(
    ...     n_colors=15,
    ...     hue=.6,
    ...     as_hex=True
    ... ))

    >>> display_palette(get_spherical_palette(n_colors=10))
    >>> display_palette(get_cubic_palette(n_colors=10))

    >>> display_palette([
    ...     darken_color(rgb, factor=0.)
    ...     for rgb in create_gradient((.6, .1, .99), (.1, .75, .1), 10)
    ... ])
    >>> display_palette([
    ...     rgb_to_hex(darken_color(rgb, factor=.5))
    ...     for rgb in create_gradient((.6, .1, .99), (.1, .75, .1), 10)
    ... ])
    """
    if isinstance(gradient, ListedColormap):
        gradient = gradient.colors

    # palplot doesn't handle LinearSegmented cmaps like "Blues" and "PiYG"
    elif isinstance(gradient, LinearSegmentedColormap):
        gradient = [gradient(i / (30 - 1)) for i in range(30)]

    sns.palplot(gradient)
    plt.show()


def _process_rgb(
    gradient: list[RGBType] | RGBType | ListedColormap,
    scale_mode: Literal["compress", "stretch"] = "compress"
) -> list[RGBType]:
    """
    Scale RGB values either from [0, 255] to [0, 1] (compress) or
    from [0, 1] to [0, 255] (stretch).

    Parameters
    ----------
    gradient : list[RGBType] | RGBType
        A tuple representing a single RGB color or a list of RGB colors,
        with all values in either [0, 255] or [0.0, 1.0].
    scale_mode : Literal["compress", "stretch"], default="compress"
        If "compress", scales from [0, 255] to [0.0, 1.0].
        If "stretch", scales from [0.0, 1.0] to [0, 255].

    Returns
    -------
    list[RGBType]
        The processed RGB values as either:
        - integers in the range [0, 255] if stretched;
        - floats in the range [0.0, 1.0] if compressed.

    Raises
    ------
    ValueError
        If RGB values are out of the expected range.

    Examples
    --------
    >>> from src.colors import _process_rgb
    >>> _process_rgb((255, 0, 0), scale_mode="compress")[0]
    (1.0, 0.0, 0.0)
    >>> _process_rgb(
    ...     [(1.0, 0.0, 0.0), (0.0, 0.502, 1.0)], scale_mode="stretch"
    ... )
    [(255, 0, 0), (0, 128, 255)]
    """
    if isinstance(gradient, tuple):
        gradient = [gradient]

    if not all(len(rgb) == 3 for rgb in gradient):
        raise ValueError("RGB tuples must have length 3")

    all_float_compressed = all(
        isinstance(value, float) and 0.0 <= value <= 1.0
        for rgb in gradient
        for value in rgb
    )
    all_int_stretched = all(
        isinstance(value, int) and 0 <= value <= 255
        for rgb in gradient
        for value in rgb
    )

    if scale_mode == "compress":
        if all_float_compressed:
            return gradient
        elif not all_int_stretched:
            raise ValueError("RGB values must be in [0, 255] to compress")
        else:
            return [tuple(val / 255.0 for val in rgb) for rgb in gradient]

    if scale_mode == "stretch":
        if all_int_stretched:
            return gradient
        elif not all_float_compressed:
            raise ValueError("RGB values must be in [0.0, 1.0] to stretch")
        else:
            return [tuple(int(val * 255) for val in rgb) for rgb in gradient]

    raise ValueError("scale_mode must be either 'compress' or 'stretch'.")


def main():
    # Comment out (2) to run all tests in script; (1) to run specific tests
    import doctest
    doctest.testmod(verbose=True)

    # from src.workflow import doctest_function
    # doctest_function(hex_to_rgb, globs=globals())

    # -- One-off tests -------------------------------------------------------

    pass


if __name__ == "__main__":
    # from src.inspection import display
    from src.stylesheet import customize_plots
    customize_plots()

    main()
