from datetime import date

project = "ACCESS-AIS3"
author = "ACCESS-NRI"
copyright = f"{date.today().year}, ACCESS-NRI"
language = "en"

extensions = ["myst_parser"]
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
root_doc = "index"
myst_heading_anchors = 3

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = "ACCESS-AIS3 documentation"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
}