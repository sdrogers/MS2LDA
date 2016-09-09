"""
LDAvis URLs
==========
URLs and filepaths for the LDAvis javascript libraries
NOT the same as the original pyLDAvis -- modified for MS2LDAvis!
"""

import os
from . import __path__, __version__
import warnings

__all__ = ["D3_URL", "LDAVIS_URL", "LDAVISMIN_URL", "LDAVIS_CSS_URL",
           "D3_LOCAL", "LDAVIS_LOCAL", "LDAVISMIN_LOCAL", "LDAVIS_CSS_LOCAL",
           "LDAVIS_GRAPH_LOCAL", "LDAVIS_GRAPH_JSON"]

D3_URL = "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"

# DEV = 'git' in __version__
DEV = True

LOCAL_JS_DIR = os.path.join(__path__[0], "js")
D3_LOCAL = os.path.join(LOCAL_JS_DIR, "d3.v3.min.js")
JQUERY_LOCAL = os.path.join(LOCAL_JS_DIR, "jquery-2.2.2.min.js")
JQUERY_UI_JS_LOCAL = os.path.join(LOCAL_JS_DIR, "jquery-ui.min.js")
JQUERY_UI_CSS_LOCAL = os.path.join(LOCAL_JS_DIR, "jquery-ui.css")
D3_TIP_LOCAL = os.path.join(LOCAL_JS_DIR, "d3.tip.v0.6.3.js")

LOCAL_IMAGE_DIR = os.path.join(__path__[0], "images")
DEFAULT_LOGO_LOCAL = os.path.join(LOCAL_IMAGE_DIR, "placeholder3.png")
DEFAULT_SHOW_GRAPH_LOCAL = os.path.join(LOCAL_IMAGE_DIR, "show_graph2.jpg")

if DEV:
    LDAVIS_URL = os.path.join(LOCAL_JS_DIR, "ldavis.js")
    LDAVIS_CSS_URL = os.path.join(LOCAL_JS_DIR, "ldavis.css")
    LDAVIS_LOCAL = os.path.join(LOCAL_JS_DIR, "ldavis.js")
    LDAVIS_CSS_LOCAL = os.path.join(LOCAL_JS_DIR, "ldavis.css")
    LDAVIS_GRAPH_LOCAL = os.path.join(LOCAL_JS_DIR, "graph.html")
    LDAVIS_GRAPH_JSON = os.path.join(LOCAL_JS_DIR, "graph.json")
else:
#     WWW_JS_DIR = "https://cdn.rawgit.com/bmabey/pyLDAvis/files/"
#     JS_VERSION = '1.0.0'
#     CSS_VERSION = '1.0.0'
#     LDAVIS_URL = WWW_JS_DIR + "ldavis.v{0}.js".format(JS_VERSION)
#     LDAVIS_CSS_URL = WWW_JS_DIR + "ldavis.v{0}.css".format(CSS_VERSION)
#     LDAVIS_LOCAL = os.path.join(LOCAL_JS_DIR,
#                            "ldavis.v{0}.js".format(JS_VERSION))
#     LDAVIS_CSS_LOCAL = os.path.join(LOCAL_JS_DIR,
#                            "ldavis.v{0}.css".format(CSS_VERSION))
    LDAVIS_URL = os.path.join(LOCAL_JS_DIR, "ldavis.js")
    LDAVIS_CSS_URL = os.path.join(LOCAL_JS_DIR, "ldavis.css")
    LDAVIS_LOCAL = os.path.join(LOCAL_JS_DIR, "ldavis.js")
    LDAVIS_CSS_LOCAL = os.path.join(LOCAL_JS_DIR, "ldavis.css")
    LDAVIS_GRAPH_LOCAL = os.path.join(LOCAL_JS_DIR, "graph.html")
    LDAVIS_GRAPH_JSON = os.path.join(LOCAL_JS_DIR, "graph.json")

LDAVISMIN_URL = LDAVIS_URL
LDAVISMIN_LOCAL = LDAVIS_LOCAL
