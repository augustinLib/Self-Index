import os

from selfindex.runtime.platform.environment import EXTERNALS_ROOT

BROWSECOMP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDEXES = os.path.join(BROWSECOMP_DIR, "indexes")
UPSTREAM = os.path.join(EXTERNALS_ROOT, "BrowseComp-Plus")
