import os

# BASE PATH: "src"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

from pathlib import Path
PROJECT_DIR = str(Path(ROOT_DIR).parents[0])
RES_DIR = os.path.join(PROJECT_DIR, 'results')