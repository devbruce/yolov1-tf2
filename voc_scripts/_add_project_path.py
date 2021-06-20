import os
import sys


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(CURRENT_PATH)

sys.path.insert(0, PROJECT_ROOT_DIR)
