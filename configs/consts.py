import os
import enum


__all__ = ['ProjectPath']


@enum.unique
class ProjectPath(enum.Enum):
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIGS_DIR = os.path.join(ROOT_DIR, 'configs')
    DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets')
    