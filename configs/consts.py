import os
import enum


__all__ = ['ProjectPath']


@enum.unique
class ProjectPath(enum.Enum):
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIGS_DIR = os.path.join(ROOT_DIR, 'configs')
    DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets')
    PREVIEW_DIR = os.path.join(ROOT_DIR, 'preview')
    
    CKPTS_DIR = os.path.join(ROOT_DIR, 'ckpts')
    VOC_CKPTS_DIR = os.path.join(CKPTS_DIR, 'voc_ckpts')

    LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
    CONSOLE_LOGS_DIR = os.path.join(LOGS_DIR, 'console_logs')
    TB_LOGS_DIR = os.path.join(LOGS_DIR, 'tb_logs')
    TB_LOGS_TRAIN_DIR = os.path.join(TB_LOGS_DIR, 'train')
    TB_LOGS_VAL_DIR = os.path.join(TB_LOGS_DIR, 'val')
