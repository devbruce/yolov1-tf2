import os
import pytz
import datetime
import logging
from termcolor import colored
from configs import ProjectPath


__all__ = ['get_current_time_str', 'train_step_console_log', 'get_logger']


def get_current_time_str():
    time_now = datetime.datetime.now(tz=pytz.timezone('Asia/Seoul'))
    time_now_strf = time_now.strftime('%Y%m%d%H%M%S')
    return time_now_strf


def train_step_console_log(total_epoch, total_step, current_epoch, current_step, losses):
    progress = colored(f'* Epoch: {current_epoch:^4} / {total_epoch:^4} | Step: {current_step:^4} / {total_step:^4}', 'green')
    current_time = colored(f' | Current Time: {get_current_time_str()}', 'blue')
    total_loss = f'>>> Total Loss: {losses["total_loss"]:<8.4f}'
    total_loss = colored(total_loss, 'red')
    loss_info = ' (coord: {:<8.4f}, obj: {:<8.4f}, noobj: {:<8.4f}, class: {:<8.4f})'
    loss_info = colored(loss_info.format(losses['coord_loss'], losses['obj_loss'], losses['noobj_loss'], losses['class_loss']), 'cyan')
    log = '\n' + progress + current_time + '\n' + total_loss + loss_info
    return log


def get_logger(fpath=None):
    if fpath:
        log_fpath = fpath
    else:
        log_fpath = os.path.join(ProjectPath.CONSOLE_LOGS_DIR.value, get_current_time_str()+'.log')
    logger = logging.getLogger(name='yolo_logger')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(log_fpath))
    return logger
