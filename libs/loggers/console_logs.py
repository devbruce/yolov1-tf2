import os
import pytz
import datetime
import logging
from termcolor import colored
from configs import ProjectPath


__all__ = ['get_current_time_str', 'train_step_console_log', 'get_logger']


def get_current_time_str(tz=pytz.timezone('Asia/Seoul'), format='%Y%m%d%H%M%S'):
    time_now = datetime.datetime.now(tz=tz)
    time_now_strf = time_now.strftime(format)
    return time_now_strf


def train_step_console_log(total_epoch, total_step, current_epoch, current_step, losses):
    progress = f'* Epoch: {current_epoch:^4} / {total_epoch:^4} | Step: {current_step:^4} / {total_step:^4}'
    current_time = f' | Current Time: {get_current_time_str()}'
    total_loss = f'>>> Total Loss: {losses["total_loss"]:<8.4f}'
    loss_info_form = ' (coord: {:<8.4f}, obj: {:<8.4f}, noobj: {:<8.4f}, class: {:<8.4f})'
    loss_info = loss_info_form.format(losses['coord_loss'], losses['obj_loss'], losses['noobj_loss'], losses['class_loss'])
    log = '\n' + progress + current_time + '\n' + total_loss + loss_info

    progress_colored = colored(progress, 'green')
    current_time_colored = colored(current_time, 'blue')
    total_loss_colored = colored(total_loss, 'red')
    loss_info_colored = colored(loss_info, 'cyan')
    log_colored = '\n' + progress_colored + current_time_colored + '\n' + total_loss_colored + loss_info_colored
    return log, log_colored


def get_logger(name='YOLO_Logger', fpath=None):
    if fpath:
        log_fpath = fpath
    else:
        log_fpath = os.path.join(ProjectPath.CONSOLE_LOGS_DIR.value, get_current_time_str()+'.log')
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    # logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(log_fpath))
    return logger
