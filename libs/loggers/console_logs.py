import os
import pytz
import datetime
import logging
from termcolor import colored
from configs import ProjectPath


__all__ = ['get_current_time_str', 'get_logger', 'train_step_console_log', 'val_console_log']


def get_current_time_str():
    time_now = datetime.datetime.now(tz=pytz.timezone('Asia/Seoul'))
    time_now_strf = time_now.strftime('%Y%m%d%H%M%S')
    return time_now_strf


def get_logger(fpath=None):
    if fpath:
        log_fpath = fpath
    else:
        log_fpath = os.path.join(ProjectPath.CONSOLE_LOGS_DIR.value, get_current_time_str()+'.log')
    logger = logging.getLogger(name='yolo_logger')
    logger.setLevel(logging.INFO)
    # logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(log_fpath))
    return logger


def train_step_console_log(total_epochs, steps_per_epoch, current_epoch, current_step, losses):
    progress = f'* Epoch: {current_epoch:^4} / {total_epochs:^4} | Step: {current_step:^4} / {steps_per_epoch:^4}'
    current_time = f' | Current Time: {get_current_time_str()}'
    total_loss = f'>>> Total Loss: {losses["total_loss"]:<8.4f}'
    loss_info_form = ' (coord: {:<8.4f}, obj: {:<8.4f}, noobj: {:<8.4f}, class: {:<8.4f})'
    loss_info = loss_info_form.format(
        losses['coord_loss'],
        losses['obj_loss'],
        losses['noobj_loss'],
        losses['class_loss'],
        )
    log = '\n' + progress + current_time + '\n' + total_loss + loss_info

    progress_colored = colored(progress, 'green')
    current_time_colored = colored(current_time, 'blue')
    total_loss_colored = colored(total_loss, 'red')
    loss_info_colored = colored(loss_info, 'cyan')
    log_colored = '\n' + progress_colored + current_time_colored + '\n' + total_loss_colored + loss_info_colored
    return log, log_colored


def val_console_log(total_epochs, current_epoch, losses, APs):
    progress = f'[Validation] * Epoch: {current_epoch:^4} / {total_epochs:^4}'
    current_time = f' | Current Time: {get_current_time_str()}'
    total_loss = f'>>> Total Loss: {losses["total_loss"]:<8.4f}'
    loss_info_form = ' (coord: {:<8.4f}, obj: {:<8.4f}, noobj: {:<8.4f}, class: {:<8.4f})'
    loss_info = loss_info_form.format(
        losses['coord_loss'],
        losses['obj_loss'],
        losses['noobj_loss'],
        losses['class_loss'],
        )
    APs = APs.copy()
    mAP = APs.pop('mAP')
    APs_log = '\n====== mAP ======\n' + f'* mAP: {mAP:<8.4f}\n'
    for cls_name, ap in APs.items():
        APs_log += f'- {cls_name}: {ap:<8.4f}\n'
    APs_log += '====== ====== ======\n'
    log = '\n' + progress + current_time + '\n' + total_loss + loss_info + '\n' + APs_log

    progress_colored = colored(progress, 'green')
    current_time_colored = colored(current_time, 'blue')
    total_loss_colored = colored(total_loss, 'red')
    loss_info_colored = colored(loss_info, 'cyan')
    APs_log_colored = colored(APs_log, 'magenta')
    log_colored = '\n' + progress_colored + current_time_colored + '\n' + total_loss_colored + loss_info_colored + '\n' + APs_log_colored
    return log, log_colored
