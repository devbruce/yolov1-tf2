__all__ = ['Configs']


class Configs:
    def __init__(self):
        self.num_classes = 20      # Number of Classes (Default: PascalVOC2012 Dataset --> 20)
        self.cell_size = 7         # Paper Default: 7
        self.boxes_per_cell = 2    # Paper Default: 2
        self.input_width = 448     # Paper Default: 448
        self.input_height = 448    # Paper Default: 448
        self.eps = 1e-6

        # Loss coefficients (Lambda coefficients of paper)
        self.lambda_coord = 5      # Paper Default: 5
        self.lambda_noobj = 0.5    # Paper Default: 0.5

        # Custom lambda coefficients. (It is not mentioned in the paper)
        self.lambda_obj = 1        # Paper Default: 1
        self.lambda_class = 1      # Paper Default: 1

        # Train
        self.epochs = 105          # Paper Default: 135 (75 (lr: 1e-2) + 30 (lr: 1e-3) + 30 (lr: 1e-4))
        self.init_lr = 1e-4        # Paper Default: 1e-2
        self.lr_decay_rate = 0.5
        self.lr_decay_steps = 40000
        self.batch_size = 32
        self.val_step = 1
        self.tb_img_max_outputs = 6
        
        # Box postprocess parameters
        self.nms_iou_thr = 0.5
        self.conf_thr = 0.5  # Used visualization

        # Dataset sampling
        self.train_ds_sample_ratio = 1.   # Use all training set
        self.val_ds_sample_ratio = 1.     # Use all validation set
