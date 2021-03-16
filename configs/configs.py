__all__ = ['Configs']


class Configs:
    def __init__(self):
        self.num_classes = 20
        self.cell_size = 7
        self.boxes_per_cell = 2
        self.input_width = 448
        self.input_height = 448
        self.eps = 1e-6

        # Loss Coefficient (Lambda coefficients of paper)
        self.lambda_coord = 5     # Paper Default: 5
        self.lambda_noobj = 0.5   # Paper Default: 0.5

        # Loss Coefficient (Custom lambda coefficients. It is not mentioned in the paper)
        self.lambda_obj = 1       # Paper Default: 1
        self.lambda_class = 1     # Paper Default: 1

        # Train
        self.epochs = 200
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.val_step = 1
        self.tb_img_max_outputs=6
        
        self.nms_iou_thr = 0.7
        self.conf_thr = 0.5
