import torch
import numpy as np
from PIL import Image

from smoke.config import cfg

from smoke.modeling.heatmap_coder import (
    get_transfrom_matrix
)

from smoke.structures.image_list import to_image_list
from smoke.data.transforms import build_transforms
from smoke.utils.timer import Timer, get_time_str

from smoke.utils.check_point import DetectronCheckpointer
from smoke.engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from smoke.utils import comm
from smoke.modeling.detector import build_detection_model
from smoke.engine.test_net import run_test


class DummyDataset(object):
    def __init__(self, cfg, root):
        super(DummyDataset, self).__init__()
        self.root = root

        self.transforms = build_transforms(cfg, False)

        self.image_files = ['/media/data/test_images/1280x720.jpg']
        self.num_samples = len(self.image_files)
        self.classes = cfg.DATASETS.DETECT_CLASSES

        self.num_classes = len(self.classes)

        self.input_width = cfg.INPUT.WIDTH_TRAIN
        self.input_height = cfg.INPUT.HEIGHT_TRAIN
        self.output_width = self.input_width // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_height = self.input_height // cfg.MODEL.BACKBONE.DOWN_RATIO

    def __len__(self):
        print ("===len", self.num_samples)
        return self.num_samples

    def __getitem__(self, idx):
        print ("===get", idx)

        img = Image.open('/media/data/test_images/1280x720.jpg')

        center = np.array([i / 2 for i in img.size], dtype=np.float32)
        size = np.array([i for i in img.size], dtype=np.float32)

        center_size = [center, size]
        trans_affine = get_transfrom_matrix(
            center_size,
            [self.input_width, self.input_height]
        )
        trans_affine_inv = np.linalg.inv(trans_affine)
        img = img.transform(
                (self.input_width, self.input_height),
                method=Image.AFFINE,
                data=trans_affine_inv.flatten()[:6],
                resample=Image.BILINEAR,
            )

        target = None
        original_idx = 0

        img, target = self.transforms(img, target)

        return img, target, original_idx


def do_test(cfg, model):
    
    dataset = DummyDataset(cfg, '.')

    img = dataset[0][0]
    
    images = to_image_list([img], cfg.DATALOADER.SIZE_DIVISIBILITY)
    output = None
    images = images.to(torch.device('cuda'))
    timer = Timer()
    with torch.no_grad():
        if timer:
            timer.tic()
        output = model(images, None)
        if timer:
            torch.cuda.synchronize()
            timer.toc()
        output = output.to(torch.device("cpu"))

    print ('===output', output)


def setup(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    return cfg

def main(args):
    cfg = setup(args)

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    model.eval() # mark as eval model (training == False)

    checkpointer = DetectronCheckpointer(
        cfg, model, save_dir=cfg.OUTPUT_DIR
    )
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
    return do_test(cfg, model)

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )