from torch import optim

from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict

# Commonly changed training configurations
num_epochs = 50   # train epochs
batch_size = 1    # total_batch_size = #GPU x batch_size
num_workers = 4   # workers for pytorch DataLoader
pin_memory = True # whether pin_memory for pytorch DataLoader
print_freq = 50   # frequency to print logs
starting_epoch = 0
max_norm = 0.1    # clip gradient norm

output_dir = None  # path to save checkpoints, default for None: checkpoints/{model_name}
find_unused_parameters = False  # useful for debugging distributed training

# define dataset for train
# coco_path = "data/coco"  # /PATH/TO/YOUR/COCODIR
train_dataset = CocoDetection(
    img_folder="/media/Storage1/wlw/restore/One_Stage_Fetus_Object_Detection_Code_v2/Dataset_Fetus_Object_Detection/Hospital_2/three_vessel_tracheal",
    ann_file="/media/Storage3/wlw/Relation-DETR/datasets/fetus_annotations_coco/3VT/c2/train/annotation.json",
    transforms=presets.detr,  # see transforms/presets to choose a transform
    train=True,
)
test_dataset = CocoDetection(
    img_folder="/media/Storage1/wlw/restore/One_Stage_Fetus_Object_Detection_Code_v2/Dataset_Fetus_Object_Detection/Hospital_2/three_vessel_tracheal",
    ann_file="/media/Storage3/wlw/Relation-DETR/datasets/fetus_annotations_coco/3VT/c2/test/annotation.json",
    transforms=None,  # the eval_transform is integrated in the model
)

# model config to train
model_path = "configs/morph_detr/morph_detr_resnet50_800_1333.py"

# experimtn setting
# det 普通检测
# noisy 噪音标签
# cd 跨域检测
ex_set = "det"

part = "3VT-B"

# model_path = "configs/relation_detr/relation_detr_resnet50_800_1333.py"

# specify a checkpoint folder to resume, or a pretrained ".pth" to finetune, for example:
# checkpoints/relation_detr_resnet50_800_1333/train/2024-03-22-09_38_50
# checkpoints/relation_detr_resnet50_800_1333/train/2024-03-22-09_38_50/best_ap.pth
resume_from_checkpoint = None

learning_rate = 1e-4  # initial learning rate
optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[10], gamma=0.1)

# This define parameter groups with different learning rate
param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)
