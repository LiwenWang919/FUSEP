#coding=utf-8
 
from mmdetection.mmdet.apis import init_detector
from mmdetection.mmdet.apis import inference_detector
# from mmdet.apis import show_result
 
# 模型配置文件
config_file = '/media/Storage1/wlw/Semi/SSOD/configs/labelmatch/labelmatch_standard_4-3.py'
 
# 预训练模型文件
checkpoint_file = '/media/Storage1/wlw/Semi/SSOD/work_dirs/labelmatch_standard_4-3/latest.pth'
 
# 通过模型配置文件与预训练文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')
 
# 测试单张图片并进行展示
img = '/media/Storage1/wlw/Semi/SSOD/Dataset_Fetus_Object_Detection/Hospital_1/four_chamber_heart/1.2.156.112601.1.4.960051513.2913.1645690308.229899.jpg'
result = inference_detector(model, img)
model.show_result(img, result, model.CLASSES)