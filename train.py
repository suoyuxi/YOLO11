# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO, convert_dota_to_yolo_obb

if __name__ == '__main__':

    convert_dota_to_yolo_obb('/data1/suoyuxi/FAIR-CSAR/DOTA_SL/','SL') # 这个函数在/ultralytics/data/converter.py x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult -> class_id x y w h
    model = YOLO(model='/data1/suoyuxi/ultralytics/workdir/SL_CFG/yolo11-obb.yaml')
    model.train(data='/data1/suoyuxi/ultralytics/workdir/SL_CFG/dota_sl.yaml',
                imgsz=1024,
                epochs=50,
                batch=4,
                workers=0,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='/data1/suoyuxi/ultralytics/workdir',
                name='SL',
                single_cls=False,
                cache=False,
                )
    # model.load()
    model.val(data='/data1/suoyuxi/ultralytics/workdir/SL_CFG/dota_sl.yaml')

    convert_dota_to_yolo_obb('/data1/suoyuxi/FAIR-CSAR/DOTA_FSI/','FSI') # 这个函数在/ultralytics/data/converter.py x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult -> class_id x y w h
    model = YOLO(model='/data1/suoyuxi/ultralytics/workdir/FSI_CFG/yolo11n-obb.yaml')
    model.train(data='/data1/suoyuxi/ultralytics/workdir/FSI_CFG/dota_fsi.yaml',
                lr0=0.05,
                imgsz=1024,
                epochs=50,
                batch=4,
                workers=0,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='/data1/suoyuxi/ultralytics/workdir',
                name='FSI',
                single_cls=False,
                cache=False,
                )
    model.load('/data1/suoyuxi/ultralytics/workdir/FSI/weights/best.pt')
    model.val(data='/data1/suoyuxi/ultralytics/workdir/FSI_CFG/dota_fsi.yaml')
    