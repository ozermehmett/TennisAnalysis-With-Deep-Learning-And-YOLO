from ultralytics import YOLO
# import torch

# print(torch.cuda.is_available())

# model = YOLO('models/yolo5_last.pt')
model = YOLO('yolov8x')
model.to('cuda')

result = model.predict('input_videos/input_video.mp4', conf=0.2, save=True)
#
# print(result)
# print('Boxes:')
# for box in result[0].boxes:
#     print(box)
