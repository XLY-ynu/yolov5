# import torch
#
# # print(torch.__version__)
# repo_dir = r'C:\Users\29138\Downloads\yolov5-6.1'
# model_path = r'C:\Users\29138\Downloads\yolov5-6.1\yolov5n._custom.pt'
#
# model = torch.hub.load(repo_dir, 'custom', path=model_path, source='local')
#
# # Images
# img = r"D:\学习资料\新建文件夹 (6)\yolov5\data\images\data1.jpg"  # or file, Path, PIL, OpenCV, numpy, list
#
# # Inference
# results = model(img)
#
# # Results
# results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
# results.show()


from openvino.inference_engine import IECore

ie = IECore()
net = ie.read_network(model=r'C:\Users\29138\Downloads\yolov5-6.1\openvino_2021\yolov5n.xml', weights=r'C:\Users\29138\Downloads\yolov5-6.1\openvino_2021\yolov5n.bin')
print(net.outputs.keys())