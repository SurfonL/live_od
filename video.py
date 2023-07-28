import cv2
import torchvision.transforms as T
from nvdiffrec.atk.utils_initiate import initiate_model, Label2Word
import numpy as np
from torchvision.utils import draw_bounding_boxes
import torch
import matplotlib.pyplot as plt


class Argument:
    def __init__(self, value):
        self.value = value

def scalef(ori_image, args):
    X = args.cfg['img_scale'][0]
    h,w,c = ori_image.shape[-3:]
    # Compute the new size of the image while maintaining its aspect ratio
    aspect_ratio = w / h
    new_h = int(X / aspect_ratio)
    new_w = int(X * aspect_ratio)
    size = (new_w, new_h)

    # Resize the image using PyTorch's resize function and compute the scale factor
    
    
    scale_factor = np.array([new_w / w, new_h / h, new_w / w, new_h / h], dtype=np.float32)
    
    # scale_factor = torch.tensor([new_w / w, new_h / h, new_w / w, new_h / h])
    # scale_factor = scale_factor.cuda() if args.model_name == 'frcnn' else scale_factor
    return scale_factor
def process_det_output(output):
    #'output:List[List(80)[Tensor(n,5)]]'
    # only batch size 1
    
    bbox_result = output[0]
    if len(bbox_result) < 80:
        bbox_result=bbox_result[0]
    labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(bbox_result)    
    scores = bboxes[...,-1]
    bboxes = bboxes[...,:-1]
    idx = scores>0.5
    
    all_preds = {'boxes': bboxes, 'labels': labels, 'scores': scores}

    bboxes = torch.tensor(bboxes[idx, :])
    labels = torch.tensor(labels[idx])
    scores = torch.tensor(scores[idx])
    return [{'boxes': bboxes, 'labels': labels, 'scores': scores, 'all_preds':all_preds}]

cap = cv2.VideoCapture('testimgs/KakaoTalk_20230308_193352849.mp4')
count=0


args = Argument(0)
args.model_name = 'yolof'
model = initiate_model(args)
        # if args.model_name == 'crcnn' or 'yolof':
    #     model.to('cuda:1')
transform = T.Compose([
    T.ToTensor(),
    T.Resize(args.cfg['img_scale']),
    T.Normalize(args.cfg['img_norm_cfg']['mean'], args.cfg['img_norm_cfg']['std']),
])
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30, (1080, 1080))
if out is None:
    print('Failed to create video writer')
# Loop over the frames of the video
while cap.isOpened():
    
    
    
    # Read the next frame from the video
    ret, img = cap.read()
    original_label = ''
    label_names = Label2Word('frcnn')

    font = 'Ubuntu-R.ttf'

    if not ret:             #ret이 False면 중지
        break

    

    net = model
    # img = cv2.imread("testimgs/KakaoTalk_20230308_193201093.jpg")

    image = img
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    h,w,c = image.shape
    size = min(h, w)
    image = cv2.getRectSubPix(image, (size, size), (w // 2, h // 2))
    image_t = transform(image).cuda()[None,...]


    scale_factor = scalef(image,args)
    imsize = np.array((*image_t.shape[-2:],3))
    img_metas = {'ori_shape': image.shape,
            'img_shape': imsize,
            'pad_shape': imsize,
            'img_norm_cfg': args.cfg['img_norm_cfg'],
            'scale_factor': scale_factor,
            }


    with torch.no_grad():
        detections = net.forward_test([image_t],[[img_metas]],rescale=True)
    detections = process_det_output(detections)
    # print(detections)

    image = torch.from_numpy(image).permute(2,0,1)
    # detections[0]['labels'] +=1
    labels = [str(label_names.id_to_label(i)) for i in detections[0]['labels']]


    det = {'labels':[], 'boxes':[], 'scores':[]}
    if not original_label == "":
        try:
            obj_idx = labels.index(original_label)
            l = ['{} {}'.format(labels[obj_idx], int(round(detections[0]["scores"][obj_idx].item(),2)*100))]
            b = detections[0]['boxes'][obj_idx][None,...]
            det['labels']=[detections[0]['labels'][obj_idx].item()]
            det['boxes']=b
            det['scores'] = [detections[0]['scores'][obj_idx].item()]
        except ValueError:
            l = torch.tensor([])
            b = torch.tensor([])
    else:
        l = labels
        b = detections[0]['boxes']
        
    c,h,w = image.shape
    color = (255,0,0) if original_label in labels else (0,255,0)

    font_size = int(w/20)
    box = draw_bounding_boxes(image, b ,l, colors = color,font=font, font_size=font_size, width=4).permute(1,2,0)
    
    box = box.cpu().numpy()
    box = cv2.cvtColor(box,cv2.COLOR_RGB2BGR)
    
    out.write(box)
    # print(box.shape, box)
    cv2.imshow('Frame', box)

    # Wait for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # plt.imshow(box)

    # plt.show()
cap.release()
out.release()
cv2.destroyAllWindows()
print('released')