from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, resize, to_tensor
from torchvision.transforms import ColorJitter
import matplotlib.pyplot as plt
import torch
import os 
import pickle
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image



# Step 1: Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.eval().cuda()

# from models.yolov3 import load_model
# model = load_model('models/model_confs/yolov3.conf',
#                     'models/state_dicts/yolov3.weights').eval().cuda()

t = ColorJitter(contrast=[0.5,0.5])


true_label = 'car'
# pictures_dir = 'results/'
clean_dir = 'photos/clean_car_yolo'
# adv_dir = 'photos/2dappleours_apple_frcnn'
adv_dir = clean_dir.replace("clean",'atk') 
save_dir = adv_dir+'_eval'
n = 0
i=0
total = len(os.listdir(clean_dir))
metric = MeanAveragePrecision()




with torch.no_grad():
    for forder_name in os.listdir(clean_dir):
        for di in os.listdir(os.path.join(clean_dir,forder_name)):
            if '_.png' in di or '_.jpg' in di:
                try:
                    
                    print(di)
                    
                    # load pickle and image
                    pickle_name = di.replace('_.jpg', ".pkl")
                    pickle_name = pickle_name.replace('_.png', ".pkl")
                    with open(os.path.join(clean_dir,forder_name,pickle_name) ,"rb") as fr:
                        clean_data = pickle.load(fr)
                        if len(clean_data["scores"]) == 0:
                            clean_data["labels"] = []       
                    try:
                        with open(os.path.join(adv_dir,forder_name,pickle_name) ,"rb") as fr:
                            adv_data = pickle.load(fr)
                            if len(adv_data["scores"]) == 0:
                                adv_data["labels"] = []
                    except:
                        continue
                            
                    #load image, font
                    adv_img = read_image(os.path.join(adv_dir,forder_name,di)).cuda()[:3,:,:]
                    preprocess = weights.transforms()
                    # adv_img = resize(adv_img,512,)/255
                    adv_img = preprocess(adv_img).cuda()
                    batch = adv_img[None,...]
                    labels = [weights.meta["categories"][i] for i in clean_data["labels"]]
                    n+=1
                    if true_label in labels:
                        i+=1
                    
                    adv_img = adv_img*255
                    adv_img = adv_img.to(torch.uint8)
                    font = 'Ubuntu-R.ttf'

                    # choose only car label
                    if labels.index(true_label) is not None:
                        int = labels.index(true_label)
                        boxes=torch.tensor(clean_data['boxes'])[int].unsqueeze(0)
                        label = []
                        label.append(labels[int])
                    box = draw_bounding_boxes(adv_img, boxes=torch.tensor(boxes),
                                            labels=labels,
                                            colors=(0,0,0),
                                            width=2, font=font, font_size=50)
                    box = box.permute(1,2,0).cpu().numpy()
                    
                    if not os.path.isdir(os.path.join(save_dir,forder_name)):
                        os.makedirs(os.path.join(save_dir,forder_name))
                    
                    box = Image.fromarray(box)
                    # plt.imshow(box)
                    box.save(os.path.join(save_dir,forder_name,di))
                    # plt.show()
                    # plt.savefig(os.path.join(save_dir,forder_name,di))
                    print("")
                    
                    clean = [dict(
                            boxes=torch.tensor(clean_data['boxes']),
                            scores=torch.tensor(clean_data['scores']),
                            labels=torch.tensor(clean_data['labels']),
                            )]
                    
                    adv = [dict(
                            boxes=torch.tensor(adv_data['boxes']),
                            scores=torch.tensor(adv_data['scores']),
                            labels=torch.tensor(adv_data['labels']),
                            )]
                    
                    metric.update(adv, clean)
                    # from pprint import pprint
                    # print(metric.compute())
                    import sys
                    # text_name = pickle_name = di.replace('_.jpg', ".txt")
                    with open(os.path.join(save_dir,'map.txt'), 'w') as f:
                        print(metric.compute(), file = f)
                        print(i/n*100, file = f)
                except Exception as e:
                    print(e)
                

            
            
            
            
    
            # # Step 2: Initialize the inference transforms
            # img = read_image(os.path.join(clean_dir,di)).cuda()[:3,:,:]
            # # img = t(img)
            # # img = img[:,:512,:512]
            
            # print(img.shape)
            # preprocess = weights.transforms()

            # img = resize(img,512,)/255
            # # img = img.rot90(dims=[2,1])
            
            # if i ==1:
            #     plt.imshow(img.permute(1,2,0).detach().cpu().numpy())
            #     plt.show()
            # # img = to_tensor
            # img = preprocess(img).cuda()
            
            # # Step 3: Apply inference preprocessing transforms
            # batch = img[None,...]
            
            
            # # Step 4: Use the model and visualize the prediction
            # prediction = model(batch)[0]
            # labels = [weights.meta["categories"][i] for i in prediction["labels"]]
            
            # if label in labels:
            #     n+=1
            # # labels = [labels[0]]
            # # prediction['boxes'] = prediction['boxes'][0]
            # ######################### no car detection
            # # l = []
            # # p = []
            # # for i in range(len(labels)):
            # #     if labels[i] != 'car':
            # #         l.append(labels[i])
            # #         p.append(prediction['boxes'][i][None,...])
            # # prediction['boxes'] = torch.concat(p, dim=0)
            # # labels = l
            # ############################
            
            # img = img*255
            # img = img.to(torch.uint8)
            # color='red'
            # if labels[0] == "apple":
            #     color=(0,255,0)
            # else:
            #     color='red'
            # box = draw_bounding_boxes(img, boxes=prediction['boxes'],
            #                         labels=labels,
            #                         colors=color,
            #                         width=4, font_size=5000000)
            # print(labels)
            # box = box.permute(1,2,0).cpu().numpy()
            # plt.imshow(box)
            # plt.show()
            # plt.savefig(f"apple_{i}.png")