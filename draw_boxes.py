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
from nvdiffrec.atk.utils_initiate import initiate_model


# Step 1: Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.eval().cuda()

# from models.yolov3 import load_model
# model = load_model('models/model_confs/yolov3.conf',
#                     'models/state_dicts/yolov3.weights').eval().cuda()

t = ColorJitter(contrast=[0.5,0.5])





weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
frcnn = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.75, rpn_post_nms_top_n_test=512, download_file=True).eval().cuda()


objects = ['car', 'apple', 'cup']
which = ['clean']

# objects = ['car']
# which = ['atk']



for o in objects:
    for w in which:
        true_label = o
        # pictures_dir = 'results/'
        # clean_dir = 'photos/clean_car_yolo'
        # adv_dir = 'photos/2dappleours_apple_frcnn'
        adv_dir = 'photos/{}_{}_frcnn'.format(w,o)
        print(adv_dir)
        save_dir = adv_dir+'_eval'
        n = 0
        i=0
        total = len(os.listdir(adv_dir))
        metric = MeanAveragePrecision()
        with torch.no_grad():
            for forder_name in os.listdir(adv_dir):
                for di in os.listdir(os.path.join(adv_dir,forder_name)):
                    if '.png' in di or '.jpg' in di:
                        if di.endswith('_.jpg'):
                            pass
                        else:
                            
                            try:
                                
                                print(forder_name, di)
                                
                                # load pickle and image
                                pickle_name = di.replace('.jpg', ".pkl")
                                pickle_name = pickle_name.replace('.png', ".pkl")
                                # with open(os.path.join(clean_dir,forder_name,pickle_name) ,"rb") as fr:
                                #     clean_data = pickle.load(fr)
                                #     if len(clean_data["scores"]) == 0:
                                #         clean_data["labels"] = []       
                                # try:
                                #     with open(os.path.join(adv_dir,forder_name,pickle_name) ,"rb") as fr:
                                #         adv_data = pickle.load(fr)
                                #         if len(adv_data["scores"]) == 0:
                                #             adv_data["labels"] = []
                                # except:
                                #     continue
                                        
                                #load image, font
                                adv_img = Image.open(os.path.join(adv_dir,forder_name,di))
                                # print(adv_img.shape)
                                preprocess = weights.transforms()
                                # adv_img = resize(adv_img,512,)/255
                                adv_img = preprocess(adv_img).cuda()
                                batch = adv_img[None,...]
                                adv_data = frcnn(batch)[0]
                                
                                boxes = adv_data["boxes"]
                                labels = []
                                b = []
                                boxes_ = torch.tensor([])
                                ban = ['vase', 'laptop','sofa', 'dining table', 'tv', 'couch', 'chair', 'surfboard', 'potted plant']
                                # ban = []
                                try: 
                                    # labels.index(true_label) is not None
                                    t = labels.index(true_label)
                                    
                                    
                                    labels.append(labels[t])
                                    boxes_ = torch.tensor(adv_data['boxes'])[t].unsqueeze(0)
                                except:
                                    for n, i in enumerate(adv_data["labels"]):
                                        k = i+1 if 'yolo' in adv_dir else i
                                        if not weights.meta["categories"][k] in ban:
                                            
                                            labels.append(weights.meta["categories"][k])
                                            b.append(adv_data["boxes"][n][None,...])
                                            boxes_ = torch.concat(b, dim=0)
                                            
                                            # break
                                
                                # labels = [weights.meta["categories"][i] for i in adv_data["labels"]]
                                
                                
                                n+=1
                                if true_label in labels:
                                    i+=1
                                
                                adv_img = adv_img*255
                                adv_img = adv_img.to(torch.uint8)
                                font = 'Ubuntu-R.ttf'
                                # boxes = torch.tensor(adv_data['boxes'])[int].unsqueeze(0)
                                # choose only car label
                                
                                
                                c = (255,0,0) if 'atk' in adv_dir else (0,255,0) 
                                
                                box = draw_bounding_boxes(adv_img, boxes=boxes_,
                                                        labels=labels,
                                                        colors= c,
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
                                
                                # clean = [dict(
                                #         boxes=torch.tensor(clean_data['boxes']),
                                #         scores=torch.tensor(clean_data['scores']),
                                #         labels=torch.tensor(clean_data['labels']),
                                #         )]
                                
                                # adv = [dict(
                                #         boxes=torch.tensor(adv_data['boxes']),
                                #         scores=torch.tensor(adv_data['scores']),
                                #         labels=torch.tensor(adv_data['labels']),
                                #         )]
                                
                                # metric.update(adv, clean)
                                # from pprint import pprint
                                # print(metric.compute())
                                # import sys
                                # text_name = pickle_name = di.replace('_.jpg', ".txt")
                                # with open(os.path.join(save_dir,'map.txt'), 'w') as f:
                                #     print(metric.compute(), file = f)
                                #     print(i/n*100, file = f)
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