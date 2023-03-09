import asyncio
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional
import argparse
import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydub
import streamlit as st
# from models.yolov3 import load_model
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from aiortc.contrib.media import MediaPlayer
from streamlit_webrtc import (RTCConfiguration, WebRtcMode,
                              WebRtcStreamerContext, webrtc_streamer)

import time
from torchvision.utils import draw_bounding_boxes
from pathlib import Path
import copy
from mmdet.core.post_processing import bbox_nms
from PIL import Image
import datetime
import pickle
import os
import glob
import time

from nvdiffrec.atk.utils_initiate import initiate_model, Label2Word


# from session import SessionState

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

now = datetime.datetime.now()
# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)



def main():
    st.header("RaS: Reconstruct-and-Shoot ")

    pages = {
        "live object detection": app_object_detection
    }
    
        # "Real time video transform with simple OpenCV filters (sendrecv)": app_video_filters,  # noqa: E501
        # "Real time audio filter (sendrecv)": app_audio_filter,
        # "Delayed echo (sendrecv)": app_delayed_echo,
        # "Consuming media files on server-side and streaming it to browser (recvonly)": app_streaming,  # noqa: E501
        # "WebRTC is sendonly and images are shown via st.image() (sendonly)": app_sendonly_video,  # noqa: E501
        # "WebRTC is sendonly and audio frames are visualized with matplotlib (sendonly)": app_sendonly_audio,  # noqa: E501
        # "Simple video and audio loopback (sendrecv)": app_loopback,
        # "Configure media constraints and HTML element styles with loopback (sendrecv)": app_media_constraints,  # noqa: E501
        # "Control the playing state programatically": app_programatically_play,
        # "Customize UI texts": app_customize_ui_texts,
    page_titles = pages.keys()

    page_title = st.sidebar.selectbox(
        "Choose the app mode",
        page_titles,
    )
    st.subheader(page_title)

    page_func = pages[page_title]
    page_func()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")


# @st.cache
def init_run(args):
    model = initiate_model(args)
    # if args.model_name == 'crcnn' or 'yolof':
    #     model.to('cuda:1')
    transform = T.Compose([
        T.ToTensor(),
        T.Resize(args.cfg['img_scale']),
        T.Normalize(args.cfg['img_norm_cfg']['mean'], args.cfg['img_norm_cfg']['std']),
        
    ])
    return  model, transform

global save_, retake_, hor
save_ = threading.Event()
retake_ = threading.Event()
# saved_ = threading.Event()
lock = threading.Lock()
detection_container = {'imgs':None, 'boxes':0, 'det_results':0}


class SaveData:
    def __init__(self) -> None:
        self.to_be_saved = {'imgs':[i for i in range(12)], 'boxed':[i for i in range(12)], 'det_results': [i for i in range(12)]}
        self.do_set_idx = False
        self.curr_idx =0
        self.disp='curr_idx: {}'.format(self.curr_idx)
    
    def display(self):
        return self.disp
    
    def insert_img_data(self,container,n):
        if n < 12:
            for key, value in self.to_be_saved.items():
                self.to_be_saved[key][n] =container[key]
            self.disp ='curr_idx: {}'.format(self.curr_idx)
            self.curr_idx = n+1
            
        if self.curr_idx >= 12:
            self.curr_idx=11
            self.disp ='maxed out curr_idx: {}'.format(self.curr_idx)
        else:
            self.disp ='curr_idx: {}'.format(self.curr_idx)
        return self.curr_idx
    
    def get_curr_idx(self):
        return self.curr_idx
    
    def reset(self):
        self.to_be_saved = {'imgs':[i for i in range(12)], 'boxed':[i for i in range(12)], 'det_results': [i for i in range(12)]}
        self.curr_idx =0
        self.disp ='curr_idx: {}'.format(self.curr_idx)
    def save(self, directory):
        right = True
        for i in self.to_be_saved['imgs']:
            t = type(i)
            if t is int:
                right = False
                self.disp = '{}th is not an image, curr_idx: {}'.format(i, self.curr_idx)
                return
        if right:
            Path(directory).mkdir(parents=True, exist_ok=True)
            for key, value in self.to_be_saved.items():
                if key in ['imgs', 'boxed']:
                    for i, v in enumerate(value):
                        Image.fromarray(v).save(directory+'/{}{}.jpg'.format(key,i))
                        # print(v)
                elif key in ['det_results']:
                    with open(directory+'/{}.pkl'.format(key), 'wb') as f:
                        pickle.dump(self.to_be_saved['det_results'], f)

            self.reset()
            self.disp = 'Images Saved! curr_idx: {}'.format(self.curr_idx)    
    def set_idx(self,idx):
        self.curr_idx = idx
        self.disp ='curr_idx: {}'.format(self.curr_idx)
# Datacontainer = SaveData()



    
label_names = Label2Word('frcnn')
def app_object_detection():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='frcnn', help='model to attack')
    args = parser.parse_args()
    
   
        

            
            
    
    

 
    # Session-specific caching   
    cache_key = "object_detection_dnn"



    col1, col2, col3 = st.columns(3)
    exp_name = col1.text_input("exp_name", "test")
    original_label = col2.text_input("name of the object", "car")
    args.model_name = col3.radio("model", ["frcnn", "yolo", "crcnn", "yolof"])
    
    model, transforms = init_run(args)
    
    t1, t2, t3, t4= st.columns(4)
    
    
    
    # exp_dir = "photos/{}.{}.{}.{}-{}_{}_{}".format(now.month,now.day,now.hour,now.minute, exp_name, original_label, model_name)
    exp_dir = "photos/{}_{}_{}/".format(exp_name, original_label, args.model_name)
    exp_dir = exp_dir.lower()
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    
    net = model
    ra = 0.5
    result_queue = (
        queue.Queue()
    )  # TODO: A general-purpose shared state object may be more useful.
    
    font = 'Ubuntu-R.ttf'
    
    original_label = original_label.lower()
    exp_name = exp_name.lower()
    
    # img_container = {"img": None, 'box':None, 'detection':None}
    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        h,w,c = image.shape
        size = min(h, w)
        image = cv2.getRectSubPix(image, (size, size), (w // 2, h // 2))
        image_t = transforms(image).cuda()[None,...]
        
        
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
        
        with lock:
            detection_container['imgs'] = image.permute(1,2,0).numpy()
            detection_container['boxed'] = box.detach().clone().cpu().numpy()
            detection_container['det_results'] = detections
            

        disp_image = box.cpu().numpy()
        
        h_c, w_c = int(h/2), int(w/2)
        disp_image[h_c-5:h_c+5, w_c-5:w_c+5,1] = 255
        disp_image[h_c-5:h_c+5, w_c-5:w_c+5,0] = 0
        disp_image[h_c-5:h_c+5, w_c-5:w_c+5,2] = 0
        
        disp_image = disp_image.astype(np.uint8)       
        disp_image = cv2.cvtColor(disp_image,cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(disp_image, format="bgr24")
    
    stop = st.checkbox("Stop")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={"video": {
            "width": 1024, "height": 1024, "framerate": {"max":1}}, 
                                "audio": False,
                                },
        async_processing=True,
    )
    
    Datacontainer = SaveData()
    if 'container' not in st.session_state:
        st.session_state['container'] = 0
    if 'datacontainer' not in st.session_state:
        st.session_state['datacontainer'] = SaveData()
    else:
        Datacontainer = st.session_state['datacontainer']
            
    def _insert():
        idx = Datacontainer.get_curr_idx()
        Datacontainer.insert_img_data(st.session_state['container'],idx)
    def _save():
        Datacontainer.save(exp_dir+d)
    def _set_curr_idx():
        Datacontainer.do_set_idx = True
    
    
    
    
    
    d = st.select_slider('what experiment',['0base', '1high', '2far', '3orange', '4shadow'], on_change=Datacontainer.reset)
    col1, col2, col3, col4 = st.columns(4)
    col1.button('append data', on_click=_insert, key='appenddataaaa')
    img_idx = col2.slider('img index',min_value=0,max_value=11, value=0, on_change=_set_curr_idx)
    if Datacontainer.do_set_idx:
        Datacontainer.set_idx(img_idx)
        Datacontainer.do_set_idx=False
    print(Datacontainer.display())
    col3.write(Datacontainer.disp)
    col4.button('save data', on_click=_save)
    # col5.button('reset', on_click=)
    
    
    # with lock:
    #     print(Datacontainer.curr_idx, Datacontainer.to_be_saved, container)
    
  
    running = False
    while webrtc_ctx.state.playing:
        with lock:
            if detection_container['imgs'] is not None:
                st.session_state['container'] = detection_container
                break
        time.sleep(0.05)
        
    
    
    # with lock:
    #     img = img_container["img"]
    #     box = img_container["box"]
    #     det = img_container["detection"]
        
    # if save_ and img != None:
    #     print('save', save_)
    #     save_data(box,img,det)


            
    
    
def set_session_state():
    if 'hor' not in st.session_state:
        st.session_state.hor = 0
    if 'save' not in st.session_state:
        st.session_state.save = False
    if 'retake' not in st.session_state:
        st.session_state.retake = False

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


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()