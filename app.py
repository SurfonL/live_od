import asyncio
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional

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
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2)

from torchvision.utils import draw_bounding_boxes
from pathlib import Path

from yolov3 import load_model
from PIL import Image
import datetime
import pickle
import os
import glob

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








@st.cache
def init_run():

    yolo = load_model('models/model_confs/yolov3.conf',
                                'models/state_dicts/yolov3.weights').eval().cuda()

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT


    frcnn = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.75, rpn_post_nms_top_n_test=512, download_file=True).eval().cuda()
    
    return  frcnn, weights,yolo





global save_, retake_, hor
save_ = threading.Event()
retake_ = threading.Event()
# saved_ = threading.Event()




def app_object_detection():
    
    
    
    
    
    set_session_state()
    class LoadImage():
        def __init__(self) -> None:
            self.cur_idx = 0
            self.past_idx = 0
            self.images = []
            
        def load_img(self, txt, d):
            images = []
            _dir = os.path.join('photos',txt,d)
            _dir = _dir.lower()
            
            for i in range(12):
                f = os.path.join(_dir,str(i)+"_.jpg")
                if os.path.exists(f):
                    im = np.array(Image.open(f))
                else:
                    im = np.random.uniform(low = 0, high=255,size=((512,512,3))).astype(np.uint8)
                images.append(im)
            self.images = images
            
        def return_img(self,disp_image):
            global hor
            if hor >= len(self.images):
                pass
            else:
                h,w,c = disp_image.shape
                img = Image.fromarray(self.images[hor])
                img = img.resize((w,h))
                img = np.array(img)
                return img
        
    def _save_data():
        # st.session_state.hor += 1
        save_.set()
        
    def _retake():
        retake_.set()    
        
    def save_data(box, img, detections):
        global save_, retake_
        
        s = save_.is_set()
        r = retake_.is_set()
        
        if s or r:
            box = Image.fromarray(box)
            img = Image.fromarray(img)
            Path(exp_dir+d).mkdir(parents=True, exist_ok=True)
            

            n = exp_dir + d+str(hor)
            box.save(n+'_.jpg')
            img.save(n+'.jpg')
            with open(n +'.pkl','wb') as fw:
                pickle.dump(detections, fw)
            if os.path.exists(n+'_.jpg'):
                with open('hor.pkl', 'wb') as p:
                    pickle.dump(hor,p)
                    print(hor)
            
            
            save_.clear()
            retake_.clear()
            
            
    
    
    frcnn, weights,yolo = init_run()

 
    # Session-specific caching   
    cache_key = "object_detection_dnn"

    
    
    transforms = T.Compose([
                            T.ToTensor(),
                            # T.Resize((512,512)),
                            ])    

    col1, col2, col3 = st.columns(3)
    exp_name = col1.text_input("exp_name", "test")
    original_label = col2.text_input("name of the object", "car")
    model_name = col3.radio("model", ["frcnn", "yolo"])
    
    t1, t2, t3, t4= st.columns(4)
    t4.button("save", on_click = _save_data)
    
    
    
    # exp_dir = "photos/{}.{}.{}.{}-{}_{}_{}".format(now.month,now.day,now.hour,now.minute, exp_name, original_label, model_name)
    exp_dir = "photos/{}_{}_{}/".format(exp_name, original_label, model_name)
    exp_dir = exp_dir.lower()
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    
    net = yolo if model_name=='yolo' else frcnn
    compare = LoadImage()
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
        image = transforms(image).cuda()[None,...]

        with torch.no_grad():
            detections = net(image)

        
        image = image*255
        image = image[0].detach().cpu().to(torch.uint8)
        if model_name == 'yolo':
            detections[0]['labels'] +=1
        labels = [str(weights.meta["categories"][i.item()]) for i in detections[0]['labels']]

        
        det = {'labels':[], 'boxes':[], 'scores':[]}
        if not original_label == "any":
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
        box = draw_bounding_boxes(image, b ,l, colors = color,font=font, font_size=font_size, width=4).permute(1,2,0).detach().cpu().numpy()
        
        save_data(box,image.permute(1,2,0).numpy(),det)

        
        # NOTE: This `recv` method is called in another thread,
        # so it must be thread-safe.
        result_queue.put(detections[0])  # TODO:
        disp_image = box
        
        try:
            if txt != "none":
                l_img = compare.return_img(disp_image)
                if isinstance(l_img, np.ndarray):
                    disp_image = disp_image*ra + l_img*(1-ra)
        except:
            pass
        
        h_c, w_c = int(h/2), int(w/2)
        
        disp_image[h_c-5:h_c+5, w_c-5:w_c+5,1] = 255
        disp_image[h_c-5:h_c+5, w_c-5:w_c+5,0] = 0
        disp_image[h_c-5:h_c+5, w_c-5:w_c+5,2] = 0
        
        disp_image = disp_image.astype(np.uint8)       
        disp_image = cv2.cvtColor(disp_image,cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(disp_image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={"video": {
            "width": 1024, "height": 1024, "framerate": {"max":2}}, 
                                  "audio": False,
                                  },
        async_processing=True,
    )
    
    
    pr = pickle.load(open('hor.pkl', 'rb'))
    t3.write('last saved: '+ str(pr))
    
    pr = 0 if pr == 11 else pr+1
    global hor
    hor = t2.number_input("horizontal start from", min_value = 0, max_value = 11, value = pr)
    t1.write("horizontal idx: {}".format(hor))
    
    
    

    
  
    

    
    
    col1, col2, col3 = st.columns(3)
    vertical = col1.radio("vertical", ["low", "high"], index = 0)
    distance = col2.radio("distance", ["close", "far"], index = 0)
    light = col3.radio("light",["white", "orange"], index = 0)
    col1, col2 = st.columns(2)
    txt = col1.text_input("load exp name", "none")
    ra = col2.slider('compare ratio', min_value=0.0, max_value=1.0, value=0.5)
    
    d = "{}-{}-{}/".format(vertical,distance,light)
    Path(exp_dir+d).mkdir(parents=True, exist_ok=True)

   
    try: 
        if txt != 'none':
            txt = txt.lower()
            compare = LoadImage()
            compare.load_img(txt,d)
    except NameError:
        txt = 'none'

    
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