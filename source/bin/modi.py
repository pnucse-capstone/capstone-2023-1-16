import os
import sys
from Enum import InpaintingType

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../lama')))
sys.path.insert(1, os.path.abspath(os.path.join(__dir__, '../MAT')))
# =====================================================================================
# yolo import
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from copy import deepcopy
import cv2
# =====================================================================================
# lama import
import logging
import traceback

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.training.data.datasets import get_transforms
from saicinpainting.evaluation.data import load_image

from torch.utils.data import Dataset
import torchvision.transforms as T

from legacy import load_network_pkl
from dnnlib.util import open_url
# =====================================================================================
LOGGER = logging.getLogger(__name__)
# lama 파일 경로
CONFIG_PATH = '../lama/configs/prediction/default.yaml'
CKPT_PATH = '../lama/big-lama/'
DEEFILLIV2_PATH = '../deepfillv2/pretrained/states_pt_places2.pth'
MAT_PATH = '../MAT/pretrained/Places_512.pkl'
# =====================================================================================
# 마우스 좌표 클래스
class MouseGesture:
    def __init__(self) -> None:
        self.is_plotted = False
        self.label = -1
        self.cx, self.cy = -1, -1

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.cx, self.cy = x, y
            print("왼쪽 버튼 눌림 \t좌표 : x : {} y : {}".format(x,y) )
            self.is_plotted = False
# =====================================================================================
# main 코드
# instruction
    # mouse_info : 마우스 좌표 클래스

    # cur_mask : 출력할 마스크 / mask_tensor : 출력할 마스크 (타입 - Tensor)
    # is_tracking : 마스크 결과가 있는지 여부
    # results : yolo 결과 (타입 - Results : https://docs.ultralytics.com/modes/predict/#working-with-results)

mouse_info = MouseGesture()
names = []
inpainting_type = InpaintingType.NONE
torch.set_printoptions(profile="full")
def main():
    # ==================================================================================
    # Variable Init
    global names    # class 이름
    global fps_str # fps 출력
    cur_mask = []
    # mask_tensor = None
    is_tracking = False

    #dataCapture()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==================================================================================
    # Model Init
    # yolo Init
    model = YOLO("yolov8n-seg.pt")
    names = model.names
    results = model(source="0", stream=True, verbose=False)
    # results = model(source="http://172.21.215.175:4747/video", stream=True, verbose=False)     # verbose : console 출력 여부
    
    if (inpainting_type == InpaintingType.LAMA):
        predict_config = OmegaConf.load(CONFIG_PATH)
        predict_config.model.path = CKPT_PATH

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(predict_config.model.path, 
                                        'models', 
                                        predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False)
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

    elif (inpainting_type == InpaintingType.DEEPFILLV2):
        generator_state_dict = torch.load(DEEFILLIV2_PATH)['G']

        if 'stage1.conv1.conv.weight' in generator_state_dict.keys():
            from deepfillv2.model.networks import Generator
        else:
            from deepfillv2.model.networks_tf import Generator

        # set up network
        generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)

        generator_state_dict = torch.load(DEEFILLIV2_PATH)['G']
        generator.load_state_dict(generator_state_dict, strict=True)

    elif (inpainting_type == InpaintingType.MAT):
        from MAT.networks.mat import Generator 
        resolution = 512

        with open_url(MAT_PATH) as f:
            G_saved = load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False) # type: ignore
        net_res = 512 if resolution > 512 else resolution
        generator = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(device).eval().requires_grad_(False)
        copy_params_and_buffers(G_saved, generator, require_all=True)
    
    # predict Init
    # ==================================================================================
    # stream loop : yolo 결과를 받아서 인페인팅 결과 출력
    # instruction
        # boxes : yolo 결과 박스 정보 (타입 - Boxes : https://docs.ultralytics.com/modes/predict/#boxes)
        # masks : yolo 결과 마스크 정보 (타입 - Masks : https://docs.ultralytics.com/modes/predict/#masks)
    for r in results:
        is_tracking = False
        boxes = r.boxes
        masks = r.masks

        # ==============================================================================
        # 마우스 좌표 정보가 없으면 원본 이미지 출력
        if (mouse_info.cx == -1 or mouse_info.cy == -1 or masks == None):
            temp_img = plot(r.orig_img, boxes, masks)
            show(temp_img)
        else:
        # ==============================================================================
        # 클릭한 상태일 때 마스크 출력
            # 마우스 좌표가 바뀐 경우
            if mouse_info.is_plotted == False:
                for i, e in enumerate(masks.data):
                    # mask_tensor = e
                    mask = e.cpu().numpy()
                    label = int(boxes[i].cls)

                    # 마스크 데이터에 마우스 좌표가 포함되어 있는지 확인
                    if mask[mouse_info.cy][mouse_info.cx] == 1:
                        cur_mask = mask
                        mouse_info.label = label
                        mouse_info.is_plotted = True
                        is_tracking = True
                        break
                
                # 마스크 데이터가 없으면 원본 이미지 출력
                if (is_tracking == False):
                    mouse_info.label = -1
                
            # 좌표가 바뀌지 않은 경우
            else:
                # (클릭한 좌표 밖으로 대상이 벗어난 경우를 처리하기 위한 코드)
                # 라벨이 있는 경우 기존 라벨을 이용해서 마스크 트래킹
                if mouse_info.label != -1:           
                    for i, box in enumerate(boxes):
                        if label == int(box.cls):
                            # mask_tensor = masks.data[i]
                            cur_mask = masks.data[i].cpu().numpy()
                            is_tracking = True
                            break
            
            # 출력할 마스크가 없는 경우 원본 이미지 출력
            if (is_tracking == False):
                # 클릭한 곳에 마스킹할 대상이 없는 경우를 위해 변수 세팅
                mouse_info.is_plotted = True
                cur_mask = []

                temp_img = plot(r.orig_img, boxes, masks)
                show(temp_img)
                
            # 출력할 마스크가 있는 경우 inpainting 실행
            else:
                # 마스크 영역 넓힘 -> iterations 횟수 올리면 테두리가 넓어짐
                # temp_mask = cur_mask
                temp_mask = cv2.dilate(cur_mask, None, iterations=3)
                
                if (inpainting_type == InpaintingType.LAMA):
                    result_img = lama_predict(model, predict_config, device, r.orig_img, temp_mask)
                    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                elif (inpainting_type == InpaintingType.DEEPFILLV2):
                    result_img = deepfillv2(generator, device, r.orig_img, temp_mask)
                elif (inpainting_type == InpaintingType.MAT):
                    result_img = MAT(generator, device, r.orig_img, temp_mask)

                show(result_img)
# =====================================================================================
# inpainting - lama
def lama_predict(model, predict_config, device, origin, mask):
    global index
    try:
        dataset = make_dataset(origin, mask)

        for img_i in range(len(dataset)):
            mask_fname = f'{img_i}.png'
            cur_out_fname = f'./output/0.png'

            # os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            batch = default_collate([dataset[img_i]])

            # start = time()
            if predict_config.get('refine', False):
                assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                # image unpadding is taken care of in the refiner, so that output image
                # is same size as the input image
                cur_res = refine_predict(batch, model, **predict_config.refiner)
                cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
            else:
                with torch.no_grad():
                    batch = move_to_device(batch, device)
                    batch['mask'] = (batch['mask'] > 0) * 1
                    batch = model(batch)                    
                    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                    unpad_to_size = batch.get('unpad_to_size', None)
                    if unpad_to_size is not None:
                        orig_height, orig_width = unpad_to_size
                        cur_res = cur_res[:orig_height, :orig_width]

            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(cur_out_fname, cur_res)

            return cur_res

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)   

# =====================================================================================
# =====================================================================================
# lama predict를 위한 코드
class InpaintingDataset(Dataset):
    def __init__(self, datadir, name, origin, mask, img_suffix='.jpg'):
        self.datadir = datadir
        self.mask_filenames = {name}
        self.img_filenames = {f'{name}_mask.png'}
        self.origin = origin
        self.mask = mask

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        image = load_image(self.origin, mode='RGB')
        mask = load_image(self.mask, mode='L')
        result = dict(image=image, mask=mask[None, ...])

        return result

def make_dataset(origin, mask, out_size=512, transform_variant='default', **kwargs):
    if transform_variant is not None:
        transform = get_transforms(transform_variant, out_size)
    
    dataset = InpaintingDataset('', 1, origin, mask)
    
    return dataset

def load_image(img, mode='RGB', return_orig=False):
    if img.all() == None : return None

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img
# =====================================================================================
# =====================================================================================
# 화면 출력 코드
def show(img):
    global mouse_info

    cv2.imshow("1", img)
    cv2.setMouseCallback("1", mouse_info.on_mouse, param=img)
    cv2.waitKey(1)
    # cv2.waitKey(500 if batch[3].startswith('image') else 1)  # 1 millisecond

# 마스크, 박스 이미지 출력
def plot(origin, boxes, masks):
    annotator = Annotator(deepcopy(origin))

    # 마스크 출력
    if (masks is not None and len(masks) != 0):
        # mask = mask.unsqueeze(0)
        im_gpu = torch.as_tensor(origin, dtype=torch.float16, device=masks.data.device).permute(
            2, 0, 1).flip(0).contiguous() / 255
        idx = boxes.cls if boxes else range(len(masks))

        annotator.masks(masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)
    
    # 박스 출력
    if (boxes is not None and len(boxes) != 0):
        for box in reversed(boxes):
            c, id = int(box.cls), None if box.id is None else int(box.id.item())
            name = ('' if id is None else f'id:{id} ') + names[c]
            annotator.box_label(box.xyxy.squeeze(), name, color=colors(c, True))
    return annotator.result()

def dataCapture():
    cam = cv2.VideoCapture("http://192.168.50.138:4747/video")

    if not cam.isOpened():
        print("failed")
        exit()

    state = False
    cnt = 0

    while True:
        ret, frame = cam.read()

        if not ret:
            print("Can't read Cam")
            break

        if state :
            print("Capturing")
        else :
            print("Stop")

        cv2.imshow("Data Cam", frame)
        if cv2.waitKey(1) == ord('c'): 
            state = not state

        if state == True :
            cv2.imwrite('data/obj{}.jpg'.format(cnt), frame)
            cnt+=1

        if cv2.waitKey(1) == ord('q'):
           break

    cam.release()
    cv2.destroyAllWindows()
# =====================================================================================
def deepfillv2(generator, device, org_img, mask_img):
    # prepare input
    image = T.ToTensor()(org_img)
    mask = T.ToTensor()(mask_img)

    _, h, w = image.shape
    grid = 8

    image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
    mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    # print(f"Shape of image: {image.shape}")

    image = (image*2 - 1.).to(device)  # map image values to [-1, 1] range
    mask = (mask > 0.5).to(dtype=torch.float32,
                           device=device)  # 1.: masked 0.: unmasked

    image_masked = image * (1.-mask)  # mask image

    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
    x = torch.cat([image_masked, ones_x, ones_x*mask],
                  dim=1)  # concatenate channels

    with torch.inference_mode():
        _, x_stage2 = generator(x, mask)

    # complete image
    image_inpainted = image * (1.-mask) + x_stage2 * mask

    # save inpainted image
    img_out = ((image_inpainted[0].permute(1, 2, 0) + 1)*127.5)
    img_out = img_out.to(device='cpu', dtype=torch.uint8)
    
    return img_out.numpy()
# =====================================================================================
def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def MAT(generator, device, org_img, mask_img):
    truncation_psi = 1
    noise_mode = 'const'

    label = torch.zeros([1, generator.c_dim], device=device)    

    mask2 = 1 - mask_img
    img_512 = cv2.resize(org_img, (512,512), interpolation=cv2.INTER_LINEAR)
    mask_512 = cv2.resize(mask2, (512,512), interpolation=cv2.INTER_LINEAR)
    
    img = (torch.from_numpy(img_512).float().to(device) / 127.5 - 1).unsqueeze(0)
    img = img.permute(0, 3, 1, 2)
    mask = torch.from_numpy(mask_512).float().to(device).unsqueeze(0).unsqueeze(0)
    

    with torch.no_grad():
        z = torch.from_numpy(np.random.randn(1, generator.z_dim)).to(device)
        output = generator(img, mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
        output = output[0].cpu().numpy()

    return output
    
# =====================================================================================

# 실행
if __name__ == '__main__':
    if (len(sys.argv) < 2):
        inpainting_type = InpaintingType.LAMA
    else:
        mode = sys.argv[1]
        for type in InpaintingType:
            if (type.name == mode.upper()):
                inpainting_type = type
                break
        
        if (type == InpaintingType.NONE):
            inpainting_type = InpaintingType.LAMA
    print(inpainting_type)
    main()