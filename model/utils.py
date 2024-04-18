import torch
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision.transforms import ToPILImage

def load_part_of_model(new_model, src_model_path, s):
    src_model = torch.load(src_model_path)
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        if k in m_dict.keys():
            param = src_model.get(k)
            if param.shape == m_dict[k].data.shape:
                m_dict[k].data = param
                print('loading:', k)
            else:
                print('shape is different, not loading:', k)
        else:
            print('not loading:', k)

    new_model.load_state_dict(m_dict, strict=s)
    return new_model

def load_part_of_model2(new_model, src_model_path):
    src_model = torch.load(src_model_path)
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        k2 = k.replace('denoise_fn.', '')
        if k2 in m_dict.keys():
            # print(k)
            param = src_model.get(k)
            if param.shape == m_dict[k2].data.shape:
                m_dict[k2].data = param
                print('loading:', k)
            # else:
            #     print('shape is different, not loading:', k)
        else:
            print('not loading:', k)

    new_model.load_state_dict(m_dict)
    return new_model

def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    # ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    # if img_np.shape[0] == 1:
    #     ar = ar[0]
    # else:
    #     assert img_np.shape[0] == 3, img_np.shape
    #     ar = ar.transpose(1, 2, 0)

    # return Image.fromarray(ar)
    return ToPILImage()(img_np)
    

def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    # return img_var.detach().cpu().numpy()[0]
    return img_var.detach().cpu().numpy().transpose(1, 2, 0) 
