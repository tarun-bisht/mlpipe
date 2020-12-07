from PIL import Image
import numpy as np
import requests

def load_image(image_path, dim=None, preserve_ratio=False, 
	preprocess_fn=None, image_type="RGB", dtype="float32"):
    img = Image.open(image_path)
    if dim:
        if preserve_ratio:
        	img.thumbnail(dim)
        else:
        	img=img.resize(dim)
    img = img.convert(image_type)
    if preprocess_fn:
    	img = preprocess_fn(img)
    img = img.astype(dtype)
    return img

def load_url_image(url, dim=None, preserve_ratio=False, 
	preprocess_fn=None, image_type="RGB", dtype="float32"):
    img_request=requests.get(url)
    img= Image.open(BytesIO(img_request.content))
    if dim:
        if preserve_ratio:
        	img.thumbnail(dim)
        else:
        	img=img.resize(dim)
    img = img.convert(image_type)
    if preprocess_fn:
    	img = preprocess_fn(img)
    img = img.astype(dtype)
    return img