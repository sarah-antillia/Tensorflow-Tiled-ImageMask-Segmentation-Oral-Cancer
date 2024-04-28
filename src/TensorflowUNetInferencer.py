# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# TensorflowUNetInfer.py
# 2023/06/05 to-arai
# 2024/04/22: Moved infer method in TensorflowModel to this class 

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="true"

import numpy as np

import shutil
import sys
import cv2
import glob
import traceback

from ConfigParser import ConfigParser
from ImageMaskDataset import ImageMaskDataset
from NormalizedImageMaskDataset import NormalizedImageMaskDataset

from TensorflowUNet import TensorflowUNet
from TensorflowAttentionUNet import TensorflowAttentionUNet 
from TensorflowEfficientUNet import TensorflowEfficientUNet
from TensorflowMultiResUNet import TensorflowMultiResUNet
from TensorflowSwinUNet import TensorflowSwinUNet
from TensorflowTransUNet import TensorflowTransUNet
from TensorflowUNet3Plus import TensorflowUNet3Plus
from TensorflowU2Net import TensorflowU2Net
from TensorflowSharpUNet import TensorflowSharpUNet
#from TensorflowBASNet    import TensorflowBASNet
from TensorflowDeepLabV3Plus import TensorflowDeepLabV3Plus
from TensorflowEfficientNetB7UNet import TensorflowEfficientNetB7UNet
#from TensorflowXceptionLikeUNet import TensorflowXceptionLikeUNet
from TensorflowModelLoader import TensorflowModelLoader

class TensorflowUNetInferencer:

  def __init__(self, config_file):
    print("=== TensorflowUNetInferencer.__init__ config?file {}".format(config_file))
    self.config = ConfigParser(config_file)
    self.num_classes = self.config.get(ConfigParser.MODEL, "num_classes")

    self.images_dir = self.config.get(ConfigParser.INFER, "images_dir")
    self.output_dir = self.config.get(ConfigParser.INFER, "output_dir")
    # 2024/04/25
    self.sharpening = self.config.get(ConfigParser.INFER, "sharpening", dvalue=False)

    # Create a UNetMolde and compile
    #model          = TensorflowUNet(config_file)
    ModelClass = eval(self.config.get(ConfigParser.MODEL, "model", dvalue="TensorflowUNet"))
    print("=== ModelClass {}".format(ModelClass))

    self.unet  = ModelClass(config_file) 
    print("--- self.unet {}".format(self.unet))
    self.model = self.unet.model
    print("--- self.model {}".format(self.model))
    self.model_loader = TensorflowModelLoader(config_file)

    # 2024/04/22 Load Model
    self.loader = TensorflowModelLoader(config_file)
    self.loader.load(self.model)

    if not os.path.exists(self.images_dir):
      raise Exception("Not found " + self.images_dir)
        
  def create_gray_map(self,):
     self.gray_map = []
     print("---- create_gray_map {}".format(self.grayscaling))
        
     if self.grayscaling !=None and self.colorize_mask: 
       (IR, IG, IB) = self.grayscaling
       for color in self.mask_colors:
         (b, g, r) = color
         gray = int(IR* r + IG * g + IB * b)
         self.gray_map += [gray]

  def infer(self):
    if os.path.exists(self.output_dir):
      shutil.rmtree(self.output_dir)
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    input_dir  = self.images_dir
    output_dir = self.output_dir
    expand     = True

    print("=== infer")
    colorize = self.config.get(ConfigParser.SEGMENTATION, "colorize", dvalue=False)
    black    = self.config.get(ConfigParser.SEGMENTATION, "black",    dvalue="black")
    white    = self.config.get(ConfigParser.SEGMENTATION, "white",    dvalue="white")
    blursize = self.config.get(ConfigParser.SEGMENTATION, "blursize", dvalue=None)
    #writer   = GrayScaleImageWriter(colorize=colorize, black=black, white=white)
    color_order = self.config.get(ConfigParser.DATASET,   "color_order", dvalue="rgb")
   
    # 2024/04/18
    self.mask_colors= self.config.get(ConfigParser.MASK, "mask_colors")
    self.grayscaling = self.config.get(ConfigParser.MASK, "grayscaling", dvalue=None)
    
    image_files  = glob.glob(input_dir + "/*.png")
    image_files += glob.glob(input_dir + "/*.jpg")
    image_files += glob.glob(input_dir + "/*.tif")
    image_files += glob.glob(input_dir + "/*.bmp")

    width        = self.config.get(ConfigParser.MODEL, "image_width")
    height       = self.config.get(ConfigParser.MODEL, "image_height")
    print("--- self.grayscaling {}".format(self.grayscaling))
    self.create_gray_map()
    self.num_classes  = self.config.get(ConfigParser.MODEL, "num_classes")
    self.mask_channels = self.config.get(ConfigParser.MASK, "mask_channels")
    self.masks_colors_order   = self.config.get(ConfigParser.MASK, "color_order")
    self.mask_colors   = self.config.get(ConfigParser.MASK, "mask_colors")
    self.colorized_output_format = self.config.get(ConfigParser.INFER, 
                                                   "colorized_output_format",
                                                   dvalue="rgb")

    merged_dir   = None
    
    merged_dir = self.config.get(ConfigParser.INFER, "merged_dir", dvalue=None)
    if merged_dir !=None:
      if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
      if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)
    mask_colorize = self.config.get(ConfigParser.INFER, "mask_colorize", dvalue=False)
    if mask_colorize:
      self.create_gray_map()

    colorized_dir = self.config.get(ConfigParser.INFER, "colorized_dir", dvalue=None)
    if mask_colorize and colorized_dir !=None:
      if os.path.exists(colorized_dir):
        shutil.rmtree(colorized_dir)
      if not os.path.exists(colorized_dir):
        os.makedirs(colorized_dir)

    for image_file in image_files:
      print("--- infer image_file {}".format(image_file))
      basename = os.path.basename(image_file)
      name     = basename.split(".")[0]    
      img      = cv2.imread(image_file)
      # convert (B,G,R) -> (R,G,B) color-order
      # 2024/04/20
      if color_order == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
      
      h, w = img.shape[:2]
      # Any way, we have to resize input image to match the input size of our TensorflowUNet model.
      img         = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
      predictions = self.predict([img], expand=expand)
      prediction  = predictions[0]
      image       = prediction[0]    

      output_filepath = os.path.join(output_dir, basename)

      # You will have to resize the predicted image to be the original image size (w, h), and save it as a grayscale image.
      mask = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)

      #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
      mask = mask*255
      mask = mask.astype(np.uint8) 
      gray_mask = mask    # This is used for merging input image with.
      if self.num_classes ==1:
        if self.sharpening:
          mask = self.sharpen(mask)
          cv2.imwrite(output_filepath, mask)
        else:
          print("=== Inference for a single classes {} ".format(self.num_classes))
          cv2.imwrite(output_filepath, mask)
          print("--- Saved {}".format(output_filepath))

        if mask_colorize and os.path.exists(colorized_dir):
          print("--- colorizing the inferred mask ")
          mask = self.colorize_mask(mask, w, h)
          colorized_filepath = os.path.join(colorized_dir, basename)
          #2024/04/20 Experimental
          #mask = cv2.medianBlur(mask, 3)
          # colorrized_output_format = "bgr"

          if self.colorized_output_format == "bgr":
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
          cv2.imwrite(colorized_filepath, mask)
          print("--- Saved {}".format(colorized_filepath))
      else:
        print("=== Inference in multi classes {} ".format(self.num_classes))
        print("----infered mask shape {}".format(image.shape))
        # The mask used in traiing     
      if merged_dir !=None:
        img   = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        if blursize:
          img   = cv2.blur(img, blursize)
        #img = cv2.medianBlur(img, 3)
        img += gray_mask
        merged_file = os.path.join(merged_dir, basename)
        cv2.imwrite(merged_file, img)
        print("--- Saved {}".format(merged_file))

  def colorize_mask_one(self, mask, color=(255, 255, 255), gray=0):
    h, w = mask.shape[:2]
    rgb_mask = np.zeros((w, h, 3), np.uint8)
    #condition = (mask[...] == gray) 
    condition = (mask[...] >= gray-10) & (mask[...] <= gray+10)   
    rgb_mask[condition] = [color]  
    return rgb_mask   
    
  def colorize_mask(self, img, w, h,):
      rgb_background = np.zeros((w, h, 3), np.uint8)
      for i in range(len(self.mask_colors)):
        color  = self.mask_colors[i]
        gray  = self.gray_map[i]
        rgb_mask = self.colorize_mask_one(img, color=color, gray=gray)
        rgb_background += rgb_mask
      rgb_background = cv2.resize(rgb_background, (w, h), interpolation=cv2.INTER_NEAREST)
      return rgb_background
           
  def predict(self, images, expand=True):
    predictions = []
    for image in images:
      #print("=== Input image shape {}".format(image.shape))
      if expand:
        image = np.expand_dims(image, 0)
      pred = self.model.predict(image)
      predictions.append(pred)
    return predictions    

  def pil2cv(self, image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2: 
        pass
    elif new_image.shape[2] == 3: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

  def sharpen(self, image):
    klist  =   [ [-1, -1, -1],[-1, 9, -1], [-1, -1, -1] ]
    kernel = np.array(klist, np.float32)
    sharpened = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened


if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    inferencer = TensorflowUNetInferencer(config_file)

    inferencer.infer()

  except:
    traceback.print_exc()
    

