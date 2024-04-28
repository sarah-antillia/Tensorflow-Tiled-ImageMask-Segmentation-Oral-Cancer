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

# TensorflowUNetTileInferencer.py
# 2023/06/08 to-arai
# 2024/04/22: Moved infer_tiles method in TensorflowModel to this class 

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="true"

import sys
import shutil
import numpy as np

import cv2
from PIL import Image
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
from TensorflowUNetInferencer import TensorflowUNetInferencer

from TensorflowModelLoader import TensorflowModelLoader

class TensorflowUNetTiledInferencer(TensorflowUNetInferencer):

  def __init__(self, config_file):
    super().__init__(config_file)

    print("=== TensorflowUNetTiledInferencer.__init__ config?file {}".format(config_file))
    #self.config = ConfigParser(config_file)
    self.images_dir = self.config.get(ConfigParser.TILEDINFER, "images_dir")
    self.output_dir = self.config.get(ConfigParser.TILEDINFER, "output_dir")
    self.tiledinfer_binarize =self.config.get(ConfigParser.TILEDINFER,   "binarize", dvalue=True) 
    print("--- tiledinfer binarize {}".format(self.tiledinfer_binarize))
    self.tiledinfer_threshold = self.config.get(ConfigParser.TILEDINFER, "threshold", dvalue=60)

    # 2024/04/25
    self.sharpening = self.config.get(ConfigParser.TILEDINFER, "sharpening", dvalue=False)

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
    
  def infer(self):
    if os.path.exists(self.output_dir):
      shutil.rmtree(self.output_dir)
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    self.infer_tiles(self.images_dir, self.output_dir, expand=True)

  # 1 Split the original image to some tiled-images
  # 2 Infer segmentation regions on those images 
  # 3 Merge detected regions into one image
  # Added MARGIN to cropping 
  def infer_tiles(self, input_dir, output_dir, expand=True):    
    image_files  = glob.glob(input_dir + "/*.png")
    image_files += glob.glob(input_dir + "/*.jpg")
    image_files += glob.glob(input_dir + "/*.tif")
    image_files += glob.glob(input_dir + "/*.bmp")
    MARGIN       = self.config.get(ConfigParser.TILEDINFER, "overlapping", dvalue=0)
    print("MARGIN {}".format(MARGIN))
    
    merged_dir   = None
    try:
      merged_dir = self.config.get(ConfigParser.TILEDINFER, "merged_dir")
      if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
      if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)
    except:
      pass

    width  = self.config.get(ConfigParser.MODEL, "image_width")
    height = self.config.get(ConfigParser.MODEL, "image_height")

    split_size  = self.config.get(ConfigParser.TILEDINFER, "split_size", dvalue=width)
    print("---split_size {}".format(split_size))
    
    tiledinfer_debug = self.config.get(ConfigParser.TILEDINFER, "debug", dvalue=False)
    tiledinfer_debug_dir = "./tiledinfer_debug_dir"
    if tiledinfer_debug:
      if os.path.exists(tiledinfer_debug_dir):
        shutil.rmtree(tiledinfer_debug_dir)
      if not os.path.exists(tiledinfer_debug_dir):
        os.makedirs(tiledinfer_debug_dir)
 
    # Please note that the default setting is "True".
    bitwise_blending  = self.config.get(ConfigParser.TILEDINFER, "bitwise_blending", dvalue=True)
    print("--- bitwise_blending {}".format(bitwise_blending))
    bgcolor = self.config.get(ConfigParser.TILEDINFER, "background", dvalue=0)  

    for image_file in image_files:
      image   = Image.open(image_file)
      #PIL image color_order = "rgb"
      w, h    = image.size

      # Resize the image to the input size (width, height) of our UNet model.      
      resized = image.resize((width, height))

      # Make a prediction to the whole image not tiled image of the image_file 
      cv_image= self.pil2cv(resized)
      predictions = self.predict([cv_image], expand=expand)
          
      prediction  = predictions[0]
      whole_mask  = prediction[0]    

      #whole_mask_pil = self.mask_to_image(whole_mask)
      #whole_mask  = self.pil2cv(whole_mask_pil)
      whole_mask  = self.normalize_mask(whole_mask)
      # 2024/03/30
      whole_mask  = self.binarize(whole_mask)

      whole_mask  = cv2.resize(whole_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
      basename = os.path.basename(image_file)
      self.tiledinfer_log = None
      
      if tiledinfer_debug and os.path.exists(tiledinfer_debug_dir):
        tiled_images_output_dir = os.path.join(tiledinfer_debug_dir, basename + "/images")
        tiled_masks_output_dir  = os.path.join(tiledinfer_debug_dir, basename + "/masks")
        if os.path.exists(tiled_images_output_dir):
          shutil.rmtree(tiled_images_output_dir)
        if not os.path.exists(tiled_images_output_dir):
          os.makedirs(tiled_images_output_dir)
        if os.path.exists(tiled_masks_output_dir):
          shutil.rmtree(tiled_masks_output_dir)
        if not os.path.exists(tiled_masks_output_dir):
          os.makedirs(tiled_masks_output_dir)
         
      w, h  = image.size

      vert_split_num  = h // split_size
      if h % split_size != 0:
        vert_split_num += 1

      horiz_split_num = w // split_size
      if w % split_size != 0:
        horiz_split_num += 1
      background = Image.new("L", (w, h), bgcolor)

      # Tiled image segmentation
      for j in range(vert_split_num):
        for i in range(horiz_split_num):
          left  = split_size * i
          upper = split_size * j
          right = left  + split_size
          lower = upper + split_size

          if left >=w or upper >=h:
            continue 
      
          left_margin  = MARGIN
          upper_margin = MARGIN
          if left-MARGIN <0:
            left_margin = 0
          if upper-MARGIN <0:
            upper_margin = 0

          right_margin = MARGIN
          lower_margin = MARGIN 
          if right + right_margin > w:
            right_margin = 0
          if lower + lower_margin > h:
            lower_margin = 0

          cropbox = (left  - left_margin,  upper - upper_margin, 
                     right + right_margin, lower + lower_margin )
          
          # Crop a region specified by the cropbox from the whole image to create a tiled image segmentation.      
          cropped = image.crop(cropbox)

          # Get the size of the cropped image.
          cw, ch  = cropped.size

          # Resize the cropped image to the model image size (width, height) for a prediction.
          cropped = cropped.resize((width, height))
          if tiledinfer_debug:
            #line = "image file {}x{} : x:{} y:{} width: {} height:{}\n".format(j, i, left, upper, cw, ch)
            #print(line)            
            cropped_image_filename = str(j) + "x" + str(i) + ".jpg"
            cropped.save(os.path.join(tiled_images_output_dir, cropped_image_filename))

          cvimage  = self.pil2cv(cropped)
          predictions = self.predict([cvimage], expand=expand)
          
          prediction  = predictions[0]
          mask        = prediction[0]    
          mask        = self.mask_to_image(mask)
          # Resize the mask to the same size of the corresponding the cropped_size (cw, ch)
          mask        = mask.resize((cw, ch))

          right_position = left_margin + width
          if right_position > cw:
             right_position = cw

          bottom_position = upper_margin + height
          if bottom_position > ch:
             bottom_position = ch

          # Excluding margins of left, upper, right and bottom from the mask. 
          mask         = mask.crop((left_margin, upper_margin, 
                                  right_position, bottom_position)) 
          iw, ih = mask.size
          if tiledinfer_debug:
            #line = "mask  file {}x{} : x:{} y:{} width: {} height:{}\n".format(j, i,  left, upper, iw, ih)
            #print(line)
            cropped_mask_filename = str(j) + "x" + str(i) + ".jpg"
            mask.save(os.path.join(tiled_masks_output_dir , cropped_mask_filename))
          # Paste the tiled mask to the background. 
          background.paste(mask, (left, upper))

      basename = os.path.basename(image_file)
      output_file = os.path.join(output_dir, basename)
      cv_background = self.pil2cv(background)

      bitwised = None
      if bitwise_blending:
        # Blend the non-tiled whole_mask and the tiled-backcround
        bitwised = cv2.bitwise_and(whole_mask, cv_background)
        # 2024/03/30
        bitwised = self.binarize(bitwised)
        bitwized_output_file =  os.path.join(output_dir, basename)
        cv2.imwrite(bitwized_output_file, bitwised)
      else:
        # Save the tiled-background. 
        if self.sharpening:
          sharpened = self.sharpen(cv_background)
          cv2.imwrite(output_file, sharpened)
        else:
          background.save(output_file)

      print("=== Saved outputfile {}".format(output_file))
      if merged_dir !=None:
        img   = np.array(image)
        img   = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #2024/03/10
        if bitwise_blending:
          mask = bitwised
        else:
          mask  = cv_background 
 
        mask  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img += mask
        merged_file = os.path.join(merged_dir, basename)
        cv2.imwrite(merged_file, img)     

  def mask_to_image(self, data, factor=255.0, format="RGB"):
    h = data.shape[0]
    w = data.shape[1]
    data = data*factor
    data = data.reshape([w, h])
    data = data.astype(np.uint8)
    image = Image.fromarray(data)
    image = image.convert(format)
    return image
  
  def normalize_mask(self, data, factor=255.0):
    h = data.shape[0]
    w = data.shape[1]
    data = data*factor
    data = data.reshape([w, h])
    data = data.astype(np.uint8)
    return data

  #2024/03/30

  def binarize(self, mask):
    if self.num_classes == 1:
      #algorithm = cv2.THRESH_OTSU
      #_, mask = cv2.threshold(mask, 0, 255, algorithm)
      if self.tiledinfer_binarize:
        #algorithm = "cv2.THRESH_OTSU"
        #print("--- tiled_infer: binarize {}".format(algorithm))
        #algorithm = eval(algorithm)
        #_, mask = cv2.threshold(mask, 0, 255, algorithm)
        mask[mask< self.tiledinfer_threshold] =   0
        mask[mask>=self.tiledinfer_threshold] = 255
    else:
      pass
    return mask     
  """
  def evaluate(self, x_test, y_test): 
    self.load_model()
    batch_size = self.config.get(ConfigParser.EVAL, "batch_size", dvalue=4)
    print("=== evaluate batch_size {}".format(batch_size))
    scores = self.model.evaluate(x_test, y_test, 
                                batch_size = batch_size,
                                verbose = 1)
    test_loss     = str(round(scores[0], 4))
    test_accuracy = str(round(scores[1], 4))
    print("Test loss    :{}".format(test_loss))     
    print("Test accuracy:{}".format(test_accuracy))
    # Added the following lines to write the evaluation result.
    loss    = self.config.get(ConfigParser.MODEL, "loss")
    metrics = self.config.get(ConfigParser.MODEL, "metrics")
    metric = metrics[0]
    evaluation_result_csv = "./evaluation.csv"    
    with open(evaluation_result_csv, "w") as f:
       metrics = self.model.metrics_names
       for i, metric in enumerate(metrics):
         score = str(round(scores[i], 4))
         line  = metric + "," + score
         print("--- Evaluation  metric:{}  score:{}".format(metric, score))
         f.writelines(line + "\n")     
    print("--- Saved {}".format(evaluation_result_csv))
  """

if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
  
    inferencer = TensorflowUNetTiledInferencer(config_file)
    inferencer.infer()

  except:
    traceback.print_exc()
    

