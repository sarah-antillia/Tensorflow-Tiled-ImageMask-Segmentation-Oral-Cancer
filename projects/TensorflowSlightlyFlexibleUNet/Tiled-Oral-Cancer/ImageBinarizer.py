# Copyright 2024 antillia.com Toshiyuki Arai
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

# ImageBinarizer.py

import os
import sys
import glob
import shutil
import numpy as np
import cv2
import traceback

class ImageBinarizer:
  def __init__(self, resize, algorithms, threshold):
    self.RESIZE     = resize
    self.algorithms = algorithms
    self.threshold  = threshold

  def binarize(self, input_dir, output_dir):
    files = glob.glob(input_dir + "/*.jpg")
    files += glob.glob(input_dir + "/*.png")
    if  os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    for file in files:
      self.binarize_one(file, output_dir)

  def binarize_one(self, mask_file, output_dir):
    mask = cv2.imread(mask_file)
    
    mask = cv2.resize(mask, (self.RESIZE, self.RESIZE))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    basename = os.path.basename(mask_file)
    for i, algorithm in enumerate(self.algorithms):
      print("algorithm {}".format(algorithm))
      if  algorithm == cv2.THRESH_BINARY + cv2.THRESH_OTSU: 
        _, mask = cv2.threshold(mask, 0, 255, algorithm)
       
      elif  algorithm == cv2.THRESH_TRIANGLE or algorithm == cv2.THRESH_OTSU: 
        _, mask = cv2.threshold(mask, 0, 255, algorithm)
      elif  algorithm == cv2.THRESH_BINARY or   algorithm ==  cv2.THRESH_TRUNC: 
        #_, mask = cv2.threshold(mask, 127, 255, self.algorithm)
        _, mask = cv2.threshold(mask, self.threshold, 255, algorithm)
      elif algorithm == None:
        mask[mask< self.threshold] =   0
        mask[mask>=self.threshold] = 255
      output_file = os.path.join(output_dir, str(i) + "_" + basename)
      cv2.imwrite(output_file, mask)


if __name__ == "__main__":
  try:
    input_dir  = "./mini_test/masks"
    output_dir = "./mini_test_masks/binarized"
    resize     = 512
    algorithms = [#cv2.THRESH_TRIANGLE, 
                  #cv2.THRESH_OTSU,
                  #cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                  cv2.THRESH_BINARY, 
                  #cv2.THRESH_TRUNC, 
                  #None
                  ]
    threshold = 118 
    binarizer = ImageBinarizer(resize,algorithms, threshold)
    binarizer.binarize(input_dir, output_dir)

  except:
    traceback.print_exc()
