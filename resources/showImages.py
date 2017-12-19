
# coding: utf-8

# # Image dataset visualization
# This notebook can be used to browse throught all images, inspect for each image what class it is annotated as, and if wanted correct the annotation. 
# 
# Changing image annotation:
# - Simply use the drop-down menue under each image to change its class.
# - As is explained in part 3 of the documentation, all images are in subfolders of *DATA_DIR\images\fashionTexture\*, where the subfolder names equal the class name. To change the class of an image it needs to be moved to the new subfolder - this is exaclty what the UI in this notebook does. 
# - Example: Given an image labeled as 'striped' with filename 105.jpg, and we change the drop-box to 'dotted'. The image is then moved from the file path *DATA_DIR\images\fashionTexture\striped\105.jpg* to *DATA_DIR\images\fashionTexture\dotted\105.jpg*. This assumes that the destination does not exist or otherwise the UI will throw an error.

# In[2]:


import sys, os
sys.path.append(".")
sys.path.append("..")
sys.path.append("libraries")
sys.path.append("../libraries")
from IPython.display import display
from helpers import getAmlLogger
from utilities_general_v2 import *
from ui_annotation import AnnotationUI
from PARAMETERS import procDir, imgOrigDir
get_ipython().magic('autosave 1')

boShowTrainingSet = True #Set to False to show test set instead

amlLogger = getAmlLogger()
if amlLogger != []:
    amlLogger.log("amlrealworld.ImageClassificationUsingCntk.showImages", "true")

# Load data
lutLabel2Id = readPickle(pathJoin(procDir, "lutLabel2Id.pickle"))
if not boShowTrainingSet:
   imgDict = readPickle(pathJoin(procDir, "imgDictTest.pickle"))
else:
   imgDict = readPickle(pathJoin(procDir, "imgDictTrain.pickle"))


# In[5]:


# Instantiate and show annotation UI
annotationUI = AnnotationUI(imgOrigDir, imgDict, lutLabel2Id, gridSize=(2, 2)) #, wZoomImgWidth = 350)
display(annotationUI.ui)

