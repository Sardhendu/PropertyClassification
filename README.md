#  Property Classification


For a Real-Estate application, it is very important to recognize if a property is vacant or not. For a simple case, 
this project is aimed to Classify whether a property is land or house.  

Here we use four different types of images.

* Assessor Image (property images released once in 5 years)
* Aerial Images extracted from Bing and Google
* Streetside Images extracted from Bing and Google 
* OSM building corners (parcel boundary).



## Overview:

#### Data Collection and Preparation:

  * **Aerial** images from Bing and Google maps.
  * **Streetside** images from Bing and Google maps.
  * **Building boundary coordinates** for chicago from Open Street Map
    
  * Overlay building boundaries (collected from OSM) on satellite static images collected from google maps. For details look here [CLICK ME !! Overlay building boundary on static images](https://github.com/Sardhendu/PropertyClassification/tree/master/semantic_segmentation)    


Here's a snapshot of each image

<img src="https://github.com/Sardhendu/PropertyClassification/blob/master/images/assessor.jpg" width="250" height="200"> <img src="https://github.com/Sardhendu/PropertyClassification/blob/master/images/streetside.jpg" width="250" height="200"> <img src="https://github.com/Sardhendu/PropertyClassification/blob/master/images/overlayed.jpg" width="250" height="200">


#### Model (Neural Net) 

  * [RESNET-18](https://github.com/Sardhendu/PropertyClassification/blob/master/conv_net/resnet.py)
  * [VGG like](https://github.com/Sardhendu/PropertyClassification/blob/master/conv_net/vgg.py)
