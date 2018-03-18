#  Property Classification


For a Real-Estate application, it is very important to recognize if a property is vacant or not. For a simple case, 
this project is aimed to classify whether a property is land or house.  

Here we use four different types of images.

* Assessor Image (property images released once in 5 years)
* Aerial Images extracted from Bing and Google
* Streetside Images extracted from Bing and Google 
* OSM building corners (parcel boundary).



## Overview:

#### Problem Scenario: 

* **Assessor** images: Assessor images we have are 5-9 years old and in 5-9 years lot's of properties (house) have been demolished and lots of houses are built on vacant land. Therefore, if the label says "a property is a house" then the image might indicate that its a "land". There are many such cases, which makes the model performance poor when only trained on 
assessor images.  

#### External Data Collection and Preparation:

Aerial, streetside images from google maps and bing maps are updated every 1-2 years and are more recent. A model would be more reliable in these images.

  * **Aerial** images from Bing and Google maps.
  * **Streetside** images from Bing and Google maps.
  * **Building boundary coordinates** for chicago from Open Street Map
    
  * Overlay building boundaries (collected from OSM) on satellite static images collected from google maps. For details look here [Overlay building boundary on static images](https://github.com/Sardhendu/PropertyClassification/tree/master/semantic_segmentation)    

Here's a snapshot of each image

<img src="https://github.com/Sardhendu/PropertyClassification/blob/master/images/assessor.jpg" width="250" height="200"> <img src="https://github.com/Sardhendu/PropertyClassification/blob/master/images/streetside.jpg" width="250" height="200"> <img src="https://github.com/Sardhendu/PropertyClassification/blob/master/images/overlayed.jpg" width="250" height="200">


#### Model (Neural Net) 

  * [RESNET-18](https://github.com/Sardhendu/PropertyClassification/blob/master/conv_net/resnet.py)
  * [VGG like](https://github.com/Sardhendu/PropertyClassification/blob/master/conv_net/vgg.py)
