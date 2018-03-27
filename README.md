#  Property Classification


For a Real-Estate application, it is very important to recognize if a property is vacant or not. For a simple case, 
this project is aimed to classify whether a property is land or house.  

Here we use four different types of images.

* Assessor Image (property images released once in 5 years)
* Aerial/Satellite Images extracted from Bing and Google
* Streetside Images extracted from Bing and Google 
* OSM building corners (parcel boundary).



## Overview:

### Problem Scenario: 

* **Assessor** images: Assessor images we have are 5-9 years old and in 5-9 years lot's of properties (house) have been demolished and lots of houses are built on vacant land. Therefore, if the label says "a property is a house" then the image might indicate that its a "land". There are many such cases, which makes the model performance poor when only trained on 
assessor images.  

### External Data Collection and Preparation:

Aerial, streetside images from google maps and bing maps are updated every 1-2 years and are more recent. A model would be more reliable in these images.

  * **Aerial** images from Bing and Google maps.
  * **Streetside** images from Bing and Google maps.
  * **Building boundary coordinates** for chicago from Open Street Map
    
  * Overlay building boundaries (collected from OSM) on satellite static images collected from google maps. For details look here [Overlay building boundary on static images](https://github.com/Sardhendu/PropertyClassification/tree/master/semantic_segmentation)    

Here's a snapshot of the images:

<div id="image-table">
    <table>
	    <tr>
    	    <td style="padding:5px">
        	    <img src="https://github.com/Sardhendu/PropertyClassification/blob/master/images/assessor.png" width="200" height="200"><figcaption>Assessor Image</figcaption>
      	    </td>
            <td style="padding:5px">
            	<img src="https://github.com/Sardhendu/PropertyClassification/blob/master/images/streetside.jpg" width="200" height="200"><figcaption>Streetside Image</figcaption>
             </td>
            <td style="padding:5px">
            	<img src="https://github.com/Sardhendu/PropertyClassification/blob/master/images/aerial.png" width="200" height="200"><figcaption>Aerial Image</figcaption>
             </td>
             <td style="padding:5px">
            	<img src="https://github.com/Sardhendu/PropertyClassification/blob/master/images/overlayed.jpg" width="200" height="200"><figcaption>Overlayed Image</figcaption>
             </td>
        </tr>
    </table>
</div>


### Models (Deep Nets) 
Let us now discuss all different models emplopyed for different types of images.

#### [RESNET-18](https://github.com/Sardhendu/PropertyClassification/blob/master/conv_net/resnet.py)

<div id="wrapper">
    <div class="twoColumn">
        <img align="right" width="200" height="200" src="https://github.com/Sardhendu/PropertyClassification/blob/master/images/zeropad_aerial.png">
    </div>
    <div class="twoColumn">
         <p>
            RESNET-18 model is trained with Satellite Images from google maps. RESNET's can go very deep and are very robust to the problem of vanishing gradient. In doing so, they are able to learn very complex features within the image. The center pixel in the google extracted image is the Latitude and Longitude of the address location. Since the RESNET-18 model takes input an image of shape 224x224x3, we central crop a 96x96 tile from the image and zero pad it to shape it as 224x224x3. <br><br>
         </p>
    </div>
</div>

--------------

#### [CONV-NET](https://github.com/Sardhendu/PropertyClassification/blob/master/conv_net/convnet.py)

<div id="wrapper">
    <div class="twoColumn">
        <img align="right" width="200" height="200" src="https://github.com/Sardhendu/PropertyClassification/blob/master/images/overlayed2.png">
    </div>
    <div class="twoColumn">
         <p>
            Convnet model is trained with Overlayed Images i.e. The idea here is expicitely provide the model with 
            the knowledge of house and land. The roof top of the houses are colored red. This allows the model to 
            learn very quickly in few steps. Our experiment shows that the model was able to learn a good distinction in just 2-3 steps. We use a simple Conv-net architecture becasue now due to the colors the model no longer needs
             a deep architecture to learn simple features. We havent tried, but judging be the overlayed pictures we 
             think even a simple could do a descent job classifying the image.<br><br><b>Challange</b>: The building 
             boundaries required to create overlayed images are collected from <b>Open Street map</b>. These may not 
             be updated as frequently as Google maps. Moreover, getting building boundaries for all location may not 
             be feasible. One way to generate colored image given an satellite view is to use <b>Fully 
             Convolutional Networks for semantic segmenting</b>[TODO]. <br>     
         </p>
    </div>
</div>
    
---------------

#### [AUTOENCODER]() [TODO]

<div id="wrapper">
    <div class="twoColumn">
        <img align="right" width="200" height="200" src="https://github.com/Sardhendu/PropertyClassification/blob/master/images/assessor2.png">
    </div>
    <div class="twoColumn">
         <p>
            Autoencoders are used for Assessor Images. Assessor images are 5-9 years old and a there are high chance 
            that a Land property then would be a house now. This means that despite the label might say house the image
             might indicate a house. So we could either trust the labels or the image. Autoencoder are unsupervised 
             techniques that do not require a label to make a classification. We feed in the autoencoder with images 
             of house and land and leave it for the autoencoder to find an encoding that could distinguish between 
             land and house. <br><br><b>Challange:</b> Assessor images might be expensive to obtain, since these 
             images are manually collected by organization/individuals. In a real scenario, finding assessor image 
             for every address is overstated.<br>    
         </p>
    </div>
</div>

--------


