#  Property Classification


For a Real-Estate organization, it is very important to recognize if a property is vacant or not. For a simple case, 
this project is aimed to classify whether a property is land or house.  

Here we use four different types of images.

* Assessor Image (property images (manually collected) released once in 5 years)
* Aerial/Satellite Images extracted from Bing and Google
* Streetside Images extracted from Bing and Google 
* OSM building corners (parcel boundary).

## Overview:

### Problem Scenario: 

* **Assessor** images: Assessor images we have are 5-9 years old and in 5-9 years lot's of properties (house) have been demolished and lots of houses are built on vacant land. Therefore, if the label says "a property is a house" then the image might indicate that its a "land". There are many such cases, which makes the model performance poor when only trained on 
assessor images.  

### External Data Collection and Preparation:

Aerial, streetside images from google maps and bing maps are updated every 1-2 years and are more recent. A model would be more reliable in these images. However, due to the 1-2 years of lag, we may still end

* **Aerial** images from Bing and Google maps. Best source of image readily available and most recent when compared to others.
* **Streetside** images from Bing and Google maps. These images are not always clear.
* **Building boundary coordinates** for chicago from Open Street Map: The OSM data is an open source contribution and hence the data may not be updted so frequently. Overlay building boundaries (collected from OSM) on satellite static images collected from google maps. For details look here [Overlay building boundary on static images](https://github.com/Sardhendu/PropertyClassification/tree/master/src/semantic_segmentation)

### Problem Continued:
Even with the external data we see lots of our labels not consistent with the image picture. So we typically have a data issue. We plan to use best all the images (Mixture of experts model). Since, there is a good chance that at least one image type would be consistent with the label. Also, we use bootstrapping techniques to correct the labels and augment them per iteration.

Here's a snapshot of the images:

<div id="image-table">
    <table>
	    <tr>
    	    <td style="padding:5px">
        	    <img src="https://github.com/Sardhendu/PropertyClassification/tree/master/src/images/assessor.png" width="200" height="200"><figcaption><center>Assessor Image</center></figcaption>
      	    </td>
            <td style="padding:5px">
            	<img src="https://github.com/Sardhendu/PropertyClassification/tree/master/src/images/streetside.jpg" width="200" height="200"><figcaption><center>Streetside Image</center></figcaption>
             </td>
            <td style="padding:5px">
            	<img src="https://github.com/Sardhendu/PropertyClassification/tree/master/src/images/aerial.png" width="200" height="200"><figcaption><center>Aerial Image</center></figcaption>
             </td>
             <td style="padding:5px">
            	<img src="https://github.com/Sardhendu/PropertyClassification/tree/master/src/images/overlayed.jpg" width="200" height="200"><figcaption></center>Overlayed Image</center></figcaption>
             </td>
        </tr>
    </table>
</div>


### Models (Deep Nets) 
Let us now discuss all different models employed for different types of images.

#### [RESNET-18](https://github.com/Sardhendu/PropertyClassification/tree/master/src/conv_net/resnet.py) + a little variation

<div id="wrapper">
    <div class="twoColumn">
         <p>
            RESNET-18 model is trained with Satellite Images from google maps. RESNET's can go very deep and are very robust to the problem of vanishing gradient. In doing so, they are able to learn very complex features within the image. The center pixel in the google extracted image is the Latitude and Longitude of the address location. We use the OSM data to crop the surronding of the building from a image. We then resize (128x128x3) and zeropad them to make a shape of 224x224x3. While 224x224x3 is not a necessity (since we dont use pretrined weights), we do it to respect the RESNET architecture. <br><br>
         </p>
    </div>
</div>

<div id="image-table">
    <table>
	    <tr>
    	    <td style="padding:5px">
        	    <img src="https://github.com/Sardhendu/PropertyClassification/tree/master/src/images/home_cropped.jpg" width="300" height="150"><figcaption><center>House bbox cropped</center></figcaption>
      	    </td>
            <td style="padding:5px">
            	<img src="https://github.com/Sardhendu/PropertyClassification/tree/master/src/images/home_resized.png" width="200" height="200"><figcaption><center>Home Resized/Pad</center></figcaption>
             </td>
            <td style="padding:5px">
            	<img src="https://github.com/Sardhendu/PropertyClassification/tree/master/src/images/land.png" width="200" height="200"><figcaption><center>Land central crop</center></figcaption>
             </td>
        </tr>
    </table>
    <table>
        <tr>
            <td style="padding:5px">
            	<img src="https://github.com/Sardhendu/PropertyClassification/tree/master/src/images/prec_recall_curve.png" width="600" height="200"><figcaption><center>Precision Recall Plot</center></figcaption>
            </td>
        </tr>
    </table>

</div>



--------------

#### [CONV-NET](https://github.com/Sardhendu/PropertyClassification/tree/master/src/conv_net/convnet.py)

<div id="wrapper">
    <div class="twoColumn">
        <img align="right" width="200" height="200" src="https://github.com/Sardhendu/PropertyClassification/tree/master/images/overlayed2.png">
    </div>
    <div class="twoColumn">
         <p>
            Convnet model is trained with Overlayed Images i.e. The idea here is to expicitely provide the model with 
            the knowledge of house and land using coloring scheme. The roof top of the houses are colored red. This allows the model to 
            learn very quickly in few steps. Our experiment shows that the model was able to learn a good distinction in just 2-3 steps. We use a simple Conv-net architecture becasue now due to the colors the model no longer needs
             a deep architecture to learn simple features. We havent tried, but judging by the overlayed pictures we 
             think even a simple model could do a descent job classifying the image.<br><br><b>Challange</b>: The 
             building boundaries required to create overlayed images are collected from <b>Open Street map</b>. These may not 
             be updated as frequently as Google maps. Moreover, getting building boundaries for all the location may 
             not be feasible. One way to generate colored image given an satellite view is to use <b>Fully 
             Convolutional Networks for semantic segmenting</b>[TODO]. <br>     
         </p>
    </div>
</div>
    
---------------

#### [CONV AUTOENCODER](https://github.com/Sardhendu/PropertyClassification/tree/master/src/conv_net/conv_autoencoder.py)

<div id="wrapper">
    <div class="twoColumn">
        <img align="right" width="200" height="200" src="https://github.com/Sardhendu/PropertyClassification/tree/master/src/images/assessor2.png">
    </div>
    <div class="twoColumn">
         <p>
            Autoencoders are used for Assessor Images. Assessor images are 5-9 years old and there is a high chance that a Land property then would be a house now. This means that despite the label might say house, the image might indicate a land. So we could either trust the labels or the image. Autoencoder are unsupervised techniques that do not require a label to make a distinction between two labels. We feed in the autoencoder with images of house and land and leave it for the autoencoder to find an encoding that could distinguish between land and house. We create a encoding space of 64 dimensions and try k-means clustering with 2 initial centers.<br><br><b>Challange:</b> Assessor images might be expensive to obtain, since these images are manually collected by organization/individuals. In a real scenario, finding assessor image for every address is overstated.<br>    
         </p>
    </div>
</div>

--------


## Data Pipeline (Using Apache Airflow):

Below is a view of Data pipeline that is achieved using Apache Airflow Framework.

 
<img src="https://github.com/Sardhendu/PropertyClassification/tree/master/src/images/pipeline.jpg" width="400" height="200"><figcaption></center>Airflow Pipeline</center></figcaption>