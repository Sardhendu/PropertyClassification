


## VGG:
    
### Feb 2:  NET
  * preprocessing: Yes
  * num_training_images = 162, batch size 32, 32 , last batch 62
  * Image rshaped = 128*128
  * 4 conv layers, 3 fully connected layers and 1 softmax layer
  * Batch normalization for each conv layers
  * Dropout for each FC layers keep_prob 0.5 each but last 0.7
  * learning_rate decay activated
  * Training accuracy fluctuating range(50-65)
  * Loss fluctuating heavily avg (100-200)
   
 * Comments: Remove dropout sees, if not done, the network would always drop the same neurons for each batch. This should be random.


### Feb 3: Changes: NET
   * The image size = 250x250
   * Random crop = 224x224
   * Dropout Removed form the Fully connected layers
 
 * Comments: A little bit more stability was introduced with the loss and accuracy. (better than before)
 
    
   * We tried by removing Batch Normalization from conv layers and removing dropout from the Fully Connected layers. The accuracy was a little worse (range 40-55) avg, but the loss came down to 20 even to 5 for some itreration. This is mainly due to the removal of dropout, since neurons were not randomly dropped they didnt have to learn everything again for a new batch hence the loss decreases. We confirmed this by activating batch normalization and removing dropouts.
   
   * At this point we are sure that we would have batch normalization. However, when we removed dropout the training accuracy at the last few steps (20-30) showed a lot of improvement the accuracy was between 70-85. Since we dont have dropout we believe that we may be highly overfitting the data. We shall see that when we add validation accuracy. 
   
   
   
### Feb 11: Changes: Data
   Many major changes happened till Feb 11. They are given below
   
   * A different set of Data was prepared. "data_loader.ipynb" contains the new data sample logic. This data was collected to build and test the model.
   
      **Condition for Image selection**
   
      * Images where last_reviewed_date >= July 2017
      * Images where address_line1 != nan or accessor_photo != nan
      * Indicator : Likely House or Likely Land
      * Randomly retain only 1100 images for each Land and House
      * Sometimes the image is not available from the source, such images were removed
      * Out of 1100 sample each Land and House, 654 land pictures were retained and 891 house images were retained
      
   * The land and house images that were retained. Those pins were used to get the bing satellite image data.
   
   
### Feb 14, NO DROPOUTS
   We made a lot of changes. We added both the aerial and assessor images, seperated 200 data points for validation and had 25 batches for training each with 64 images.
   Other changes were made: Now we dont use steps rather epochs over 64 batches.
   
   * Assessor Images:
   Findings, The model didn't perform as good as the previous model discussed in Feb 3. The reason for this is that now the data is selected at random and we have seen that some images despite being labeled as land are actually house and vice-a-versa. The average accuracy for the 1st epoch was very very bad (appx50%). So it took approximately 2 epoch (64 batch each) for the loss to come down to 10.0. the training accuracy fluctuated between 70 - 80 for the last few batches in the 2nd epoch.
   
    With central crop (A bit better than random crop)
    AVG Batch Accuracy for EPOCH 1 is:  57.875
    AVG Batch Accuracy for EPOCH 2 is:  65.75
    AVG Batch Accuracy for EPOCH 3 is:  70.125
    AVG Batch Accuracy for EPOCH 4 is:  70.625
    AVG Batch Accuracy for EPOCH 5 is:  75.4375
    AVG Batch Accuracy for EPOCH 6 is:  75.0625
    
    Cross Validation:
    Epoch: 2, Cross Validation Validation Accuracy= 73.00000
    Epoch: 3, Cross Validation Validation Accuracy= 71.00000
    Epoch: 4, Cross Validation Validation Accuracy= 77.00000
    Epoch: 5, Cross Validation Validation Accuracy= 74.00000
    Epoch: 6, Cross Validation Validation Accuracy= 66.50000
   
   * Aerial Images:
   Findings, The model didnt perform nearly good. May be because the satellite images didn't comply with the many-a-times with the labels given. It just that the model insn't able to fit good decision boundry. 
   
    AVG Batch Accuracy for EPOCH 1 is: : 52.4375
    AVG Batch Accuracy for EPOCH 2 is: : 58.75
    AVG Batch Accuracy for EPOCH 3 is: : 64.0625
    AVG Batch Accuracy for EPOCH 4 is: : 65.5625
    AVG Batch Accuracy for EPOCH 5 is: : 77.125
    AVG Batch Accuracy for EPOCH 6 is: : 78.125 (Few batch accuracy went down to 60% and 50%, which reduced the average accuracy)
    
    Epoch: 0, Cross Validation Validation Accuracy= 50.50000
    Epoch: 1, Cross Validation Validation Accuracy= 59.00000
    Epoch: 2, Cross Validation Validation Accuracy= 60.50000
    Epoch: 5, Cross Validation Validation Accuracy= 57.00000
   
### Feb 15 Adding dropout to the last three FC layers (0.5, 0.5, 0.8)

    Epoch: 1, AVG Training Accuracy= 54.37500
    Epoch: 2, AVG Training Accuracy= 58.18750
    Epoch: 3, AVG Training Accuracy= 60.43750
    Epoch: 4, AVG Training Accuracy= 60.50000
    Epoch: 5, AVG Training Accuracy= 62.37500
    Epoch: 6, AVG Training Accuracy= 63.87500
    
    Epoch: 2, Cross Validation Accuracy= 65.50000
    Epoch: 3, Cross Validation Accuracy= 64.50000
    Epoch: 4, Cross Validation Accuracy= 74.00000
    Epoch: 5, Cross Validation Accuracy= 71.00000
    Epoch: 6, Cross Validation Accuracy= 67.50000
    
    Aerial Images:
    
    Epoch: 1, AVG Training Accuracy= 53.50000
    Epoch: 2, AVG Training Accuracy= 55.50000
    Epoch: 3, AVG Training Accuracy= 58.00000
    Epoch: 4, AVG Training Accuracy= 59.31250
    Epoch: 5, AVG Training Accuracy= 57.87500
    Epoch: 6, AVG Training Accuracy= 59.12500
    
    Epoch: 2, Cross Validation Accuracy= 56.50000
    Epoch: 3, Cross Validation Accuracy= 57.50000
    Epoch: 4, Cross Validation Accuracy= 60.00000
    Epoch: 5, Cross Validation Accuracy= 62.00000
    Epoch: 6, Cross Validation Accuracy= 62.00000
   
   
   
