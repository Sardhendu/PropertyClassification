

# TO-DO's

1. Make Vizulalization to see the overlayed images, if limiting the to a shape of 100x100x3 makes land images and house images distinguish from each other. **DONE**
2. If the above logic makes sense then build an inception model with input as 96x96x3 and compare overlayed images with assessor images **NO NEED *RESNET-18 DOES FINE AFTER PADDING***
3. Increase the batch size to 128, and see if the CPU is able to handle it or not. **DONE**
4. Run a model and see statistics for both the assessor images and overlayed images **DONE**
5. In the Test Data set sort the checkpoints based on epoch and batch_num (check proper sorting) and check if the batch number is rendering properly while collection the statistics. **DONE**


## New LIst
* We saw that the overlayed model is working awesome, but the catch is that sometimes the overlay is wrong, for example a fake building is overlayed on top of a land. This could be the reason because OSM data could be old and the fake building could actually have been demolished.

* Aerial images are more recent and can be trusted, so it would be a good idea to train teh model n the ensemble od overlayed and aerial images.

How to go about doing it.
1. Train overlayed images for 4-5 steps, since they learn very fast, we could even use a simple neural network (not necessarily a RESNET or INCEPTION). 
2. Train aerial images for more steps, since they take time to learn.
3. Get the features from trained models form both the type of images.
4. Train another network, stacking both the features.
5. Check if the model performance increases.

Benefits.
1. Having a multi-step network would remove the dependency of having both type of images i.e Aerial, Overlayed or in this case even assessor. If all different image types are available for an address then we can use the ensemble network or we can just run 1 network and give the best possible prediction based on the data.  