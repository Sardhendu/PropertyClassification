

# TO-DO's

1. Make Vizulalization to see the overlayed images, if limiting the to a shape of 100x100x3 makes land images and house images distinguish from each other. **DONE**
2. If the above logic makes sense then build an inception model with input as 96x96x3 and compare overlayed images with assessor images **NO NEED**
3. Increase the batch size to 128, and see if the CPU is able to handle it or not. **DONE**
4. Run a model and see statistics for both the assessor images and overlayed images
5. In the Test Data set sort the checkpoints based on epoch and batch_num (check proper sorting) and check if the batch number is rendering properly while collection the statistics.