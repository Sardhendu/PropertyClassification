


from conv_net.train import TrainConvEnc
max_batches = 14
import logging
logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
tr_obj = TrainConvEnc(dict(inp_img_shape=[250, 300, 3],
                           crop_shape=[128, 128, 3],
                           out_img_shape=[128, 128, 3],
                           use_checkpoint=True,
                           save_checkpoint=True,
                           write_tensorboard_summary=False
                           ),
                      device_type='cpu',
                      which_net='autoencoder',  # vgg
                      image_type='assessor_code')
tr_loss_arr, cv_loss_arr, l_rate_arr = tr_obj.run(num_epochs=100, num_batches=max_batches + 1, get_stats_at=1,
                                                    plot=True, cluster_score=True)  # + 1)



# inp_img_shape=[250, 300, 3], crop_shape=[196, 128, 3],  out_img_shape=[128, 128, 3]).preprocessImageGraph()