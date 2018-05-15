
from keras import metrics
import tensorflow as tf
from conv_net import ops
from keras import backend as K

def sample_gaussian_1var(z_mean, z_log_var, epsilon_sd):
    ''' WE SAMPLE FROM A RANDOM UNIT GAUSSIAN :
        In order to estimate z (latent space) from x (input space), we need to sample z from p(z|x). Since we learn
        analytically by gradient updating. This means that the notation should be differentiable, which even means
        that the model is deterministic i.e a fixed value of parameter to the network will always output the same
        output. Therefore, the only source to add stocasticity (randomness) is via the input x.

        Denoising Autoencoder: In denoising Autoencoders we learn that adding random noise to the input image actually
        controls overfitting by trying to learn a reconstruction to map output (learned from noisy input) to the
        actual input.
        
        Variational Autoencoder: In this case we incorporate a probabilistic sampling node (
        random_normal) which makes the model stocastic. Here, epsilon is the randomness (p(epsilon)). The Encoding
        layer send to the decoder is just the addition of Mean and variance latent representation. Here,
        we just multiply the randomness to the variance latent representation and add it to the mean latent
        representation. And thus we bring Radomness to out model and the inference and generation become entirely differentiable and
        learnable.

        Why do we do unit Gaussian: Theoratically, the encodings can take any values, squashing the
        encodings with activation doesn't seem the right form (at least at this case). Having a gaussian with unit variance is sort of normalizing and adding random noise to control overfitting. For example if we are learning mnist data, say digit 1 has a mean 5.2 and digit 9 has a mean 6.4. Now suppose, digit 1 comes in and the mean is found to be 6, then the network may say than its 9. By sampling method we learn the variance representation. So VAE method would say digit 1 = N(mean=5.2, sd = 1) and digit 9 = N(mean=6.4, sd=0.1). Using this phenomena the network would learn the right classification. IMAGE MIXTURES OF GAUSSIAN.
    '''
    
    # latent_dim = 32
    with tf.name_scope("sample_gaussian"):
        epsilon = tf.random_normal(shape=tf.shape(z_log_var), mean=0, stddev=epsilon_sd, dtype=tf.float32)
        sum_mean_std = z_mean + epsilon * tf.exp( z_log_var /2)
    return sum_mean_std


def vae_loss(x, x_decoded_mean_squash, z_mean, z_log_var, input_dimension):
    '''
    :param x:
    :param x_decoded_mean:
    :return:

    The loss si defined by two quantities:

    1. Reconstruction loss: Cna be binary_cross_entropy (sigmoid_cross_entropy) or mean_squared_error

    2. The regularized loss: kl_loss = The kl_loss can be thought as the relative entropy between the prior p(z) and
    the posterior distribution q(z). The posterior q(z).
    The formula seems a little weird because we strategically choose certain conjugate priors over z that let us
    analytically integrate the KL divergence, yielding a closed form equation. So,
    -DKL(q(z|x) || p(z))) = 1/2 (summation(1 + log(z_var) - z_mean^2 -z_var), which can also be written as
    -DKL(q(z|x) || p(z))) = 1/2 (summation(1 + z_var_log - z_mean^2 - exp(z_var_log))
    '''

    # Compute VAE loss
    with tf.name_scope("binary_cross_entropy_loss"):
        xent_loss = input_dimension * metrics.binary_crossentropy(
                K.flatten(x),
                K.flatten(x_decoded_mean_squash))

    with tf.name_scope("KL_divergence_loss"):
        """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
        kl_loss = - 0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    
    vae_loss = K.mean(xent_loss + kl_loss)
    
    
    # generated_flat = tf.reshape(self.generated_images, [self.batchsize, 224 * 224 * 3])
    # kl_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_log_var) - tf.log(tf.square(z_log_var)) - 1, 1)
    # vae_loss = K.mean(xent_loss + kl_loss)
    
    # The KL-divergence loss tries to bring the latent variables closer to a unit gaussian distribution. and the
    # binary_cross_entropy minimizes the reconstruction error. Former is trying to make things seems a bit random,
    # whereas the later is trying to learn the input properly
    # vae_loss = tf.reduce_mean(b_xent_loss + kl_loss, name="VAE_total_cost")
    
    return vae_loss, xent_loss, kl_loss
