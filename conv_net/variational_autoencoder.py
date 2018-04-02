



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda

from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist



batch_size = 100
original_dim = 784
latent_dim = 32
intermediate_dim = 256
epochs = 50

epsilon_std = 1.0

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    ''' WE SAMPLE FROM A RANDOM UNIT GAUSSIAN :
        In order to estimate z (latent space) from x (input space), we need to sample z from p(z|x). Since we learn
        analytically by gradient updating. This means that the notation should be differentiable, which even means
        that the model is deterministic i.e a fized value of parameter to the network will always output the same
        output. Therefore, the only souce to add stocasticity (randomness) is via the input x.
     
        Denoising Autoencoder: In denoising Autoencoders we learn that adding random noise to the input image actually
        controls overfitting by trying to learn a reconstruction to map output (learned from noisy input) to the
        actual input. Variational Autoencoder: In this case we incorporate a probabilistic sampling node (
        random_normal) which makes the model stocastic. Here, epsilon id the randomness (p(epsilon)). The Encoding layer send to the decoder is just the addition of Mean and variance latent representation. Here, we just multiply the randomness to the variance
        latent representation and add it to the mean latent representation.
        And thus we bring Radomness to out model and the inference and generation become entirely differentiable and
        learnable.
        
        Why do we do unit Gaussian: Theoratically, the encodings can take any values, squashing the
        encodings with activation doesn't seem a right. Having a gaussian with unit variance is sort of normalizing
        and adding random noise to control overfitting. For example if we are learning mnist data, say digit 1 has a
        mean 5.2 and digit 9 has a mean 6.4. Now suppose, digit 1 comes in and the mean is found to be 6,
        then the network may say than its 9. By sampling method we learn the variance representation. So VAE method
        would say digit 1 = N(mean=5.2, sd = 1) and digit 9 = N(mean=6.4, sd=0.1). Using this phenomena the network
        would learn the right classification. IMAGE MIXTURES OF GAUSSIAN.
    '''
    z_mean, z_log_var = args
    print (K.shape(z_log_var), K.shape(z_mean))
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=epsilon_std)
    print (K.shape(epsilon))
    stacked_mean_std = z_mean + epsilon* K.exp(z_log_var/2)
    return stacked_mean_std



z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

############
decoder_h = Dense(intermediate_dim, activation='relu')
print(decoder_h)
decoder_mean = Dense(original_dim, activation = 'sigmoid')
print (decoder_mean)
h_decoded = decoder_h(z)
print (h_decoded)
x_decoded_mean = decoder_mean(h_decoded)
print (K.shape(x_decoded_mean))


# instantiate VAE model
vae = Model(x, x_decoded_mean)

xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)  # Entropy loss of input and output
# oss_ = (rho * tf.log(rho / rho_hat)) + (rho_hat * tf.log((1 - rho) / (1 - rho_hat)))
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)  # KL-divergence loss
print ('62342367462387468236426378462 ', kl_loss.shape)
# The KL-divergence loss tries to bring the latent variables closer to a unit gaussian distribution.
vae_loss = K.mean(xent_loss + kl_loss)

aa = vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop', loss=aa)
vae.summary()


#########  model
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[np.where((y_train==6) | (y_train == 8))[0]]
y_train = y_train[np.where((y_train==6) | (y_train == 8))[0]]

x_test = x_test[np.where((y_test==6) | (y_test == 8))[0]]
y_test = y_test[np.where((y_test==6) | (y_test == 8))[0]]

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print (x_train.shape)
print (x_test.shape)

#
vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

# # build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans = kmeans.fit(x_test_encoded)
labels = kmeans.predict(x_test_encoded)
centroids = kmeans.cluster_centers_

labels[labels==0] = 6
labels[labels==1] = 8


from conv_net.utils import Score
Score.accuracy(y_test, labels)


plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()