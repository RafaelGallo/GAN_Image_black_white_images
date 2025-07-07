# Generative Adversarial Network (GAN) - Colorize Old Black-and-White Photos

![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/github/license/RafaelGallo/GAN_Image_black_white_images)
![Colab](https://img.shields.io/badge/Colab-Ready-yellow?logo=googlecolab)
![Pix2Pix](https://img.shields.io/badge/Architecture-Pix2Pix_GAN-green)
![U-Net](https://img.shields.io/badge/Generator-U--Net-yellow)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python\&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?logo=tensorflow\&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-2.x-D00000?logo=keras\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy\&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557C?logo=matplotlib\&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-10.0+-C957BC?logo=python\&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-2E8B57?logo=python\&logoColor=white)
![os](https://img.shields.io/badge/os-Built--in-lightgrey)
![glob](https://img.shields.io/badge/glob-Built--in-lightgrey)
![itertools](https://img.shields.io/badge/itertools-Built--in-lightgrey)
![pathlib](https://img.shields.io/badge/pathlib-Built--in-lightgrey)
![tqdm](https://img.shields.io/badge/tqdm-ProgressBar-4CAF50)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter\&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-Cloud_Notebook-F9AB00?logo=googlecolab\&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle\&logoColor=white)

![Banner](https://github.com/RafaelGallo/GAN_Image_black_white_images/blob/main/img/71560838_network%20connections%20banner%20design%20with%20connecting%20lines%20and%20dots%200609.jpg?raw=true)

## Table of Contents

* [About the Project](#about-the-project)
* [Project Structure](#project-structure)
* [Model Architecture](#model-architecture)
* [Training Pipeline](#training-pipeline)
* [New Image Colorization](#new-image-colorization)
* [Results and Visualizations](#results-and-visualizations)
* [Technologies Used](#technologies-used)
* [References](#references)
* [Author](#author)

## About the Project

This project leverages a Generative Adversarial Network (GAN) to learn the mapping from grayscale (single-channel) images to their corresponding color (three-channel RGB) versions. The core architecture integrates a U-Net-based generator with a PatchGAN discriminator, enabling pixel-wise translation with high fidelity. The implementation is built using TensorFlow 2.x and Keras, and it processes landscape image datasets where each grayscale input is paired with its original colored version. The goal is to restore realistic colors to black-and-white photos by training the model to minimize both adversarial loss and pixel-level L1 distance between the generated and ground truth images.

## Project Structure

```
GAN_Image_black_white_images/
├── img/                            # Illustrative images and banner
├── model/                          # Saved models (.h5)
├── dataset/                        # Grayscale and RGB images (train/test)
├── scripts/
│   ├── train_pix2pix.py            # Training script
│   ├── generator_unet.py           # U-Net generator architecture
│   └── inference_colorize.py       # Inference for new images
├── README.md
```

## Model Architecture

### Generator (U-Net)

The U-Net takes a 256x256x1 grayscale image and returns a 256x256x3 color image. The architecture includes:

* 8 Downsample blocks with LeakyReLU
* 7 Upsample blocks with ReLU and skip connections
* Final output with `tanh` activation scaled to \[-1, 1]

```python
def Generator():
    ...
    inputs = Input(shape=[256, 256, 1])
    ...
    x = last(x)
    return Model(inputs=inputs, outputs=x)
```

### Discriminator (PatchGAN)

Takes the grayscale image concatenated with either the real or generated color image and returns a local probability map (30x30x1) of authenticity.

```python
def Discriminator():
    ...
    x = Concatenate()([inp, tar])
    ...
    return Model(inputs=[inp, tar], outputs=last)
```

## Training Pipeline

1. Image preprocessing:

```python
gray_img = load_img(path, target_size=(256, 256), color_mode='grayscale')
gray_img = img_to_array(gray_img).astype("float32") / 127.5 - 1
```

2. Data Generator using `tf.data.Dataset.from_generator`

3. Loss functions:

* `BinaryCrossentropy` for GAN
* `L1 loss` to enforce similarity with the target

```python
total_gen_loss = gan_loss + (100 * l1_loss)
```

4. Optimizers:

```python
Adam(learning_rate=2e-4, beta_1=0.5)
```

5. Training loop with `@tf.function`:

```python
@tf.function
def train_step(input_image, target):
    ...
```

## New Image Colorization

Once trained, the `.h5` model can be used to colorize unseen grayscale images:

```python
generator = tf.keras.models.load_model("Pix2Pix_Colorizer.h5", compile=False)
```

Helper functions:

```python
def processar_imagem_pb(path):
    img = Image.open(path).convert("L").resize((256, 256))
    return (np.array(img).astype(np.float32) / 127.5 - 1.0)

def gerar_colorido(img_array):
    input_tensor = np.expand_dims(img_array, axis=(0, -1))
    pred = generator.predict(input_tensor)[0]
    return (pred + 1) / 2.0  # Rescale to [0, 1]
```

## Results and Visualizations

| Before (Grayscale)                                                                                 | After (Colorized)                                                                                               |
| -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| ![](https://github.com/RafaelGallo/GAN_Image_black_white_images/blob/main/input/1000.jpg?raw=true) | ![](https://github.com/RafaelGallo/GAN_Image_black_white_images/blob/main/output/__results___11_1.png?raw=true) |

### Example 1 – Snowy Mountains

![Result 1](https://github.com/RafaelGallo/GAN_Image_black_white_images/blob/main/output/__results___11_2.png?raw=true)
*The Pix2Pix GAN successfully colorized a black-and-white image of a snowy mountain landscape.*

### Example 2 – Waves on Rocks

![Result 2](https://github.com/RafaelGallo/GAN_Image_black_white_images/blob/main/output/__results___9_11.png?raw=true)
*Comparison between input grayscale image, ground truth color, and the predicted output.*

### Example 3 – Mountain Scene with Forest

![Result 3](https://github.com/RafaelGallo/GAN_Image_black_white_images/blob/main/output/__results___9_15.png?raw=true)
*The model reconstructs vibrant greens and blues in forest and sky areas.*

### Example 4 – Distant Mountain Ridge

![Result 4](https://github.com/RafaelGallo/GAN_Image_black_white_images/blob/main/output/__results___9_13.png?raw=true)
*Color consistency is preserved in natural transitions between landscape elements.*

### Example 5 – Tree Line and Farmland

![Result 5](https://github.com/RafaelGallo/GAN_Image_black_white_images/blob/main/output/__results___9_31.png?raw=true)
*GAN-based prediction accurately infers green tones for vegetation and blue sky.*

## Technologies Used

* Python 3.10+
* TensorFlow 2.x
* Keras
* NumPy, Pillow, Matplotlib, Seaborn
* tqdm, pathlib, glob, os
* Jupyter Notebook, Google Colab, Kaggle

## References

* [Pix2Pix GAN - Paper](https://arxiv.org/abs/1611.07004)
* [Utkarsh Saxena - Kaggle Pix2Pix Colorizer](https://www.kaggle.com/code/utkarshsaxenadn/landscape-colorizer-pix2pixgan)
* [TensorFlow Pix2Pix Tutorial](https://www.tensorflow.org/tutorials/generative/pix2pix)
* [U-Net Paper](https://arxiv.org/abs/1505.04597)

## Author

**Rafael Gallo**
Deep Learning developer focused on image reconstruction and generation.
[![LinkedIn](https://img.shields.io/badge/-Rafael%20Gallo-blue?logo=Linkedin\&style=flat)](https://www.linkedin.com/in/rafaelgallo)
