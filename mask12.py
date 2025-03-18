import os
import random
import numpy as np
import tensorflow as tf

# Example seed
seed_value = 42

os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


# Import necessary libraries
import os
import time
import random
import pathlib
import itertools
from glob import glob
from tqdm import tqdm_notebook, tnrange

# Import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import label
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow, concatenate_images

# Import Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

print('Modules loaded')

# **Create needed functions**

# Function to create dataframe
def create_df(data_dir):
    images_paths = []
    masks_paths = glob(f'{data_dir}/*/*_mask*')

    if not masks_paths:
        print("No mask files found. Please check the paths.")
        return pd.DataFrame()  # Return an empty DataFrame if no masks found

    for i in masks_paths:
        images_paths.append(i.replace('_mask', ''))

    df = pd.DataFrame(data={'images_paths': images_paths, 'masks_paths': masks_paths})

    print(f"Found {len(df)} image-mask pairs.")  # Debug print to check the number of pairs
    return df

# Function to split dataframe into train, valid, test
def split_df(df):
    # create train_df
    train_df, dummy_df = train_test_split(df, train_size= 0.8)

    # create valid_df and test_df
    valid_df, test_df = train_test_split(dummy_df, train_size= 0.5)

    return train_df, valid_df, test_df

# Function to create image generators

from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Define the image and mask generator function
def create_gens(df, batch_size=32, aug_dict=None):
    while True:  # Loop forever so the generator never terminates
        # Shuffle data at each epoch
        df = df.sample(frac=1).reset_index(drop=True)
        
        for start in range(0, len(df), batch_size):
            # End index for current batch
            end = min(start + batch_size, len(df))
            
            batch_images = []
            batch_masks = []
            
            for i in range(start, end):
                # Load image and mask
                image = load_img(df['images_paths'].iloc[i], target_size=(128, 128))
                mask = load_img(df['masks_paths'].iloc[i], target_size=(128, 128), color_mode="grayscale")
                
                # Convert to numpy array
                image = img_to_array(image) / 255.0  # Normalize image
                mask = img_to_array(mask) / 255.0  # Normalize mask
                
                batch_images.append(image)
                batch_masks.append(mask)
                
            # Convert lists to numpy arrays
            batch_images = np.array(batch_images)
            batch_masks = np.array(batch_masks)
            
            # If augmentation is required, apply it here
            if aug_dict:
                # You can add augmentations such as rotation, flip, etc. (e.g., using `ImageDataGenerator` or custom code)
                pass
            
            # Yield the batch
            yield batch_images, batch_masks


# U-Net architecture
def unet(input_size=(128, 128, 3)):
    inputs = Input(input_size)

    # DownConvolution / Encoder Leg
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(inputs)
    bn1 = Activation("relu")(conv1)
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(bn1)
    bn1 = BatchNormalization(axis=3)(conv1)
    bn1 = Activation("relu")(bn1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)
    
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(pool1)
    bn2 = Activation("relu")(conv2)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(bn2)
    bn2 = BatchNormalization(axis=3)(conv2)
    bn2 = Activation("relu")(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(filters=256, kernel_size=(3, 3), padding="same")(pool2)
    bn3 = Activation("relu")(conv3)
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), padding="same")(bn3)
    bn3 = BatchNormalization(axis=3)(conv3)
    bn3 = Activation("relu")(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(filters=512, kernel_size=(3, 3), padding="same")(pool3)
    bn4 = Activation("relu")(conv4)
    conv4 = Conv2D(filters=512, kernel_size=(3, 3), padding="same")(bn4)
    bn4 = BatchNormalization(axis=3)(conv4)
    bn4 = Activation("relu")(bn4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(filters=1024, kernel_size=(3, 3), padding="same")(pool4)
    bn5 = Activation("relu")(conv5)
    conv5 = Conv2D(filters=1024, kernel_size=(3, 3), padding="same")(bn5)
    bn5 = BatchNormalization(axis=3)(conv5)
    bn5 = Activation("relu")(bn5)
    drop5 = Dropout(0.5)(bn5)

    # UpConvolution / Decoder Leg
    up6 = concatenate([Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), padding="same")(drop5), conv4], axis=3)
    conv6 = Conv2D(filters=512, kernel_size=(3, 3), padding="same")(up6)
    bn6 = Activation("relu")(conv6)
    conv6 = Conv2D(filters=512, kernel_size=(3, 3), padding="same")(bn6)
    bn6 = BatchNormalization(axis=3)(conv6)
    bn6 = Activation("relu")(bn6)

    up7 = concatenate([Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding="same")(bn6), conv3], axis=3)
    conv7 = Conv2D(filters=256, kernel_size=(3, 3), padding="same")(up7)
    bn7 = Activation("relu")(conv7)
    conv7 = Conv2D(filters=256, kernel_size=(3, 3), padding="same")(bn7)
    bn7 = BatchNormalization(axis=3)(conv7)
    bn7 = Activation("relu")(bn7)

    up8 = concatenate([Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), padding="same")(bn7), conv2], axis=3)
    conv8 = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(up8)
    bn8 = Activation("relu")(conv8)
    conv8 = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(bn8)
    bn8 = BatchNormalization(axis=3)(conv8)
    bn8 = Activation("relu")(bn8)

    up9 = concatenate([Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding="same")(bn8), conv1], axis=3)
    conv9 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(up9)
    bn9 = Activation("relu")(conv9)
    conv9 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(bn9)
    bn9 = BatchNormalization(axis=3)(conv9)
    bn9 = Activation("relu")(bn9)

    conv10 = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(bn9)

    return Model(inputs=[inputs], outputs=[conv10])

# Specify data directory
data_dir = '/home/hkaja2@cfreg.local/localization/efficientnet/lgg-mri-segmentation/kaggle_3m'

# Create DataFrame and split it
df = create_df(data_dir)
if df.empty:
    raise ValueError("The DataFrame is empty. Check the image and mask paths.")
train_df, valid_df, test_df = split_df(df)

# Augmentation configuration
tr_aug_dict = dict(rotation_range=0.2,
                   width_shift_range=0.05,
                   height_shift_range=0.05,
                   shear_range=0.05,
                   zoom_range=0.05,
                   horizontal_flip=True,
                   fill_mode='nearest')

# Create generators
#train_gen = create_gens(train_df, aug_dict=tr_aug_dict)
#valid_gen = create_gens(valid_df, aug_dict={})
#test_gen = create_gens(test_df, aug_dict={})

# Initialize and compile U-Net model
model = unet(input_size=(128, 128, 3))
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Add early stopping and model checkpoint for monitoring the training process
early_stop = EarlyStopping(patience=5, verbose=1, monitor='val_loss', restore_best_weights=True)
checkpoint = ModelCheckpoint('unet_model.keras', monitor='val_loss', verbose=1, save_best_only=True)


# Train the model using the generators
train_gen = create_gens(train_df, batch_size=40, aug_dict=tr_aug_dict)
valid_gen = create_gens(valid_df, batch_size=40)
test_gen = create_gens(test_df, aug_dict={})

history = model.fit(train_gen,
                    steps_per_epoch=len(train_df) // 40,  # Batch size is 40
                    epochs=10,
                    validation_data=valid_gen,
                    validation_steps=len(valid_df) // 40,
                    callbacks=[early_stop, checkpoint])


# Optionally, save the trained model
model.save('final_unet_model.h5')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from skimage.metrics import structural_similarity as ssim

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(y_true, y_pred, threshold=0.5):
    y_true_bin = (y_true > threshold).astype(np.uint8).flatten()
    y_pred_bin = (y_pred > threshold).astype(np.uint8).flatten()
    iou = jaccard_score(y_true_bin, y_pred_bin, zero_division=0)
    dice = 2 * np.sum(y_true_bin * y_pred_bin) / (np.sum(y_true_bin) + np.sum(y_pred_bin) + 1e-7)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    ssim_score = ssim(y_true.squeeze(), y_pred.squeeze(), data_range=y_pred.max() - y_pred.min())
    return {"IoU": iou, "Dice": dice, "Precision": precision, "Recall": recall, "F1-score": f1, "SSIM": ssim_score}

def draw_bounding_boxes(image, mask):
    _, binary_mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw in red
    return image

def visualize_predictions_with_metrics(model, test_df, save_path="output/predictions_vs_localizationgt.png", num_samples=5):
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    sample_df = test_df.sample(num_samples).reset_index(drop=True)
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, num_samples * 4))  # Four panels for images and metrics

    for i in range(num_samples):
        image = load_img(sample_df['images_paths'].iloc[i], target_size=(128, 128))
        mask = load_img(sample_df['masks_paths'].iloc[i], target_size=(128, 128), color_mode="grayscale")

        image_array = img_to_array(image) / 255.0
        mask_array = img_to_array(mask) / 255.0
        predicted_mask = model.predict(np.array([image_array]))[0]
        predicted_mask_binarized = (predicted_mask.squeeze() > 0.5).astype(np.uint8)

        # Compute metrics
        metrics = calculate_metrics(mask_array, predicted_mask_binarized)

        # Draw bounding boxes on the original image based on the ground truth mask
        bounded_image = draw_bounding_boxes(image_array.copy(), mask_array * 255)

        # Display the images
        axes[i, 0].imshow(bounded_image)
        axes[i, 1].imshow(mask_array.squeeze(), cmap='gray')
        axes[i, 2].imshow(predicted_mask.squeeze(), cmap='gray')

        # Display metrics as text in the fourth subplot
        metrics_text = "\n".join([f"{key}: {value:.3f}" for key, value in metrics.items()])
        axes[i, 3].text(0.1, 0.5, metrics_text, transform=axes[i, 3].transAxes, verticalalignment='center', fontsize=12)
        axes[i, 3].set_title("Metrics")
        axes[i, 3].axis('off')

        for ax in axes[i, :3]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Predictions with Metrics saved at: {save_path}")

# Call the function after your training is complete
visualize_predictions_with_metrics(model, test_df, save_path="output/predictions_vs_localizationgt.png", num_samples=5)


# Example usage



# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_gen, steps=len(test_df) // 40)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')


