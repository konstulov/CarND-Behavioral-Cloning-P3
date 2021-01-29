import argparse
import csv
import cv2
import numpy as np
import os

from models import get_nvidia_model

driving_log_fname = 'driving_log.csv'
driving_data_subdir = 'IMG'

# Parse input arguments
parser = argparse.ArgumentParser(description='Train car to drive.')
parser.add_argument(
    '--basedir',
    type=str,
    default='./data',
    help='Base directory with training data.')
parser.add_argument(
    '--datadir',
    type=str,
    default='',
    help='Folder with training data which contains driving log file and IMG subdirectory.')
parser.add_argument(
    '--augmentations',
    type=int,
    default=None,
    help='Number of images to apply data augmentation to.')
parser.add_argument(
    '--epochs',
    type=int,
    default=6,
    help='Number of epochs to train for.')
parser.add_argument(
    '--outname',
    type=str,
    default='model.h5',
    help='Name for the output model H5 file.')
args = parser.parse_args()

driving_log_path = os.path.join(args.basedir, args.datadir, driving_log_fname)
driving_data_dir = os.path.join(args.basedir, args.datadir, driving_data_subdir)
max_augmentations = args.augmentations

print("driving_log_path = '%s'" % driving_log_path)
print("driving_data_dir = '%s'" % driving_data_dir)
print("max_augmentations = %s" % max_augmentations)

# Load the Keras model and print its summary
model = get_nvidia_model()
print(model.summary())

# Read in the drive log csv
lines = []
with open(driving_log_path) as csvfile:
    reader = csv.reader(csvfile)
    header_line = next(reader)
    for line in reader:
        lines.append(line)

# Load the training data and augment it (by using both left, center, and right cameras and flipping images vertically)
images = []
measurements = []
for j, line in enumerate(lines):
    correction = 0.2 # parameter to offset the steering for left/right camera
    steering_center = float(line[3])
    # create adjusted steering measurements for the side camera images
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    if max_augmentations is None or j < max_augmentations:
        steerings = [steering_center, steering_left, steering_right]
    else:
        steerings = [steering_center]
    for i, measurement in enumerate(steerings):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = os.path.join(driving_data_dir, filename)
        image = cv2.imread(current_path)
        images.append(image)
        measurements.append(measurement)

augmented_images = []
augmented_measurements = []
for j, image in enumerate(images):
    measurement = measurements[j]
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    if max_augmentations is None or j < max_augmentations:        
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement*-1.0)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# Train the model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=args.epochs)

# Save the model
model.save(args.outname)