# Import TensorFlow, a powerful library for deep learning and machine learning
import tensorflow as tf

# Import necessary classes from Keras, a high-level neural networks API
from tensorflow.keras.models import Sequential  # Sequential model to stack layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # Different layers for the CNN
import matplotlib.pyplot as plt  # Library for plotting graphs
import os  # Library for interacting with the operating system

# Paths to the dataset and model directory
dataset_dir = 'mango_disease_dataset'  # Directory where the dataset is stored
model_dir = 'model'  # Directory where the model will be saved

# Create the model directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)  # Create the directory to save the model

# Load datasets using image_dataset_from_directory
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,  # Directory containing the dataset
    validation_split=0.2,  # Split 20% of data for validation
    subset="training",  # Specify this is the training subset
    seed=123,  # Seed for random number generator for reproducibility
    image_size=(150, 150),  # Resize all images to 150x150 pixels
    batch_size=32  # Number of images to be processed in each batch
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,  # Directory containing the dataset
    validation_split=0.2,  # Split 20% of data for validation
    subset="validation",  # Specify this is the validation subset
    seed=123,  # Seed for random number generator for reproducibility
    image_size=(150, 150),  # Resize all images to 150x150 pixels
    batch_size=32  # Number of images to be processed in each batch
)

# Normalization layer to scale the pixel values
normalization_layer = tf.keras.layers.Rescaling(1./255)  # Rescale pixel values from [0, 255] to [0, 1]

# Apply the normalization to the datasets
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))  # Normalize training data
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))  # Normalize validation data

# Building the Convolutional Neural Network (CNN) Model
# Rectified Linear Unit(ReLU)
model = Sequential([  # Using Sequential model to stack layers
    tf.keras.Input(shape=(150, 150, 3)),  # Input layer with shape 150x150x3 (image size with 3 color channels)
    Conv2D(32, (3, 3), activation='relu'),  # First convolutional layer with 32 filters and 3x3 kernel
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer to reduce spatial dimensions
    Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer with 64 filters and 3x3 kernel
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer to reduce spatial dimensions
    Conv2D(128, (3, 3), activation='relu'),  # Third convolutional layer with 128 filters and 3x3 kernel
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer to reduce spatial dimensions
    Flatten(),  # Flatten the 3D output to 1D for the fully connected layers
    Dense(512, activation='relu'),  # Fully connected layer with 512 neurons and ReLU activation
    Dropout(0.5),  # Dropout layer to prevent overfitting by dropping 50% of neurons randomly
    Dense(8, activation='softmax')  # Output layer with 8 neurons (number of classes) and softmax activation
])

# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification
    optimizer='adam',  # Adaptive Moment Estimation(Adam) optimizer for efficient training
    metrics=['accuracy']  # Metric to evaluate during training and testing
)

# Display the model architecture
model.summary()

# Callback to save the best model during training
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(model_dir, 'best_model.keras'),  # File path to save the model
    save_best_only=True,  # Save only the best model (with highest validation accuracy)
    monitor='val_accuracy',  # Monitor validation accuracy
    mode='max',  # Save model with maximum validation accuracy
    verbose=1  # Print messages during the saving process
)

# Training the model
history = model.fit(
    train_dataset,  # Training dataset
    validation_data=validation_dataset,  # Validation dataset
    epochs=40,  # Number of epochs (iterations over the entire dataset)
    callbacks=[checkpoint_callback]  # List of callbacks to use during training
)

# Extract training history for plotting
acc = history.history['accuracy']  # Training accuracy
val_acc = history.history['val_accuracy']  # Validation accuracy
loss = history.history['loss']  # Training loss
val_loss = history.history['val_loss']  # Validation loss

# Range of epochs for plotting
epochs_range = range(40)

# Plotting Training and Validation Accuracy and Loss
plt.figure(figsize=(8, 8))  # Create a figure with a specified size

# Plot training and validation accuracy
plt.subplot(1, 2, 1)  # Create a subplot for accuracy
plt.plot(epochs_range, acc, label='Training Accuracy')  # Plot training accuracy
plt.plot(epochs_range, val_acc, label='Validation Accuracy')  # Plot validation accuracy
plt.legend(loc='lower right')  # Position the legend in the lower right corner
plt.title('Training and Validation Accuracy')  # Title of the accuracy plot

# Plot training and validation loss
plt.subplot(1, 2, 2)  # Create a subplot for loss
plt.plot(epochs_range, loss, label='Training Loss')  # Plot training loss
plt.plot(epochs_range, val_loss, label='Validation Loss')  # Plot validation loss
plt.legend(loc='upper right')  # Position the legend in the upper right corner
plt.title('Training and Validation Loss')  # Title of the loss plot

# Show the plots
plt.show()
