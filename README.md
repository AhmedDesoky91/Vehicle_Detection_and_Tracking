# **Vehicle Detection and Tracking**
---
![intro](https://i.imgur.com/AVVIes3.png)


**In this project, I have worked on the development of algorithm pipleline that detects vehicle objects in a visual scene and keeps tracking of them as long as the vehicle is apparently visual and detected in the scene.**

**Firstly, I have tested my algorithm on a group of test images.**

**Then, I have applied this pipeline to a test video stream.**

## Pipeline Architecture:
---
1. Extract vehicles and non-vehicles features
   * Prepare the dataset (get all images)
   * Split the data into training dataset and test dataset
   * Choose feature extraction paramters and extract the features
2. Traing a Support Vector Machine Classifier
   * Shuffle the data
   * Scale features
   * Fit the data to get the classification model
   * Testing our model accuracy using the test dataset
3. Find cars in a test image
   * Find cars in an image
     * Apply feature extraction for the test image using HoG sub-sampling using different sacles of the image
     * Scale features
     * Predict these features either to be car or non-car (false positives elimination using the `decision_function`)
     * Return the bounding box for the predicted cars in the image
   * Apply heat map for getting rid of multiple detection of the same object 

## Environment:
---
* Ubuntu 16.04 LTS
* Python 3.6.4
* OpenCV 3.1.0
* Anaconda 4.4.10

In the following, I will be disucssing the project pipeline.

> You will find some itnernal functions during my algorithm pipleline. All these functions are listed in my Python notebook file (vehicle_detection_tracking.ipynb) with some explanation under the heading `support functions`


## Step 1: Extract vehicles and non-vehicles features
---

  * Prepare the dataset (get all images)
  * Choose feature extraction paramters and extract the features
    * Split the data into training dataset and test dataset

### Prepare the dataset (get all images)
I have made a folder containing all vehicle image and another folder containing all non-vehicle images for ease of reading them.
You can find the piece of code that used for getting all images in the python file `get_all_images.py`

Here is the python code for reading all the images in vehicles and non-vehicles lists
```python
# Read in cars and notcars
car_images = glob.glob('vehicles_dataset/*.png')
non_car_images = glob.glob('non-vehicles_dataset/*.png')
cars = []
notcars = []
for image in car_images:
    cars.append(image)
for image in non_car_images:
    notcars.append(image)
print("Cars .. ",len(cars))
print("Not cars",len(notcars)) 
```

From the dataset, I got 8792 of cars and 8968 of non-cars which means our dataset is balanced in terms of distribution of classes as they are nearly close.


### Choose feature extraction paramters and extract the features
Here, I have applied tha feature extraction function to get the feature used in training the classifier.
Alfter lots and lots of experiments, I have used the `YCrCb` as I found the output is the most convenient in false positives elimination
Here also I am using the spatial, histogram, and HoG features in my features vector
After getting our features and labels, I have splitted the dataset into training (80%) and test (20%) datasets

```python
# Feature extraction parameters# Feature 
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
```

### Step2: Traing a Support Vector Machine Classifier
  * Shuffle the data:  
  Here, I shuffle the dataset in order to prevent the classifier to biased to images ordering
  * Scale features:  
  Features scaling before training the classifier. This is to prevent the classifier to be biased towards specific features. Also, in order to ease the fitting. The same scaling is applied to the test dataset
  * Fit the data to get the classification model:  
  I used a linear SVM classifier as our problems is a two class problem
  * Testing our model accuracy using the test dataset:  
  I have tested my model using the test dataset and get a result of 98 %

Here is the code I used for training my classifier:
```python
# Fit a per-column scaler
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)
print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
```

### Step 3:  Find cars in a test image
  * Find cars in an image
    * Apply feature extraction for the test image using HoG sub-sampling using different sacles of the image
    * Scale features
    * Predict these features either to be car or non-car (false positives elimination using the `decision_function`)
    * Return the bounding box for the predicted cars in the image
  * Apply heat map for getting rid of multiple detection of the same object 

#### Find cars in an image
In the following, I applying the `find_cars` function to find cars in an images.
This function is applying the HoG sub-samlping to search on cars in an image
This function is doing the following:
  * Apply feature extraction for the test image using HoG sub-sampling using different sacles of the image
  * Scale features
  * Predict these features either to be car or non-car (false positives elimination using the `decision_function`)
  
> It worth to mention that I have used the classifier's `decision_function` rather that `predict` in order to have higher confident detections from the classifier rather than all the positive detection. We can find below from the test images that we do not have any false positives in the test images

Here is the code for fiding cars in an image:
```python
test_images = glob.glob('test_images/*.jpg')
num_of_rows = 8
num_of_cols = 1
image_size = 40
ystart = 400
ystop = 656
scale = 1.5
fig, axs = plt.subplots(num_of_rows, num_of_cols, figsize=(image_size, image_size))
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
i = 0
for test_image in test_images:
    image = mpimg.imread(test_image)
    draw_image = np.copy(image)
    hot_windows = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)      
#     image = X_train[index].squeeze()
    axs[i].axis('off')
    axs[i].imshow(window_img)
    write_name = 'test_images_output/find_cars_'+str(i)+'_.jpg'
    write_image = cv2.cvtColor(window_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(write_name, write_image)
    i = i + 1
```
Here is a sample output of applying `find_cars` to an image:
![find_car_image](https://i.imgur.com/jQgLOnz.jpg)

#### Apply heat map for getting rid of multiple detection of the same object
As we can see from the previous images that we do not have false positives but we have multiple detection of the same vehicle.
Here I have used the heat maps to git rid of the multiple detections.
```python
from scipy.ndimage.measurements import label
test_images = glob.glob('test_images/*.jpg')
num_of_rows = 8
num_of_cols = 2
image_size = 30
ystart = 400
ystop = 656
scale = 1.5
fig, axs = plt.subplots(num_of_rows, num_of_cols, figsize=(image_size, image_size))
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
i = 0
for test_image in test_images:
    image = mpimg.imread(test_image)
    draw_image = np.copy(image)
    hot_windows = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)     
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
    # Apply threshold to help remove false positives
    #heat = apply_threshold(heat,0)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    axs[i].axis('off')
    axs[i].imshow(heatmap, cmap="hot")
    write_name = 'test_images_output/heat_map_'+str(i)+'_.jpg'
    cv2.imwrite(write_name, heatmap)
    i = i + 1
    axs[i].axis('off')
    axs[i].imshow(draw_img)
    i = i + 1
    write_name = 'test_images_output/multiple_detection_elim_'+str(i)+'_.jpg'
    write_image = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(write_name, write_image)
```

Here is a sample output:
![heat_map_image](https://i.imgur.com/iXufiIo.jpg)



### Process Frame Pipeline:
Here I have define a `process_frame` function. In this function I have used the `find_cars` function with heat maps but with different scales to make sure of getting the cars in different sizes through the video frame

Here is the python code for the process frame algorithm pipeline:
```python
def process_frame(img):
    rectangles = []
    colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 11
    pix_per_cell = 16
    cell_per_block = 2
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    ystart = 400
    ystop = 600
    scale = 1.5
    rectangles.append(find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))

    scale = 1.75
    rectangles.append(find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    
    scale = 2
    rectangles.append(find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    
    rectangles = [item for sublist in rectangles for item in sublist] 
    heatmap_img = np.zeros_like(img[:,:,0])
    heatmap_img = add_heat(heatmap_img, rectangles)
    heatmap_img = apply_threshold(heatmap_img, 0)
    labels = label(heatmap_img)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img
```


## Conclusion
---
  * The algorithm pipeline was acceptedly detecting and tracking vehicle in test images and project test video
  * The classifier needs improvement specially for better detecting and tracking the white car
  * This improvement may be trying of getting more features with color spaces experiments, but I think the breakthrough will get of using YOLO or SSD models
