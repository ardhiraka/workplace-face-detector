# Workplace Face Detector

Recognize specific faces and open work-related apps and url afterward.
You can swap in your own dataset of faces and open work-related apps after those face being recognized.

After specific faces detected, this model automatically open Visual Studio Code, and 2 urls pointed to my work-related webpages and Google Slide.
Pretending you're working.

I'm using [dlib](http://dlib.net/) and [face_recognition](https://github.com/ageitgey/face_recognition) utilities.

## Project Directories

- dataset/ : Contains face images organized into subfolders by name, my partner's name, and others.
- images/ : Contains test images that we’ll use to verify the operation of our model.
- face_detection_model/ : Contains a pre-trained Caffe deep learning model provided by OpenCV to detect faces. This model detects and localizes faces in an image.
- output/ : Contains output pickle files. The output files include:
    - embeddings.pickle : A serialized facial embeddings file. Embeddings have been computed for every face in the dataset and are stored in this file.
    - le.pickle : Label encoder. Contains the name labels for the people that model can recognize.
    - recognizer.pickle : Linear Support Vector Machine (SVM) model.
    
## Files

- extract_embeddings.py: Responsible for using a deep learning feature extractor to generate a 128-D vector describing a face.
- openface.nn4.small2.v1.t7: A Torch deep learning model which produces the 128-D facial embeddings.
- train_model.py : Linear SVM model will be trained by this script.
- recognize_video.py : Recognize who is in frames of a video stream and open specific apps and url.
  
## Usage

```shell
python recognize_video.py -d 'your_detection_model' \
-m 'your_embedding_model' \
-r 'your_recognizer' \
-l 'your_le_pickle_file'
```

example

```shell
python recognize_video.py -d face_detection_model \
-m openface.nn4.small2.v1.t7 \
-r output/recognizer.pickle \
-l output/le.pickle
```
