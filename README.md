# face_rec-dlib

To  enhance efficiency and accuracy,we used dlib, a modern toolkit that includes machine learning algorithms and tools for creating complex software. Although dlib is written in C++, we implemented it using Python, which provided a robust face recognition module based on deep metric learning. This module is known for its efficiency and high accuracy.

The implementation was carried out in the PyCharm IDE, which offered an integrated development environment conducive to efficient coding and debugging.

The code doesn’t explicitly import or use dlib library, but the face_recognition library used in this code is built on top of dlib. Specifically, face_recognition uses dlib for its face detection and face encoding functionalities. So indirectly dlib library is being employed through the face_recognition library.

Here’s a more detailed breakdown:


Face Detection: The function face_recognition.face_locations internally uses either of two different models from Dlib’s Histogram of Oriented Gradients (HOG) or Convolutional Neural Network (CNN) based face detector.


Face Encodings: The function calculates faces with embeddings using faces_encodings method from dlib’s pre-trained neural network model.


The code does face recognition in real time using a video feed from the webcam. 

It captures frames from a webcam, processes each frame for face detection and comparison against known face encodings, and finally presents the results in real time. Face recognition is accomplished through libraries such as face_recognition; OpenCV is used for the presentation of results visually through bounding boxes and labels.

To facilitate real-time face recognition, it is essential to have a set of precomputed face encodings from known individuals. This section describes the process of generating and storing these face encodings.
The Images from a specified directory (images) are loaded, and the corresponding names (derived from the filenames) are stored in lists.

 How does face Encoding helps : 


Provides Known Face Encodings:  we require a set of known face encodings to compare against the faces detected in the webcam feed. This snippet generates those encodings from a set of pre-labeled images.


Saves Encodings for Future Use: By saving the encodings to a file (encodingcas.p), you can load them directly in the real-time face recognition code without needing to regenerate them each time. This makes the process more efficient.




