<div style="text-align: center; color: blue;">

# Vehicle Detection and Counting with YOLOv8

</div>

**Vehicle Detection and Counting** is a Python script designed to detect and count vehicles (specifically cars) in a video using the YOLOv8 model from the `ultralytics` library. The script processes the video, counts the cars crossing a designated line, and saves the final count to an Excel file.



## Key Features:
- Detects cars in a video using YOLOv8.
- Counts cars that cross a specific line in the frame.
- Outputs the video with bounding boxes around detected cars.
- Saves the final car count to an Excel file.

## Requirements:
- Python 3.x
- OpenCV
- ultralytics (YOLOv8)
- pandas

  ## Main Concept

The primary goal of this project is to develop a vehicle detection and counting system using the YOLOv8 (You Only Look Once) object detection model. The system processes video footage to detect and track vehicles, specifically cars, and counts them as they cross a designated line in the frame. 

Key components of the concept:
- **Object Detection with YOLOv8**: YOLOv8 is a state-of-the-art object detection model capable of detecting multiple objects, such as vehicles, in real-time. It offers a balance between accuracy and performance, making it suitable for applications like traffic monitoring.
- **Vehicle Counting**: The system counts cars that cross a predefined line, representing a key part of vehicle counting for traffic analysis. The centroid of each car's bounding box is monitored to ensure it has passed the line before being counted.
- **Data Storage**: The final vehicle count is stored in an Excel file using the Pandas library, providing an easy way to analyze the data after the video processing.

This project can be extended to include other vehicle types, such as trucks and bikes, and adapted for real-time applications in traffic management and surveillance systems.


## YOLOv8 Architecture

The YOLOv8 architecture is composed of three main parts: **Backbone**, **Neck**, and **Head**. Each part plays a critical role in efficiently detecting and classifying objects within an image or video frame.

### 1. Backbone
The **backbone** is responsible for extracting meaningful features from the input image. YOLOv8 uses a modified CSP (Cross Stage Partial) Darknet architecture, which introduces several improvements:
- **CSPDarknet**: Enhances gradient flow by partitioning the feature map into two parts and passing only one through the residual block, improving computational efficiency.
- **Convolutional Layers**: YOLOv8 uses a stack of convolutional layers for feature extraction.
- **Activation Function**: It employs the SiLU (Sigmoid Linear Unit) activation function to enhance non-linearity and model expressiveness.
- **Residual Blocks**: These help the model learn more complex features by adding the input back into the output, avoiding the vanishing gradient problem.

### 2. Neck
The **neck** aggregates features from different levels of the backbone and enhances them for detection tasks. YOLOv8 uses the **Path Aggregation Network (PAN)** as its neck:
- **Feature Pyramid Network (FPN)**: This helps to merge feature maps from different layers, allowing the model to leverage both low-level and high-level features for detecting objects of different sizes.
- **PAN (Path Aggregation Network)**: PAN improves information flow by connecting bottom-up and top-down pathways, ensuring fine-grained and global features are available to the detection head.

### 3. Head
The **head** of YOLOv8 is responsible for predicting object classes, bounding boxes, and confidence scores. The improvements in the head are designed to provide:
- **Anchor-Free Detection**: YOLOv8 moves away from anchor-based methods, which makes it faster and reduces computational complexity. It directly predicts the center of the object, the width, height, and classification.
- **Bounding Box Regression**: YOLOv8 predicts precise bounding box coordinates for objects in the image.
- **Class Prediction**: The head outputs a confidence score for each object class detected.

### Summary of YOLOv8 Enhancements:
- **Anchor-Free Design**: This simplifies object detection, reducing hyperparameters and boosting speed.
- **Improved Backbone (CSPDarknet)**: More efficient feature extraction through cross-stage connections and residual blocks.
- **Improved Neck (PAN and FPN)**: Enhanced multi-scale feature aggregation for better detection of objects at different sizes.
- **Real-Time Speed**: As with previous YOLO versions, YOLOv8 prioritizes real-time performance while maintaining high accuracy.

### Applications of YOLOv8
- **Object Detection**: YOLOv8 is used for detecting objects in images and videos, including cars, people, animals, etc.
- **Real-Time Performance**: Suitable for real-time applications such as video surveillance, autonomous driving, and traffic monitoring.
- **Multi-Class Detection**: YOLOv8 can detect multiple object classes within a single image frame, making it versatile for a wide range of tasks.


### Running the Code

1. Install the necessary libraries:
    ```bash
    pip install ultralytics opencv-python pandas
    ```
2. Update the paths for your input video and Excel output in the script.
3. Run the script:
    ```bash
    python vehicle_detection.py
    ```

4. The output video will be saved in `CAR_detected.avi`, and the final car count will be saved to an Excel file.

![image1](https://github.com/user-attachments/assets/b23fb50d-f75d-4cdf-bf0c-c09b60313207)
![image2](https://github.com/user-attachments/assets/5be00ee9-4aea-474a-a4b2-42b9ec7488df)
![image3](https://github.com/user-attachments/assets/eee76b67-e431-469e-ab54-85c4ba0e6ae2)




## Conclusion

This project demonstrates a practical implementation of vehicle detection and counting using the YOLOv8 model. By processing video frames and analyzing the movement of cars across a designated line, the system can effectively count vehicles with high accuracy. The results are logged and stored in an Excel file for further analysis.

Future enhancements may include:
- Implementing multi-class detection (e.g., bikes, trucks).
- Improving the system for more complex traffic scenarios.
- Exploring real-time integration and performance optimization.

Overall, this solution offers a fast and efficient method for vehicle tracking in video surveillance and traffic management systems.

## References

1. **YOLOv8 Model Documentation**  
   [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/) - Official documentation for using YOLOv8 models for object detection.

2. **OpenCV for Video Processing**  
   [OpenCV Documentation](https://docs.opencv.org/) - Guide on using OpenCV for video capture, processing, and handling in Python.

3. **Pandas for Data Handling**  
   [Pandas Documentation](https://pandas.pydata.org/docs/) - Official documentation for using Pandas to manage data in Excel files.

4. **Vehicle Detection and Counting Tutorial**  
   [Vehicle Counting with YOLOv8](https://example.com/vehicle-detection) - An example tutorial on vehicle counting using YOLOv8.


