import cv2

video = cv2.VideoCapture('Minimal Office - Desk Tour.mp4')

# Parameters for tuning
detectionThresholdValue = 0.55 #if the model detects a PERSON with equal or more than 55% accuracy then it will be shown in the output

# Video Output Ratio
video.set(3, 1280)
video.set(4, 720)

# Importing list of categories from the coco.names
categories=[]
categoryFilename = 'cocoLARGE.names'

with open(categoryFilename, 'rt') as f:
    categories = f.read().rstrip('\n').split('\n') # read the file f and strip ans split the name based on new line (\n)

print(categories)

# configs and weights (1)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' # Balanced between Accuracy and Speed of detection
weightsPath = 'frozen_inference_graph.pb'

# configs and weights (2) much slower
# configPath = 'yolov3.cfg'
# weightsPath = 'yolov3.weights'

# Deep Neural Network (DNN) Model development
model_01 = cv2.dnn_DetectionModel(weightsPath, configPath) #from openCV, few parameters can be changed from documentation
model_01.setInputSize(320, 320)
model_01.setInputScale(1.0/ 127.5) #Why ??
model_01.setInputMean((127.5, 127.5, 127.5))
model_01.setInputSwapRB(True)

while True: 
    success, img = video.read()
    categoryIDs, confidence, bbox = model_01.detect(img, confThreshold=detectionThresholdValue) # list of three parameters, confThreshold=0.5 means, detection accuracy 50%
    print(categoryIDs, bbox) # [1] [[ 60  40 373 461]], here 1 means 1st name in the coco.names
    
    if len(categoryIDs)!= 0: 
        # The fun part - Detected 
        for catID, confi, box in zip(categoryIDs.flatten(), confidence.flatten(), bbox): # zip() function returns a zip object contains multiple lists or tuples. flatten
            cv2.rectangle(img, box, color=(0,255,0), thickness = 3)
            # cv2.putText(img, categories[catID].upper(),(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            # cv2.putText(img, "({:.3f}%)".format(confidence[0]*100.0),(box[0]+150,box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.putText(img, "{} {}".format(categories[catID].upper(),"({:.2f}%)".format(confidence[0]*100.0)),(box[0]+10,box[1]-13), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Output", img)
    cv2.waitKey(1)







