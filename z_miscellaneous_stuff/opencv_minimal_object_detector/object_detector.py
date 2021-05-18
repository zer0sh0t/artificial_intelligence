import cv2

config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_path = 'frozen_inference_graph.pb'
class_file = 'coco.names'
with open(class_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

net = cv2.dnn_DetectionModel(weights_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

thresh = 0.6
nms_thresh = 0.4
cap = cv2.VideoCapture(0)
cap.set(3, 640) # width
cap.set(4, 480) # height

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    class_ids, confs, bboxes = net.detect(img, confThreshold=thresh)
    if len(class_ids) != 0:    
        bboxes = list(bboxes)
        confs = list(confs.reshape(1,-1)[0])
        confs = list(map(float, confs))
        indices = cv2.dnn.NMSBoxes(bboxes, confs, thresh, nms_thresh)

        for i in indices:
            i = i[0]
            bbox = bboxes[i]
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            cv2.rectangle(img, (x, y),(x + w, h + y), color=(0, 255, 0), thickness=2)
            cv2.putText(img, f'{class_names[class_ids[i][0] - 1]}({round(confs[i] * 100, 2)})', 
                                (bbox[0] + 10, bbox[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('output', img)
    cv2.waitKey(1)