#! /usr/bin/env python
# -*- coding: utf-8 -*-



from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import cv2
import numpy as np
print(cv2.__version__)


from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker  import Tracker
from tools import generate_detections as gdet
import imutils.video



warnings.filterwarnings('ignore')


def main():
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    asyncVideo_flag = False

    weightsPath="model_data/yolov3.weights"
    configPath="model_data/yolov3.cfg"
    labelsPath="model_data/coco.names"
    file_path = 'test_video/test_video1.avi'

    LABELS = open(labelsPath).read().strip().split("\n")  #物体类别
    video_capture = cv2.VideoCapture(file_path)
    if asyncVideo_flag:
        video_capture.start()
    if writeVideo_flag:
        if asyncVideo_flag:
            w = int(video_capture.cap.get(3))
            h = int(video_capture.cap.get(4))
        else:
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('test_video/output_video.mp4', fourcc, 30, (w, h))
        frame_index = -1

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()

    num_id=[]#保存id
    img_next_name=int(0)
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
             break

        t1 = time.time()
        # 必须将boxes在遍历新的图片后初始化
        boxs = []
        confidence1 = []
        classIDs = []
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        (H, W) = frame.shape[:2]
        # 得到 YOLO需要的输出层
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        #从输入图像构造一个blob，然后通过加载的模型，给我们提供边界框和相关概率
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        #在每层输出上循环
        for output in layerOutputs:
            # 对每个检测进行循环
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                #过滤掉那些置信度较小的检测结果
                if confidence > 0.9:
                    #框后接框的宽度和高度
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    #边框的左上角
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # 更新检测出来的框
                   # 批量检测图片注意此处的boxes在每一次遍历的时候要初始化，否则检测出来的图像框会叠加
                    boxs.append([x, y, int(width), int(height)])
                    #print(boxes)
                    confidence1.append(float(confidence))
                    #print(confidences)
                    classIDs.append(classID)
        # print('boxs:',boxs)
        # print('confidence1:',confidence1)
        # print('classIDs:',classIDs)

        features = encoder(frame,boxs)
        detections = [Detection(bbox, confidence, feature,classID) for bbox, confidence ,feature ,classID in zip(boxs, confidence1, features,classIDs)]
        #print(detections)
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        #print(boxes)# 还是检测时的框大小
        scores = np.array([d.confidence for d in detections])

        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        #print(type(indices[0]))
        detections = [detections[i] for i in indices]
        #print(detections) #list

        #confidence8 = [detections[i].confidence for i in indices]
        #print(confidence8)

        #type_name = [LABELS[classIDs[i]] for i in indices]
        #print(type_name)

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        #print(type(tracker)) #Tracker类
        #print(tracker.tracks) #list 用于保存track类

        m=0
        boxes = []#框大小 必须int类型
        confidence2=[]#跟踪置信度
        type1=[]#跟踪类别
        #获取id，如果id是新的id，则将这个图像框截取下来
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                    continue
            bbox = track.to_tlbr()
            num=track.track_id
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]) ),(0,255,0), 2)#绿色
            cv2.putText(frame, LABELS[track.classID]+str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)#绿色
            # print('num_id',num)
            if num in num_id:
                continue
            else:
                m=m+1
                # print('一张图标记了 {} 个'.format(m))
                #print(track) #track类
                num_id.append(num)
                bbox = track.to_tlbr()

                xm=track.confidence
                # print('置信度：',xm)

                xm2 = track.classID
                # print("类别编号：",xm2)


                boxes.append([ int(bbox[0]), int(bbox[1]) , int(bbox[2]) , int(bbox[3]) ])#跟踪后的box
                confidence2.append(float(xm))#跟踪后的置信度
                type1.append(LABELS[xm2])#跟踪类别

        cv2.imshow('', frame)
        # print('boxes:',boxes)
        # print('confidence2',confidence2)
        # print('type1',type1)

        #confidences=[]#临时保存中间置信度（根据边框多少分配置信度）
        type2=[]#临时保存中间类别（根据边框多少分配置类别）

        if confidence2:
            # print('type1',type1)
            # print('confidences:',confidence2)
            # 极大值抑制
            idxs = cv2.dnn.NMSBoxes(boxes, confidence2, 0.5,0.3)
            img_pre_name='_00000'
            if len(idxs) > 0:
                for i in idxs.flatten() :
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    # cv2.rectangle(frame, (x, y), (w,  h), (0,255,0), 2)
                    # cv2.imshow('', frame)
                    #print(i)
                    savepath = r"img/out_img1"  # 图像保存地址
                    savepath=savepath+'/'+type1[i]
                    # print('savepath',savepath)
                    # 如果输出的文件夹不存在，创建即可
                    if not os.path.isdir(savepath):
                        os.makedirs(savepath)
                    cut = frame[y:h, x:w]
                    if cut.size != 0:
                        # boxes的长度即为识别出来的车辆个数，利用boxes的长度来定义裁剪后车辆的路径名称
                        img_next_name=img_next_name+1
                        # print(type1[i]+img_pre_name+str(img_next_name)+'.jpg')
                        filename=type1[i]+img_pre_name+str(img_next_name)+'.jpg'

                        cv2.imwrite(savepath+"/"+filename,cut)

        if writeVideo_flag: # and not asyncVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps_imutils.update()

        fps = (fps + (1./(time.time()-t1))) / 2
        print("FPS = %f"%(fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()




