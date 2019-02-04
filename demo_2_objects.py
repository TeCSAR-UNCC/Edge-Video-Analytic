
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw,ImageFont
import cv2
import os
from skimage.measure import compare_ssim as ssim
import time

from yolo_v3_2_objects import yolo_v3, load_weights, detections_boxes, non_max_suppression,_iou
import yolo_v3_tiny

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_img', '', 'Input image')
tf.app.flags.DEFINE_string('output_img', '', 'Output image')
tf.app.flags.DEFINE_string('class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string('weights_file', 'yolov3-tiny.weights', 'Binary file with detector weights')

tf.app.flags.DEFINE_integer('size', 416, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')


def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names
    
def deduce_index(index,resolution):
    flatten_index=index*85
    block_numb=flatten_index/255
    row_n=block_numb/resolution + 1
    col_n=block_numb%resolution
    #print(row_n)
    #print(col_n)
    return row_n,col_n


def draw_boxes(filtered_boxes,feature_maps, img, cls_names, detection_size,inter_resolution_threshold,num_obj,iteration,obj_num_frame):
    draw = ImageDraw.Draw(img)
    count=0
    obj_count=0
    box_list=[]
    box_cls=[]
    bbox_list=[]
    #feature_list_13=[]
    #feature_list_26=[]
    #feature_list_52=[]
    feature_list=[]
    ssim_array=np.zeros([obj_num_frame,num_obj])
    for boxes in filtered_boxes:
       
        for cls, bboxs in boxes.items():
            #color = tuple(255)
            if cls==0:
                for box, score, index in bboxs:
                    draw_box=True
                    
                    if count==1:
                        if iteration==246:
                            print('box list')
                            print(box_list)
                            print(box_cls)
                        for b,cl in zip(box_list,box_cls):
                            iou=0.0
                            if cl==cls:
                                iou=_iou(b,box)
                                #print(iou)
                                if(iou>inter_resolution_threshold):
                                    draw_box=False
                        if(draw_box):
                            if(iteration==0):
                                if(obj_count>=num_obj):
                                    break
                                feature_map=feature_maps[count]
                                row_n,col_n=deduce_index(int(index),26)
                                feature_vector=feature_map[0,row_n,col_n,:]
                                feature_list.append(feature_vector)
                                box_for_iou=box
                                #box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
                                draw.rectangle(box, outline='green')
                                font = ImageFont.truetype("./arial.ttf", 15)
                                draw.text(box[:2], 'Object {} '.format(obj_count), fill='white', font=font)
                                bbox_list.append(box)
                                #feature_vector_array[0][obj_count]=feature_vector
                                #bbox_coord_array[0][obj_count]=box
                                prev_feature_vector[iteration][obj_count]=feature_vector
                                obj_count+=1
                            else:
                                feature_map=feature_maps[count]
                                row_n,col_n=deduce_index(int(index),26)
                                feature_vector=feature_map[0,row_n,col_n,:]
                             
                                box_for_iou=box
                             #box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
                                ssim_index=np.zeros([num_obj])
                                for feature in range(num_obj):
                                    ssim_index[feature]=ssim(feature_vector,prev_feature_vector[0][feature])
                                ssim_array[obj_count]=ssim_index  
                                bbox_list.append(box)
                                feature_list.append(feature_vector)
                                obj_count+=1    
    
                    elif count==2:
                        
                        for b,cl in zip(box_list,box_cls):
                            iou=0.0
                            if cl==cls:
                                iou=_iou(b,box)
                                #print(iou)
                                if(iou>inter_resolution_threshold):
                                    draw_box=False
                        if(draw_box):  
                            if(iteration==0):
                                if(obj_count>=num_obj):
                                    break
                                feature_map=feature_maps[count]
                                row_n,col_n=deduce_index(int(index),52)
                                feature_vector=feature_map[0,row_n,col_n,:]
                                feature_list.append(feature_vector)
                                box_for_iou=box
                                #box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
                                draw.rectangle(box, outline='green')
                                font = ImageFont.truetype("./arial.ttf", 15)
                                draw.text(box[:2], 'Object {} '.format(obj_count), fill='white',font=font)
                                bbox_list.append(box)
                                #feature_vector_array[0][obj_count]=feature_vector
                                #bbox_coord_array[0][obj_count]=box
                                prev_feature_vector[iteration][obj_count]=feature_vector
                                obj_count+=1
                            else:
                               feature_map=feature_maps[count]
                               row_n,col_n=deduce_index(int(index),52)
                               feature_vector=feature_map[0,row_n,col_n,:]
                                 
                               box_for_iou=box
                                 #box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
                               ssim_index=np.zeros([num_obj])
                               for feature in range(num_obj):
                                   ssim_index[feature]=ssim(feature_vector,prev_feature_vector[0][feature])
                               ssim_array[obj_count]=ssim_index  
                               bbox_list.append(box)
                               feature_list.append(feature_vector)
                               obj_count+=1
    
                    else:
                         #print(feature_map.shape)
                         #feature_list_13.append(feature_map[0,row_n,col_n,:])
                         if(iteration==0):
                             if(obj_count>=num_obj):
                                 break
                             feature_map=feature_maps[count]
                             row_n,col_n=deduce_index(int(index),15)
                             feature_vector=feature_map[0,row_n,col_n,:]
                             feature_list.append(feature_vector)
                             box_for_iou=box
                             #box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
                             draw.rectangle(box, outline='green')
                             font = ImageFont.truetype("./arial.ttf", 15)
                             draw.text(box[:2], 'Object {} '.format(obj_count), fill='white',font=font)
                             bbox_list.append(box)
                             #feature_vector_array[0][obj_count]=feature_vector
                             #bbox_coord_array[0][obj_count]=box
                             prev_feature_vector[iteration][obj_count]=feature_vector
                             obj_count+=1
                         else:
                             feature_map=feature_maps[count]
                             row_n,col_n=deduce_index(int(index),13)
                             feature_vector=feature_map[0,row_n,col_n,:]
                             
                             box_for_iou=box
                             #box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
                             ssim_index=np.zeros([num_obj])
                             for feature in range(num_obj):
                                 ssim_index[feature]=ssim(feature_vector,prev_feature_vector[0][feature])
                             ssim_array[obj_count]=ssim_index  
                             bbox_list.append(box)
                             feature_list.append(feature_vector)
                             obj_count+=1

                    box_list.append(box_for_iou)
                    box_cls.append(cls)
                
        count+=1  
    print('Iteration:')
    print(iteration)
    if(iteration>0):

        ssim_array=ssim_array[~np.all(ssim_array == 0, axis=1)]

        if len(ssim_array)>0:
            feature_bbox_tensor_builder(ssim_array,feature_list,bbox_list,num_obj,draw)

    return feature_list,bbox_list
    
def feature_bbox_tensor_builder(ssim_array,feature_list,bbox_list,num_obj,draw):

     max_ssim_index_col=np.argmax(ssim_array,axis=0)
     max_ssim_index_row=np.argmax(ssim_array,axis=1) 
     max_val_col=ssim_array.max(axis=0)
     max_val_row=ssim_array.max(axis=1)
     obj_obtained=0
     num_obj_det=ssim_array.shape[0]
     boxes_taken=[]
     #print(max_val_col)
     #print(max_val_row)    
     #print(ssim_array)
     max_box_array=np.nonzero(np.in1d(max_val_row, max_val_col))[0]
     #print('max_box_array:')
     #print(max_box_array)
     for i in range(len(max_box_array)):
         if obj_obtained==num_obj:
             break
         
         max_id=max_box_array[i]
         obj_numb=np.where(max_val_col==max_val_row[max_id])[0]
         draw.rectangle(bbox_list[max_id], outline='green')
         font = ImageFont.truetype("./arial.ttf", 15)
         draw.text(bbox_list[max_id][:2], 'Object {}'.format(obj_numb[0]), fill='white',font=font) 
         prev_feature_vector[1][obj_numb]=feature_list[max_id]
         boxes_taken.append(max_id)
         #print('run for')
         obj_obtained+=1
     next_pass=0
     unoccu_objects=np.unique(np.where(prev_feature_vector[1]==0)[0])
     print('unoccu_objects')
     print(unoccu_objects)
     if obj_obtained<num_obj_det:
         while len(unoccu_objects)>0 and obj_obtained<num_obj:
             col_max=max_ssim_index_col[unoccu_objects[0]]
             if col_max not in boxes_taken:
                 #print('run while')
                 draw.rectangle(bbox_list[col_max], outline='green')
                 font = ImageFont.truetype("./arial.ttf", 15)
                 draw.text(bbox_list[col_max][:2], 'Object {}'.format(unoccu_objects[0]), fill='white',font=font) 
                 prev_feature_vector[1][unoccu_objects[0]]=feature_list[col_max]
                 boxes_taken.append(col_max)
                 obj_obtained+=1
                 unoccu_objects=unoccu_objects[1:]
             else:
                 next_pass+=1
                 ssim_col=ssim_array[:,col_max]
                 new_max=np.argsort(ssim_col)[:-1][-next_pass]
                 max_ssim_index_col[unoccu_objects[0]]=new_max
                 continue
    
                                
def convert_to_original_size(box, size, original_size):
    x_scale=float(original_size[0])/float(size[0])
    y_scale=float(original_size[1])/float(size[1])
    #print(x_scale,y_scale)
    x_upper_left=int(np.round(box[0]*x_scale))
    y_upper_left=int(np.round(box[1]*y_scale))
    x_lower_right=int(np.round(box[2]*x_scale))
    y_lower_right=int(np.round(box[3]*y_scale))
    x_center=float(x_lower_right-x_upper_left)/2.0
    y_center=float(y_lower_right-y_upper_left)/2.0
    return [x_upper_left,y_upper_left,x_lower_right,y_lower_right]

#def get_feature_vectors(results, feature_maps):
    




num_obj=2
prev_feature_vector=np.zeros([2,num_obj,255])
#feature_vector_array=np.zeros([test_set_length,num_obj,255])
#bbox_coord_array=np.zeros([test_set_length,num_obj,4])
test_dict={}  
img_names_dict={} 
img_count=0
#image=cv2.imread(test_path)
#img_resized = cv2.resize(image,(416, 416))

classes = load_coco_names(FLAGS.class_names)
inter_resolution_threshold=0.38
# placeholder for detector inputs
#inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3])
BATCH_SIZE = 1
inputs = tf.placeholder(tf.float32, [BATCH_SIZE, FLAGS.size, FLAGS.size, 3])
img_count=0

with tf.variable_scope('detector'):
    detections,feature_map1 = yolo_v3_tiny.yolo_v3_tiny2(inputs, len(classes), data_format='NHWC')
    load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)
    #box1=detections_boxes(detect_1)
    #box2=detections_boxes(detect_2)
    #box3=detections_boxes(detect_3)
    

boxes,feature_map_yolo = detections_boxes(detections,feature_map1)

#added
cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
vc = cv2.VideoCapture(0)
rval, frame = vc.read()
#end added

with tf.Session() as sess:
    sess.run(load_ops)
    i=0;
    while True:
        start_time = time.time()  
        #for x in range(BATCH_SIZE)      
        rval, frame = vc.read()
        img = Image.fromarray(frame)
        img_resized = img.resize(size=(FLAGS.size, FLAGS.size))
        detected_boxes,feature_maps = sess.run([boxes,feature_map_yolo], feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})
        filtered_boxes, obj_num_frame = non_max_suppression(detected_boxes,np.array((FLAGS.size, FLAGS.size)), np.array(img.size),confidence_threshold=FLAGS.conf_threshold,iou_threshold=FLAGS.iou_threshold)
        #print(filtered_boxes)
        feature_list,bbox_list=draw_boxes(filtered_boxes,feature_maps, img, classes, (FLAGS.size, FLAGS.size),inter_resolution_threshold,num_obj,i, obj_num_frame)
        #np.save('./feature_map_13.npy',np.array(feature_list_13))
        #print(bbox_list)
        cv2.imshow("preview",np.array(img))
        key = cv2.waitKey(20)
        FPS = BATCH_SIZE/(time.time()-start_time);
        print('FPS ', FPS)
        if key == 27: # exit on ESC
            break
        if i>0:
            prev_feature_vector=np.flip(prev_feature_vector,axis=0)
            prev_feature_vector[1]=0
        i=i+1
        if(i>100000):        
            i=1
    cv2.destroyWindow("preview")
