
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
    block_numb=index/3
    row_n=block_numb/resolution 
    col_n=block_numb%resolution
    rem=index%3
    print(row_n)
    print(col_n)
    print(rem)
    return row_n,col_n,rem


def draw_boxes(filtered_boxes,feature_maps, img, cls_names, detection_size,inter_resolution_threshold,num_obj,iteration,obj_num_frame,missing_objects_fv,missing_objects_pix,new_objects_fv,new_objects_pix):
    draw = ImageDraw.Draw(img)
    count=0
    obj_count=0
    box_list=[]
    box_cls=[]
    bbox_list=[]
    global object_track_array
    global noobj_counter
    global obj_counter
    global missing_object_pos
    global prev_feature_vector
    global prev_bbox_pix
    global obj_tracker 
    #feature_list_13=[]
    #feature_list_26=[]
    #feature_list_52=[]
    noobj_counter_prev_frame=0
    obj_counter_prev_frame=0
    feature_list=[]
    ssim_array=np.zeros([obj_num_frame,num_obj])
    #print("Filtered boxes: ", filtered_boxes)
    for boxes in filtered_boxes:
        #print("Box instance: ", boxes)
        for cls, bboxs in boxes.items():
            #print("BBox instance: ", bboxs)
            #color = tuple(255)
            if cls==0 and obj_num_frame>0:
                for box, score, index in bboxs:
                    #print("Little box: ", box)
                    draw_box=True
                    #print("Count ",count)
                    if count==1:
                        dimension = 26
                    elif count==2:
                        dimension = 52
                    else:
                        dimension = 13
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
                        if(obj_tracker==0):
                            if(obj_count>=num_obj):
                                break
                            feature_map=feature_maps[count]
                            row_n,col_n,rem=deduce_index(int(index),dimension)                              
#                                if rem==0:
#                                    feature_vector=feature_map[0,row_n,col_n,:85]
#                                elif rem==1:
#                                    feature_vector=feature_map[0,row_n,col_n,85:170]
#                                else:
#                                    feature_vector=feature_map[0,row_n,col_n,170:]
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
                            prev_feature_vector[obj_tracker][obj_count]=feature_vector
                            prev_bbox_pix[obj_tracker][obj_count]=box
                            object_track_array[obj_tracker][obj_count]=1
                            obj_count+=1
                            obj_tracker+=1
                        else:
                            feature_map=feature_maps[count]
                            row_n,col_n,rem=deduce_index(int(index),dimension)
#                                if rem==0:
#                                    feature_vector=feature_map[0,row_n,col_n,:85]
#                                elif rem==1:
#                                    feature_vector=feature_map[0,row_n,col_n,85:170]
#                                else:
#                                    feature_vector=feature_map[0,row_n,col_n,170:]
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
                            obj_tracker+=1    

                    box_list.append(box_for_iou)
                    box_cls.append(cls)
                
        count+=1  
    print('Iteration:')
    print(iteration)
    #print('Box list', box_list, 'End of box list')
#        ssim_mask=ssim_array>=0.2
#        ssim_array=ssim_array*ssim_mask
    ssim_array=ssim_array[~np.all(ssim_array == 0, axis=1)]
#        max_ssim_index_col=np.argmax(ssim_array,axis=0)
#        max_ssim_index_row=np.argmax(ssim_array,axis=1)
    #max_ssim=[]
    #box_number=0
#        print(max_ssim_index_col)
#        print(max_ssim_index_row)
    if len(bbox_list)>0:
       obj_counter+=1
       noobj_counter=0
    else:
       noobj_counter+=1
       obj_counter=0
    #print(ssim_array)
    if len(ssim_array)>0 and obj_tracker>0 :
        missing_objects_fv,missing_objects_pix,new_objects_fv,new_objects_pix=feature_bbox_tensor_builder(ssim_array,feature_list, bbox_list,num_obj,draw,missing_objects_fv,missing_objects_pix,new_objects_fv,new_objects_pix)
   
    if noobj_counter==1 and obj_counter==0:
        if (object_track_array[0]!=object_track_array[1]).any:
            changed_elements=np.where((object_track_array[0]!=object_track_array[1])==True)[0]
            print(changed_elements)
            for i in range(len(changed_elements)):
                if object_track_array[1][changed_elements[i]]==0:
                    print('changed element')
                    print(changed_elements[i])
                    fv=tuple(prev_feature_vector[0][changed_elements[i]])
                    pix=tuple(prev_bbox_pix[0][changed_elements[i]])
                    missing_objects_fv.append(fv)
                    missing_objects_pix.append(pix) 
                    missing_object_pos.append(changed_elements[i])

    return feature_list,bbox_list,missing_objects_fv,missing_objects_pix,new_objects_fv,new_objects_pix
    
def feature_bbox_tensor_builder(ssim_array,feature_list,bbox_list,num_obj,draw,missing_objects_fv,missing_objects_pix,new_objects_fv,new_objects_pix):
     global prev_feature_vector
     global object_track_array
     global prev_bbox_pix
     global missing_object_pos
     max_ssim_index_col=np.argmax(ssim_array,axis=0)
     max_ssim_index_row=np.argmax(ssim_array,axis=1) 
     max_val_col=ssim_array.max(axis=0)
     max_val_row=ssim_array.max(axis=1)
     obj_obtained=0
     num_obj_det=ssim_array.shape[0]
     boxes_taken=[]
     new_object_featurelist=[]
     new_object_boxpixlist=[]
     print(max_val_col)
     print(max_val_row)    
     print(ssim_array)
     max_box_array=np.nonzero(np.in1d(max_val_row, max_val_col))[0]
     #print('max_box_array:')
     #print(max_box_array)
     if noobj_counter==0 and obj_counter>1: 
         for i in range(len(max_box_array)):
             if obj_obtained==num_obj:
                 break
             
             max_id=max_box_array[i]
             obj_numb=np.where(max_val_col==max_val_row[max_id])[0]
             draw.rectangle(bbox_list[max_id], outline='green')
             font = ImageFont.truetype("./arial.ttf", 15)
             draw.text(bbox_list[max_id][:2], 'Object {}'.format(obj_numb[0]), fill='white',font=font) 
             prev_feature_vector[1][obj_numb]=feature_list[max_id]
             prev_bbox_pix[1][obj_numb]=bbox_list[max_id]
             object_track_array[1][obj_numb]=1
             boxes_taken.append(max_id)
             #print('run for')
             obj_obtained+=1
         next_pass=0    
         unoccu_objects=np.unique(np.where(prev_feature_vector[1]==0)[0])
         new_object_ssim_array=np.zeros([len(unoccu_objects),len(missing_objects_pix)])
         ssim_new_obj_count=0
         #print(boxes_taken)
         print('unoccu_objects', unoccu_objects)
         print('num_obj_det', num_obj_det)
         print('obj_obtained', obj_obtained)
         if obj_obtained<num_obj_det:
             while len(unoccu_objects)>0 and obj_obtained<num_obj_det:
                 col_max=max_ssim_index_col[unoccu_objects[0]]
                 print('col_max')
                 print(col_max)
                 if col_max not in boxes_taken:
                     if object_track_array[0][unoccu_objects[0]]==1 or len(missing_objects_pix)<1:
                         print('run while')
                         draw.rectangle(bbox_list[col_max], outline='green')
                         font = ImageFont.truetype("./arial.ttf", 15)
                         draw.text(bbox_list[col_max][:2], 'Object {}'.format(unoccu_objects[0]), fill='white',font=font) 
                         prev_feature_vector[1][unoccu_objects[0]]=feature_list[col_max]
                         prev_bbox_pix[1][unoccu_objects[0]]=bbox_list[col_max]
                         object_track_array[1][unoccu_objects[0]]=1
                         boxes_taken.append(col_max)
                         obj_obtained+=1
                         unoccu_objects=unoccu_objects[1:]
                         print(obj_obtained)
                         print(obj_obtained<num_obj_det)
                     else:
                         new_object_ssim_ar=np.zeros([len(missing_objects_pix)])
                         for j in range(len(missing_objects_pix)):
                             new_object_ssim_ar[j]=ssim(feature_list[col_max],np.asarray(missing_objects_fv[j]))
                         new_object_featurelist.append(feature_list[col_max])
                         new_object_boxpixlist.append(bbox_list[col_max])
                         print(new_object_ssim_ar) 
                         print(ssim_new_obj_count)
                         new_object_ssim_array[ssim_new_obj_count]=new_object_ssim_ar
                         obj_obtained+=1
                         unoccu_objects=unoccu_objects[1:]
                         ssim_new_obj_count+=1       
                 else:
                     print('else')
                     print(boxes_taken)
                     print(object_track_array[0][unoccu_objects[0]])
                     next_pass+=1
                     ssim_col=ssim_array[:,unoccu_objects[0]]
                     new_max=np.argsort(ssim_col)[:-1][-next_pass]
                     max_ssim_index_col[unoccu_objects[0]]=new_max
                     continue
         if (object_track_array[0]!=object_track_array[1]).any:
            changed_elements=np.where((object_track_array[0]!=object_track_array[1])==True)[0]
            print(changed_elements)
            for i in range(len(changed_elements)):
                if object_track_array[1][changed_elements[i]]==0:
                    print('changed element')
                    print(changed_elements[i])
                    fv=tuple(prev_feature_vector[0][changed_elements[i]])
                    pix=tuple(prev_bbox_pix[0][changed_elements[i]])
                    missing_objects_fv.append(fv)
                    missing_objects_pix.append(pix) 
                    missing_object_pos.append(changed_elements[i])
        
     
     if noobj_counter==0 and obj_counter==1:
         ssim_new_obj_count=0
         new_object_ssim_array=np.zeros([1,len(missing_objects_pix)])
         new_object_ssim_ar=np.zeros([len(missing_objects_pix)])
         for j in range(len(missing_objects_pix)):
             new_object_ssim_ar[j]=ssim(feature_list[0],np.asarray(missing_objects_fv[j]))
         new_object_featurelist.append(feature_list[0])
         new_object_boxpixlist.append(bbox_list[0])
         print(new_object_ssim_ar) 
         print(ssim_new_obj_count)
         new_object_ssim_array[ssim_new_obj_count]=new_object_ssim_ar
         obj_obtained+=1
         #unoccu_objects=unoccu_objects[1:]
         ssim_new_obj_count+=1
         
         
     if len(new_object_featurelist)>0:
        new_object_ssim_array=new_object_ssim_array.reshape([-1,len(missing_object_pos)]) 
        new_object_ssim_array=new_object_ssim_array[~np.all(new_object_ssim_array == 0, axis=1)]
        if new_object_ssim_array.shape[0]==1 and new_object_ssim_array.shape[1]>=1:
                  max_new_index_row=np.argmax(new_object_ssim_array,axis=1)
                  new_object_pos=max_new_index_row[0]
                  draw.rectangle(new_object_boxpixlist[0], outline='green')
                  font = ImageFont.truetype("./arial.ttf", 15)
                  draw.text(new_object_boxpixlist[0][:2], 'Object {}'.format(missing_object_pos[new_object_pos]), fill='white',font=font) 
                  print(missing_object_pos)
                  prev_feature_vector[1][missing_object_pos[new_object_pos]]=new_object_featurelist[0]
                  prev_bbox_pix[1][missing_object_pos[new_object_pos]]=new_object_boxpixlist[0]
                  object_track_array[1][missing_object_pos[new_object_pos]]=1  
                  #obj_obtained+=1
                  del missing_object_pos[new_object_pos]
                  del missing_objects_fv[new_object_pos]
                  del missing_objects_pix[new_object_pos]         
     #missing_objects_fv,missing_objects_pix,new_objects_fv,new_objects_pix=check_track_objects(missing_objects_fv,missing_objects_pix,new_objects_fv,new_objects_pix)
     print(new_object_ssim_array.shape)    
     print(missing_object_pos)
     print(missing_objects_pix)
     print(new_object_ssim_array)
     prev_feature_vector=np.flip(prev_feature_vector,axis=0)
     prev_feature_vector[1]=0 
     prev_bbox_pix=np.flip(prev_bbox_pix,axis=0)
     prev_bbox_pix[1]=0
     object_track_array=np.flip(object_track_array,axis=0)
     object_track_array[1]=0
     #print(new_objects_pix)
     #print(missing_objects_pix)
     return missing_objects_fv,missing_objects_pix,new_objects_fv,new_objects_pix

def check_track_objects(missing_objects_fv,missing_objects_pix,new_objects_fv,new_objects_pix):
#    global object_track_array
#    global prev_feature_vector
#    global prev_bbox_pix
    print(object_track_array)
    print(prev_bbox_pix)
    print(object_track_array[0]!=object_track_array[1])
    if (object_track_array[0]!=object_track_array[1]).any:
        changed_elements=np.where((object_track_array[0]!=object_track_array[1])==True)[0]
        print(changed_elements)
        for i in range(len(changed_elements)):
            if object_track_array[1][changed_elements[i]]==0:
                print('object missing')
                fv=tuple(prev_feature_vector[0][changed_elements[i]])
                pix=tuple(prev_bbox_pix[0][changed_elements[i]])
                missing_objects_fv.append(fv)
                missing_objects_pix.append(pix)
            if object_track_array[1][changed_elements[i]]==1:
                print('new object')
                fv=tuple(prev_feature_vector[1][changed_elements[i]])
                pix=tuple(prev_bbox_pix[1][changed_elements[i]])
                new_objects_fv.append(fv)
                new_objects_pix.append(pix)
    print(new_objects_pix)
    print(missing_objects_pix)
    #print(new_objects_fv)
    #print(missing_objects_fv)
    return missing_objects_fv,missing_objects_pix,new_objects_fv,new_objects_pix
                                
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
    

global object_track_array
global prev_feature_vector
global prev_bbox_pix
global missing_object_pos
global noobj_counter
global obj_counter
global obj_tracker

#global missing_objects_fv
#global missing_objects_pix
#global new_objects_fv
#global new_objects_pix

obj_tracker=0
num_obj=3
prev_feature_vector=np.zeros([2,num_obj,255])
prev_bbox_pix=np.zeros([2,num_obj,4])
object_track_array=np.zeros([2,num_obj])
missing_objects_fv=[]
missing_objects_pix=[]
new_objects_fv=[]
new_objects_pix=[]
missing_object_pos=[]
noobj_counter=0
obj_counter=0
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
        print(filtered_boxes)
        print(prev_bbox_pix)
        feature_list,bbox_list,missing_objects_fv,missing_objects_pix,new_objects_fv,new_objects_pix=draw_boxes(filtered_boxes,feature_maps, img, classes, (FLAGS.size, FLAGS.size),inter_resolution_threshold,num_obj,i, obj_num_frame,missing_objects_fv,missing_objects_pix,new_objects_fv,new_objects_pix)
        #np.save('./feature_map_13.npy',np.array(feature_list_13))
        #print(bbox_list)
        cv2.imshow("preview",np.array(img))
        key = cv2.waitKey(20)
	DELAY = time.time()-start_time
	DELAYMS = DELAY * 1000
        FPS = BATCH_SIZE/(DELAY);
        #print('DELAY', DELAY)        
        #print('FPS ', FPS)
        print('Output Delay: {0:.2f}ms'.format(DELAYMS))
        print('FPS: {0:.2f}'.format(FPS)) 
        if key == 27: # exit on ESC
            break
      
        i=i+1
        if(i>100000):        
            i=1
    cv2.destroyWindow("preview")
