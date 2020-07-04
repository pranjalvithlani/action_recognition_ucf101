# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 17:09:03 2020

@author: Pranjal Vithlani
"""

import cv2
import os


nframes_per_clip = 8 # 8 equally distributed frames from video
loc = './data/UCF-101/'
loc = '/common/users/pv189/pranjal_projects/UCF-101/'
out = './data/frame_data/'
if not os.path.exists(out):
    os.mkdir(out)
    
class_names = sorted(os.listdir(loc))
#print(len(class_names))
err  = []

for class_id in class_names:
    class_path = loc + class_id + '/'
    if not os.path.exists(out + class_id + '/'):
        os.mkdir(out + class_id + '/')
        
    for video_clip in sorted(os.listdir(class_path)):
        clip_path = class_path + video_clip

        vidcap = cv2.VideoCapture(clip_path)
        success,image = vidcap.read()
        count = 0
        total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)-1
        #print(total_frames)
        #fps = vidcap.get(cv2.CAP_PROP_FPS)
        multiplier = total_frames // nframes_per_clip
        #print(multiplier)
        while success:
            frameId = int(round(vidcap.get(1)))
            if frameId % multiplier == 0:
                cv2.imwrite(out + class_id + '/'+video_clip[:-4]+"_frame%02d.jpg" % count, image)  # save frame as JPEG file
                count+=1
                
            if count == 8:
                break
            success,image = vidcap.read()
            
            
        
        vidcap.release()
        if count == 8:
            print("done: "+video_clip)
        else:
            print("----------not done: "+video_clip)
            err.append(video_clip)

print(err)
    
