# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 02:56:27 2020

@author: Pranjal Vithlani
"""

import os
import pandas as pd
from sklearn import preprocessing



loc = './data/ucfTrainTestlist/'
#loc = '/common/users/pv189/pranjal_projects/UCF-101/'
out = './data/frame_data/'
nframes_per_clip = 8

not_use = ['v_BaseballPitch_g07_c02.avi', 'v_BaseballPitch_g08_c01.avi', 'v_Basketball_g13_c02.avi', 'v_Basketball_g13_c03.avi', 'v_Basketball_g13_c04.avi', 'v_Basketball_g16_c01.avi', 'v_Basketball_g16_c02.avi', 'v_Basketball_g16_c03.avi', 'v_Basketball_g16_c05.avi', 'v_Basketball_g16_c06.avi', 'v_Basketball_g18_c03.avi', 'v_Basketball_g18_c05.avi', 'v_Biking_g22_c03.avi', 'v_Biking_g23_c04.avi', 'v_Diving_g23_c06.avi', 'v_GolfSwing_g01_c05.avi', 'v_GolfSwing_g22_c01.avi', 'v_GolfSwing_g22_c02.avi', 'v_GolfSwing_g22_c03.avi', 'v_JugglingBalls_g18_c02.avi', 'v_JugglingBalls_g18_c03.avi', 'v_JugglingBalls_g18_c04.avi', 'v_JumpRope_g17_c01.avi', 'v_JumpRope_g17_c02.avi', 'v_JumpRope_g17_c04.avi', 'v_PlayingTabla_g15_c01.avi', 'v_PoleVault_g14_c04.avi', 'v_RockClimbingIndoor_g04_c03.avi', 'v_SoccerJuggling_g01_c01.avi', 'v_SoccerJuggling_g01_c04.avi', 'v_SoccerJuggling_g12_c03.avi', 'v_SoccerJuggling_g16_c01.avi', 'v_SoccerJuggling_g16_c02.avi', 'v_SoccerJuggling_g24_c06.avi', 'v_SoccerJuggling_g24_c07.avi', 'v_TennisSwing_g09_c04.avi', 'v_TennisSwing_g16_c01.avi', 'v_TennisSwing_g16_c02.avi', 'v_TennisSwing_g16_c04.avi', 'v_TennisSwing_g16_c05.avi', 'v_TennisSwing_g16_c06.avi', 'v_TrampolineJumping_g06_c04.avi', 'v_VolleyballSpiking_g22_c01.avi', 'v_WalkingWithDog_g06_c05.avi']

err = []
frames = []
category = []

class_names = sorted(os.listdir(loc))
le = preprocessing.LabelEncoder()

file_names = sorted(os.listdir(loc))

for file_name in file_names:
    file1 = open(loc+file_name,"r") 
    if file_name[:5] == 'train':
        for x in file1:
            frame = x.split(' ')[0].split('/')[1].split('.')[0]
            if frame+'.avi' not in not_use:
                frame = frame+"_frame0"+str(nframes_per_clip-1)+".jpg"
                
                frames.append(frame)
                category.append(x.split(' ')[0].split('/')[0])
    else:
        for x in file1:
            # print(file_name)
            # print(x)
            frame = x.split('/')[1].split('.')[0]
            if frame+'.avi' not in not_use:
                frame = frame+"_frame0"+str(nframes_per_clip-1)+".jpg"
                
                frames.append(frame)
                category.append(x.split('/')[0])
        
    file1.close() 
    

    le.fit(category)
    category_number = le.transform(category)
    
    dic = {'frames': frames,
            'category_name': category,
            'category_number' : category_number
            }
    
    df = pd.DataFrame(dic, columns= ['frames', 'category_name', 'category_number'])
    
    df.to_csv (r'./data/'+file_name[:-4]+'_frames'+str(nframes_per_clip)+'.csv', index = False, header=True)