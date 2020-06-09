# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:07:39 2020

@author: ADMIN
"""
import cv2

class ImageObj:
    
    def __init__(self,image,coorx,coory):
        self.image=image
        self.coorx=coorx
        self.coory=coory
        self.coorx_match=0
        self.coory_match=0
        self.w=150
        self.h=500
        self.px_marc1=0
        self.py_marc1=0
        self.px_marc2=image.shape[1]
        self.py_marc2=image.shape[0]
        
      
       
    def draw_rectangle(self):
        imgc=self.image.copy()
        cv2.rectangle(imgc, (self.coorx,self.coory),(self.coorx+self.w,self.coory+self.h) , (0,0,255), 3)
        #cv2.rectangle(imgc, (self.coorx,self.coory),(self.coorx+3,self.coory+3) , (0,0,255), 3)
        if(self.px_marc1 !=0 and self.px_marc2 != 0):
            cv2.rectangle(imgc, (self.px_marc1,self.py_marc1),(self.px_marc2,self.py_marc2) , (57,255,20),4)      
        
        return imgc
    
    def draw_circle(self):
        imgc=self.image.copy()
        cv2.circle(imgc, (self.coorx,self.coory) , 8, (255,0,0), 3)
        cv2.circle(imgc, (self.coorx_match,self.coory_match) , 8, (255,255,0), 3)
        return imgc
    
    def set_p1_marc(self,x1,y1):
        self.px_marc1=x1
        self.py_marc1=y1
        
    def set_p2_marc(self,x2,y2):
        self.px_marc2=x2
        self.py_marc2=y2
    
    def draw_marc(self):
        imgc=self.image.copy()
        cv2.rectangle(imgc, (self.px_marc1,self.py_marc1),(self.px_marc1-self.px_marc2,self.py_marc1+self.py_marc2) , (57,255,20), 4)
        return imgc
    
    def set_y(self,y):
        self.coory=y
        
    def set_xmatch(self,x):
        self.coorx_match=x

    def set_ymatch(self,y):
        self.coory_match=y        
        
    
    def set_wSquare(self,w):
        self.w=w
    
    def set_hSquare(self,h):
        self.h=h
        
    def set_xcoor(self,x):
        self.coorx=x
    
    
    def get_image(self):
        return self.image