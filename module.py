import cv2
import numpy as np
import matplotlib.pyplot as plt




class Point(object):
    def __init__(self, model_path:str):
        self.net = cv2.dnn.readNet(model_path)
        self.out = 'StatefulPartitionedCall/StatefulPartitionedCall/sequential/dense_1/MatMul'
        
    def point_face(self, output):
        x, y = output[0::2], output[1::2]
        return x, y
    
    def visualize(self, image, x, y):
        plt.imshow(image,cmap='gray')
        plt.plot(x, y, '.', color='green',linewidth=2, markersize=9)
        plt.show()     
    
    def __call__(self, image_path:str):
        image = cv2.imread(image_path, 0)
        image = cv2.resize(image, (96, 96))
        img_blob = cv2.dnn.blobFromImage(image, 0.003922, (96, 96), swapRB=False, crop=False)
        self.net.setInput(img_blob)
        output  = self.net.forward(self.out)[0] * 96
        x, y = self.point_face(output)
        self.visualize(image, x, y)

if __name__ == '__main__':
    cls = Point(model_path = 'face-point.pb')
    cls("img/2.jpg")
