import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Load the digits dataset
digits=load_digits()
x=digits.data
y=digits.target
# Split the dataset into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)

acc=accuracy_score(y_test,model.predict(x_test))
print(f"Model trained with accuracy:{acc*100:.2f}%")
image_path=input("\nEnter yourhandwritten image filename(digit.png):")
img=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error could not find the image file")
    exit()
img_resized=cv2.resize(img,(8,8),interpolation=cv2.INTER_AREA)
if np.mean(img_resized)>127:
    img_resized=255-img_resized
img_thresh=cv2.threshold(img_resized,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
img_scaled=(img_thresh/16).astype(np.float32)

input_data=img_scaled.flatten().reshape(1,-1)
Predicted_digit=model.predict(input_data)[0]
print(f"\n The Predicted digit is:{Predicted_digit}")
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(img,cmap='gray')
plt.title(" original image")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(img_thresh,cmap='gray')
plt.title("predicted image")
plt.axis('off')
plt.tight_layout()
plt.show()
