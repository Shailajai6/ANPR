import streamlit as st
from PIL import Image
import requests
import base64
from io import BytesIO
import os
#import tensorflow as tf
#import keras
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
from keras.models import model_from_json
import imutils
from sklearn.preprocessing import LabelEncoder
from IPython.display import Image as IPythonImage
from imageai.Detection.Custom import CustomObjectDetection
from tempfile import NamedTemporaryFile

#print(tf.keras.__version__)
#print(keras.__version__)
#print(tf.__version__)
main_bg = "background.jpg"
main_bg_ext = "jpg"
model_path = 'model/detection_model-ex-005--loss-0003.767.h5'
json_path = 'model/detection_config.json'

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path)
detector.setJsonPath(json_path)
detector.loadModel()

def display_img(img_path):
    img = IPythonImage(filename=img_path)
    st.image(Image.open(img))


def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction
# Load model architecture, weight and labels
json_file = open('model/ResNets_character_recognition_spyder_new.json')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model/License_character_recognition_spyder_new.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('model/license_character_classes_Spyder.npy')
#img_y1=img_to_array(img)
#img_y1=np.expand_dims(img_y1, axis=0)
currdir = os.getcwd()
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

st.sidebar.info("This is an Licence plate detection  web deployment Model.")

st.set_option('deprecation.showfileUploaderEncoding', False)

#st.title("Image Captioning")
st.markdown("<h1 style='text-align: center; color: green;'>Licence Plate Detection Model</h1>", unsafe_allow_html=True)
st.write("")

st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
status = st.radio("Hello, Do you want to Upload an Image or Insert an Image URL?",("Upload Image","Insert URL"))
if status == 'Upload Image':
    st.success("Please Upload an Image")
    file_up = st.file_uploader("Upload an image", type=['jpg','png','jpeg'])
    temp_file = NamedTemporaryFile(delete=False)
    if file_up is not None:
            temp_file.write(file_up.getvalue())
            image =Image.open(file_up)
            file_details = {"FileName":file_up.name,"FileType":file_up.type}
            st.write(file_details)
            #lpimg = cv2.imread(file_up)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Just a second...")
            st.write(file_up)
            with open("uploaded.jpg","wb") as f:
                f.write(file_up.getbuffer())
            detections = detector.detectObjectsFromImage(input_image="uploaded.jpg", output_image_path='nplate3-detected.jpg')

            for obj in detections:
                st.write(obj['name'])
                st.write(obj['percentage_probability'])
                st.write(obj['box_points'])
                x,y,w,h = obj['box_points']
            st.image("nplate3-detected.jpg")
            #display_img('nplate3-detected.jpg')
            lpimg = cv2.imread("uploaded.jpg")
            crop_img = lpimg[y:h, x:w]
            st.image(crop_img)

            
            if (len(crop_img)): #check if there is at least one license image
                # Scales, calculates absolute values, and converts the result to 8-bit.
                plate_image = cv2.convertScaleAbs(crop_img)
    
                # convert to grayscale and blur the image
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray,(7,7),0)
                
                # Applied inversed thresh_binary 
                binary = cv2.threshold(blur, 180, 255,
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                
                kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
            keypoints = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(keypoints)
            #contours = sorted(contours, key= cv2.contourArea, reverse=True)
            contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0], reverse = False)
            test_roi=plate_image.copy()
            
            st.image(crop_img, caption = "Licence Plate Detected", use_column_width =False)
            col1, col2 = st.beta_columns(2)

            col1.header("blur")
            col1.image(blur, use_column_width=True)            
            col2.header("Grayscale")            
            col2.image(gray, use_column_width=True)
            col1, col2 = st.beta_columns(2)
            col1.header("binary")
            col1.image(binary, use_column_width=True)            
            col2.header("dilation")            
            col2.image(thre_mor, use_column_width=True)
            crop_characters=[]
            for c in sort_contours(contours):
                digit_w,digit_h=30,60
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h/w
                if 1<=ratio<=3.5: # Only select contour with defined ratio
                    if h/plate_image.shape[0]>0.2: # Select contour which has the height larger than 50% of the plate
                        # Draw bounding box arroung digit number
                        cv2.rectangle(test_roi, (x+1, y+1), ((x+1) + (w+1), (y+1) + (h+1)), (0, 255,0), 2)
            
                        # Sperate number and gibe prediction
                        curr_num = thre_mor[y:y+h,x:x+w]
                        curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                        _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        crop_characters.append(curr_num)
            st.write("Detect {} letters...".format(len(crop_characters)))
            st.image(test_roi)
            final_string = ''
            for i,character in enumerate(crop_characters):
                #fig.add_subplot(grid[i])
                title = np.array2string(predict_from_model(character,model,labels))
                #plt.title('{}'.format(title.strip("'[]"),fontsize=20))
                final_string+=title.strip("'[]")
            st.write(final_string)

else:
    st.success("Please Insert Web URL")
    url = st.text_input("Insert URL below")
    if url:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Just a second...")
        #vehicle, LpImg,cor = get_plate(image)
        st.image(crop_img, caption = "Licence Plate Detected", use_column_width =True)





