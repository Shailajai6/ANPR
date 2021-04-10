import streamlit as st
from PIL import Image
import base64
import cv2
import numpy as np
from keras.models import model_from_json
import imutils
import urllib.request
from sklearn.preprocessing import LabelEncoder
from IPython.display import Image as IPythonImage
from imageai.Detection.Custom import CustomObjectDetection
from tempfile import NamedTemporaryFile
import streamlit_theme as stt

stt.set_theme({'primary':'#262730','textColor':'#FFFFFF'})
#main_bg = "background.jpg"
main_bg='https://previews.123rf.com/images/eric4094/eric40940903/eric4094090300005/4570324-abstract-design-yellow-colour-background.jpg'
main_bg_ext = "jpg"
weburl = "https://capstoneprojectmksk.s3.amazonaws.com/detection_model-ex-015--loss-0006.450.h5"

filename = weburl.split('/')[-1]
urllib.request.urlretrieve(weburl, filename)

def detector_model():
    #model_path = 'model/detection_model-ex-005--loss-0003.767.h5'
    model_path=filename
    json_path = 'model/detection_config.json'
    
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path)
    detector.setJsonPath(json_path)
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image="uploaded.jpg", output_image_path='nplate3-detected.jpg')
    for obj in detections:
        print('hi')
        x,y,w,h = obj['box_points']
    lpimg = cv2.imread("uploaded.jpg")
    crop_img = lpimg[y:h, x:w]
    return crop_img

def load_model():

    # Load model architecture, weight and labels for character recognition
    json_file = open('model/ResNets_character_recognition_spyder_new.json')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model/License_character_recognition_spyder_new.h5")
    print("[INFO] Model loaded successfully...")
    return model
labels = LabelEncoder()
labels.classes_ = np.load('model/license_character_classes_Spyder.npy')
model=load_model()

def display_img(img_path):
    img = IPythonImage(filename=img_path)
    st.image(Image.open(img))


def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b: b[1][i], reverse=reverse))
    return cnts
def licenseplate(img):
    plate_image = cv2.convertScaleAbs(img) 
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
    return plate_image,gray,blur,binary,thre_mor,contours


def characters_crop(contours,plate_image,thre_mor):
    crop_characters=[]
    test_roi=plate_image.copy()
    for c in sort_contours(contours):
        digit_w,digit_h=30,60
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if 1<=ratio<=3.5:
            if h/plate_image.shape[0]>0.2: # Select contour which has the height larger than 50% of the plate
               # Draw bounding box arroung digit number
                cv2.rectangle(test_roi, (x+1, y+1), ((x+1) + (w+1), (y+1) + (h+1)), (0, 255,0), 2)
                 # Sperate number and gibe prediction
                curr_num = thre_mor[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)
    return crop_characters,test_roi
    

def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction
def streampredict(image,file_up):
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    if file_up is  not None and file_up !=0:
        with open("uploaded.jpg","wb") as f:
            f.write(file_up.getbuffer())
    else:
        pass
        
    crop_img=detector_model()
    col1, col2 = st.beta_columns(2)
    col1.header("image")
    col1.image('nplate3-detected.jpg', use_column_width=True)            
    col2.header("cropimage")            
    col2.image(crop_img, use_column_width=True)
    plate_image,gray,blur,binary,thre_mor,contours=licenseplate(crop_img)
    sorted_cont=sort_contours(contours)
#    test_roi=plate_image.copy()        
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
    crop_characters,test_roi=characters_crop(sorted_cont,plate_image,thre_mor)
    st.write("Detect {} letters...".format(len(crop_characters)))
    st.image(test_roi)
    final_string = ''
    for i,character in enumerate(crop_characters):
            title = np.array2string(predict_from_model(character,model,labels))
            final_string+=title.strip("'[]")
    st.title(final_string)

    
    
def main():
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://image.freepik.com/free-photo/black-sport-car-dark-background-3d-render_68747-40.jpg");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

   
    
    st.sidebar.info("This is an Licence plate detection  web deployment Model.")
    
    st.set_option('deprecation.showfileUploaderEncoding', False)
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
                streampredict(image,file_up)
                
    else:
        st.success("Please Insert Web URL")
        url = st.text_input("Insert URL below")
        if url:
            urllib.request.urlretrieve(url,'uploaded.jpg')
            image=Image.open('uploaded.jpg')
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Just a second...")
            streampredict(image,file_up=0)
main()





