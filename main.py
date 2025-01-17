import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import img_to_array, load_img
import tf_keras # keras 2
# import torch
# import torchvision
# from torchvision import transforms

# streamlit run app.py
# http://localhost:8501/

IMG_SIZE = 224



st.title('Pathological detection based on Medical images')
poster = Image.open('model architecture.jpg')
st.image(poster, caption='Model architecture', use_container_width=True)


# Dropdown menu to select dataset list
dataset = st.selectbox('Select kind of image using to detect', ['', 'Chest X-ray', 'Knee X-ray', 'Chest CT', 'Brain CT', 'Brain MRI', 'Breast MRI', 'Breast US.1', 'Breast US.2'])

model_group_list = {
    'Chest X-ray': [
        'models/01_07/01_07_01/best_model_val_acc.h5',
        'models/01_07/01_07_02/best_model_val_acc.h5',
        'models/01_07/01_07_03/best_model_val_acc.h5',
        'models/01_07/01_07_04/best_model_val_acc.h5',
        'models/01_07/01_07_05/best_model_val_acc.h5',
    ],
    'Knee X-ray': [
        'models/02_08/02_08_01/best_model_val_acc.h5',
        'models/02_08/02_08_02/best_model_val_acc.h5',
        'models/02_08/02_08_03/best_model_val_acc.h5',
        'models/02_08/02_08_04/best_model_val_acc.h5',
        'models/02_08/02_08_05/best_model_val_acc.h5',
    ],
    'Chest CT': [
        'models/03_03/03_03_01/best_model_val_acc.h5',
        'models/03_03/03_03_02/best_model_val_acc.h5',
        'models/03_03/03_03_03/best_model_val_acc.h5',
        'models/03_03/03_03_04/best_model_val_acc.h5',
        'models/03_03/03_03_05/best_model_val_acc.h5',
    ],
    'Brain CT': [
        'models/09_07/09_07_01/best_model_val_acc.h5',
        'models/09_07/09_07_02/best_model_val_acc.h5',
        'models/09_07/09_07_03/best_model_val_acc.h5',
        'models/09_07/09_07_04/best_model_val_acc.h5',
        'models/09_07/09_07_05/best_model_val_acc.h5',
    ],
    'Brain MRI': [
        'models/04_07/04_07_01/best_model_val_acc.h5',
        'models/04_07/04_07_02/best_model_val_acc.h5',
        'models/04_07/04_07_03/best_model_val_acc.h5',
        'models/04_07/04_07_04/best_model_val_acc.h5',
        'models/04_07/04_07_05/best_model_val_acc.h5',
    ],
    'Breast MRI': [
        'models/07_08/07_08_01/best_model_val_acc.h5',
        'models/07_08/07_08_02/best_model_val_acc.h5',
        'models/07_08/07_08_03/best_model_val_acc.h5',
        'models/07_08/07_08_04/best_model_val_acc.h5',
        'models/07_08/07_08_05/best_model_val_acc.h5',
    ],
    'Breast US.1': [
        'models/05_08/05_08_01/best_model_val_acc.h5',
        'models/05_08/05_08_02/best_model_val_acc.h5',
        'models/05_08/05_08_03/best_model_val_acc.h5',
        'models/05_08/05_08_04/best_model_val_acc.h5',
        'models/05_08/05_08_05/best_model_val_acc.h5',
    ],
    'Breast US.2': [
        'models/08_08/08_08_01/best_model_val_acc.h5',
        'models/08_08/08_08_02/best_model_val_acc.h5',
        'models/08_08/08_08_03/best_model_val_acc.h5',
        'models/08_08/08_08_04/best_model_val_acc.h5',
        'models/08_08/08_08_05/best_model_val_acc.h5',
    ]
}

testset_path_list = {
    'Chest X-ray': 'medical_testsets/origin/COVID19_Pneumonia_Normal_Chest_Xray_PA',\
    'Knee X-ray': 'medical_testsets/origin/Osteoporosis-Knee-Xray-Dataset',\
    'Chest CT': 'medical_testsets/origin/Chest-CT-Scan-images-Dataset',\
    'Brain CT': 'medical_testsets/origin/Brain-Stroke-CT-Image-Dataset',\
    'Brain MRI': 'medical_testsets/origin/Brain-Tumor-MRI-Dataset',\
    'Breast MRI': 'medical_testsets/origin/Breast-Cancer-Patients-MRI',\
    'Breast US.1': 'medical_testsets/origin/Ultrasound-Breast-Images-for-Breast-Cancer',\
    'Breast US.2': 'medical_testsets/origin/MT-Small-Dataset',\
}


processed_testset_path_list = {
    'Chest X-ray': 'medical_testsets/preprocessed/COVID19_Pneumonia_Normal_Chest_Xray_PA_Median',\
    'Knee X-ray': 'medical_testsets/preprocessed/Osteoporosis-Knee-Xray_unsharpMasking',\
    'Chest CT': 'medical_testsets/preprocessed/Chest-CT-Scan-images_DWT',\
    'Brain CT': 'medical_testsets/preprocessed/Brain-Stroke-CT-Image-Dataset_Median-Mean-Hybrid-Filter',\
    'Brain MRI': 'medical_testsets/preprocessed/Brain-Tumor-MRI_Median-Mean-Hybrid_Filter',\
    'Breast MRI': 'medical_testsets/preprocessed/Breast-Cancer-Patients-MRI_unsharpMasking_bilateralfilter',\
    'Breast US.1': 'medical_testsets/preprocessed/Ultrasound-Breast-Images-for-Breast-Cancer_unsharpMasking_bilateralfilter',\
    'Breast US.2': 'medical_testsets/preprocessed/MT-Small-Dataset_Unsharp-Masking_Bilateral_Filter',\
}

class_names_list = {
    'Chest X-ray': ['covid', 'normal', 'pneumonia'],\
    'Knee X-ray': ['normal', 'osteoporosis'],\
    'Chest CT': ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma'],\
    'Brain CT': ['Normal', 'Stroke'],\
    'Brain MRI': ['glioma', 'meningioma', 'notumor', 'pituitary'],\
    'Breast MRI': ['Healthy', 'Sick'],\
    'Breast US.1': ['benign', 'malignant'],\
    'Breast US.2': ['Benign', 'Malignant']
}


COLOR = 'rgb'
INTERPOLATION = 'bilinear'

@st.cache_resource
def load_model(model_group):
    efficientnet = tf_keras.models.load_model(model_group[0])
    resnet = tf_keras.models.load_model(model_group[1])
    densenet = tf_keras.models.load_model(model_group[2])
    vgg = tf_keras.models.load_model(model_group[3])
    mobilenet = tf_keras.models.load_model(model_group[4])
    return efficientnet, resnet, densenet, vgg, mobilenet

# Load multiple ảnh theo cách sử dụng image_dataset_from_directory
def load_images_from_directory(directory, target_size=(224, 224), interpolation='bilinear'):
    """
    prediction = model.predict(img)
    print(prediction)
    total_sum = tf.reduce_sum(img)
    print(f"The sum of all values in the image is: {total_sum.numpy()}")
    ```output
    [[0.5879362]]
    The sum of all values in img is: 8400836.0
    ```
    """
    # Tạo dataset từ thư mục
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=32,
        image_size=target_size,
        shuffle=False
    )

    # Lấy ảnh và nhãn từ dataset
    for images, labels in dataset:
        break

    # Chuyển ảnh sang numpy array
    images = images.numpy()

    # Chuyển kiểu dữ liệu thành float32
    images = tf.convert_to_tensor(images, dtype=tf.float32)

    return images

# Load ảnh theo cách sử dụng tf.keras.utils.load_img
def load_image(filepath, target_size=(224, 224), interpolation='bilinear'):
    """
    prediction = model.predict(img)
    print(prediction)
    total_sum = tf.reduce_sum(img)
    print(f"The sum of all values in the image is: {total_sum.numpy()}")
    ```output
    [[0.27553144]]
    The sum of all values in idx_441 is: 8429490.0
    ```
    """
    # Đọc ảnh và giải mã
    img = tf.keras.utils.load_img(filepath, target_size=target_size, color_mode='rgb', interpolation=interpolation)

    # Chuyển ảnh sang numpy array
    img = tf.keras.utils.img_to_array(img)


    # Chuyển kiểu dữ liệu thành float32
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    

    # Mô phỏng batch size 1
    img = np.expand_dims(img, axis=0)
    return

# Load ảnh theo cách tương tự image_dataset_from_directory
def load_single_image(img, target_size=(224, 224), interpolation='bilinear', img_type='bytes'):
    """
    Args: imge_type: 'bytes' or 'file'

    prediction = model.predict(img)
    print(prediction)
    total_sum = tf.reduce_sum(img)
    print(f"The sum of all values in the image is: {total_sum.numpy()}")
    ```output
    [[0.5879362]]
    The sum of all values in the image is: 8400836.0
    ```
    """
    # Đọc ảnh và giải mã
    if img_type != 'bytes':
        img = tf.io.read_file(img) # Read image file

    img = tf.image.decode_image(img, channels=3, expand_animations=False)

    # Resize ảnh
    img = tf.image.resize(img, target_size, method=interpolation)

    # Đảm bảo dữ liệu ảnh có kiểu float32 (giống cách 2)
    img = tf.cast(img, tf.float32)

    # Mô phỏng batch size 1
    img = tf.expand_dims(img, axis=0)
    return img


if dataset != '':
    st.write(f'You selected {dataset}')
    efficientnet, resnet, densenet, vgg, mobilenet = load_model(model_group_list[dataset])
    class_names = class_names_list[dataset]




testset_path_list = processed_testset_path_list

st.header('Upload a medical image')
uploaded_file = st.file_uploader("Choose an image file", \
                                 type = ['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Test image')
    
    # Option to define if there is existing label
    label_exists = st.checkbox('Does this image have an existing label?')

    if label_exists:
        existing_label = st.selectbox('Select existing label', class_names)

    if st.button('Predict'):

        ## Resize the image
        # # Option 1: Using tf's load_img()
        # img_224 = load_img(
        #     uploaded_file,
        #     target_size = (224, 224),
        #     color_mode = COLOR,
        #     interpolation = INTERPOLATION
        # )
        # img_380 = load_img(
        #     uploaded_file,
        #     target_size = (380, 380),
        #     color_mode = COLOR,
        #     interpolation = INTERPOLATION
        # )
        # # Option 2: Using PIL's Image
        # # img = img.convert(mode='RGB')
        # # img_224 = img.resize((224, 224), Image.BILINEAR)
        # # img_380 = img.resize((380, 380), Image.BILINEAR)
        # # st.image(img_380, caption='Resized image')

        # ## Preprocess the image
        # img_224 = img_to_array(img_224) # not needed if using tf's load_img()
        # img_224 = img_224 / 255.0
        # img_224 = np.expand_dims(img_224, axis=0)
        # img_380 = img_to_array(img_380)
        # img_380 = img_380 / 255.0
        # img_380 = np.expand_dims(img_380, axis=0)

        # Read the uploaded file as bytes
        image = uploaded_file.getvalue()

        img_224 = load_single_image(image, target_size=(224, 224), interpolation='bilinear')
        img_380 = load_single_image(image, target_size=(380, 380), interpolation='bilinear')

        # Predict using the 5 models
        predictions = {
            'efficientnet': efficientnet.predict(img_380),
            'resnet': resnet.predict(img_224),
            'densenet': densenet.predict(img_224),
            'vgg': vgg.predict(img_224),
            'mobilenet': mobilenet.predict(img_224)
        }

        num_classes = len(class_names)
        


        st.write('Prediction')

        for model, prediction in predictions.items():
            # st.write(f'{model}: {prediction}') # print out probabilities
            if num_classes == 2:
                prediction = 1 if prediction > 0.5 else 0
            else:
                prediction = np.argmax(prediction, axis=-1)[0]

            st.write(f'{model}: {class_names[prediction]}. \t\tLabel: {existing_label if label_exists else "Unknown"}')
    
        









