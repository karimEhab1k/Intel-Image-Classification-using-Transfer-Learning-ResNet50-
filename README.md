## Intel Image Classification using Transfer Learning (ResNet50)

A deep learning project that classifies natural scene images from the 
[Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) 
into 6 categories: **buildings, forest, glacier, mountain, sea, and street**.

###Approach
- Built on top of **ResNet50** pretrained on ImageNet (Transfer Learning)
- Fine-tuned the last 10 layers while freezing the rest
- Used **GlobalAveragePooling2D** + Dense softmax head for classification
- Applied **ReduceLROnPlateau** and **ModelCheckpoint** callbacks for stable training

###Dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) 
and place it under a `data/` folder in the project root.

###How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Place the dataset in the `data/` folder
4. Open and run `intel_image_classification_resnet50.ipynb`

###Tech Stack
- Python, TensorFlow / Keras
- ResNet50 (ImageNet weights)
- Matplotlib, NumPy
