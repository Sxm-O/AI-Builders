import torch
import torch.nn as nn
from torchvision import models  # Add this import statement
from torchvision import transforms
from PIL import Image
import streamlit as st


#set title
st.markdown("<h1 style='text-align: center;'>🐍 Snake Classification 🐍 </h1>", unsafe_allow_html=True)

#set header
st.markdown("<div style='text-align: left;'><br>Please upload a snake image</div>", unsafe_allow_html=True)



#load model
# Define class names  
class_names = ['งูสามเหลี่ยม (Bungarus fasciatus)', 'งูก้นขบ (Cylindrophis ruffus)', 'งูจงอาง (Ophiophagus hannah)', 'งูทางมะพร้าว (Coelognathus radiatus)', 'งูกะปะ (Calloselasma rhodostoma)', 'งูเห่า (Cobra)', 'งูเขียวหางไหม้ (Green pit viper)', 'งูปี่แก้วลายแต้ม (Oligodon barroni)', 'งูเขียวปากจิ้งจก (Oriental whip snake)', 'งูลายสาบคอแดง (Red necked keelback)', 'งูเหลือม (Reticulated python)', "งูแมวเซา (Siamese Russell's Viper)", 'งูแสงอาทิตย์ (Sunbeam snake)']

# Define your model
model = models.resnet34(pretrained=True)
for param in model.parameters():
    param.requires_grad = True
model.fc = nn.Linear(in_features=512, out_features=13)

# Load the trained model's parameters
model.load_state_dict(torch.load('C:/Users/USER - 14350/Desktop/AI builders/my_model3-2.pth', map_location=torch.device('cpu')))
 
# Preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model's input size
    transforms.ToTensor(),           # Convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])


# Define venomous snake species
venomous = ['งูสามเหลี่ยม (Bungarus fasciatus)', 'งูจงอาง (Ophiophagus hannah)', 'งูกะปะ (Calloselasma rhodostoma)', 'งูเห่า (Cobra)', "งูแมวเซา (Siamese Russell's Viper)",'งูลายสาบคอแดง (Red necked keelback)','งูเขียวหางไหม้ (Green pit viper)']

def classify_image(image):
    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add unsqueeze to add batch dimension
    
    # Use your model to predict the class
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    # Calculate confidence
    confidence = torch.softmax(outputs, dim=1)[0] * 100
    
    # Get the predicted class label
    predicted_class_index = predicted.item()
    predicted_class_name = class_names[predicted_class_index] if predicted is not None else "Unknown"
    
    # Check if the predicted snake species is venomous
    if predicted_class_name in venomous:
        snake_venomous = "Venomous(มีพิษ)"
    else:
        snake_venomous = "Non-Venomous(ไม่มีพิษ)"
    
    # Return the predicted class label, confidence, and venomous status
    return predicted_class_name, confidence[predicted_class_index].item(), snake_venomous

#upload file
uploaded_image = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Perform classification
    predicted_class, confidence, venomous_status = classify_image(image)
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
    st.write(f"Venomous Status: {venomous_status}")







