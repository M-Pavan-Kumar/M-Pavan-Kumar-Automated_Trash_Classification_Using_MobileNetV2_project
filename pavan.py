from tkinter import messagebox, simpledialog, ttk, filedialog, END, RIGHT, LEFT, Y, X, RIDGE, RAISED, GROOVE, CENTER, Frame, Label, Button, Text, Scrollbar, Tk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import joblib
from scipy.stats import mode
import threading
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from keras.layers import Convolution2D, MaxPooling2D, Conv2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import model_from_json, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import normalize, LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform

# Ensure model directory exists
model_folder = "model"
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# Global variables
X = []
Y = []
X_train = None
X_test = None
y_train = None
y_test = None
Model1 = None
categories = []
filename = ""

# Create main window
main = Tk()
main.geometry("1300x1200")
main.title("Waste Classification System")

# Initialize MobileNetV2 base model
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Define color scheme
PRIMARY_COLOR = "#1E88E5"  # Sky blue
SECONDARY_COLOR = "#64B5F6"  # Lighter sky blue
BG_COLOR = "#E3F2FD"  # Very light sky blue
TEXT_COLOR = "#0D47A1"  # Dark blue
BUTTON_COLOR = "#2196F3"  # Medium sky blue
BUTTON_TEXT_COLOR = "white"
HIGHLIGHT_COLOR = "#42A5F5"  # Sky blue highlight

# Function to update progress in the text widget
def update_progress(message):
    text.insert(END, message + '\n')
    text.see(END)
    main.update_idletasks()

def uploadDataset():
    global filename, categories
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    if not filename:  # User cancelled
        return
    
    try:
        categories = [d for d in os.listdir(filename) if os.path.isdir(os.path.join(filename, d))]
        if not categories:
            messagebox.showerror("Error", "No class folders found in the selected directory")
            return
            
        update_progress('Dataset loaded')
        update_progress("Classes found in dataset: " + str(categories))
    except Exception as e:
        messagebox.showerror("Error", f"Error loading dataset: {str(e)}")

def process_in_background(func):
    """Decorator to run a function in a background thread"""
    def wrapper(*args, **kwargs):
        threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True).start()
    return wrapper

@process_in_background
def MobileNetV2_feature_extraction():
    global X, Y, base_model, categories, filename
    if not filename or not categories:
        messagebox.showerror("Error", "Please upload dataset first")
        return
        
    text.delete('1.0', END)
    update_progress("Starting feature extraction. This may take a while...")

    model_data_path = os.path.join(model_folder, "X.npy")
    model_label_path_GI = os.path.join(model_folder, "Y.npy")

    try:
        if os.path.exists(model_data_path) and os.path.exists(model_label_path_GI):
            update_progress("Loading preprocessed features from files...")
            X = np.load(model_data_path)
            Y = np.load(model_label_path_GI)
        else:
            update_progress("Processing images and extracting features...")
            X = []
            Y = []
            data_folder = filename
            total_files = sum([len([f for f in os.listdir(os.path.join(data_folder, d)) 
                                  if f.endswith('.jpg')]) for d in categories])
            processed = 0
            
            for class_label, class_name in enumerate(categories):
                class_folder = os.path.join(data_folder, class_name)
                update_progress(f"Processing class: {class_name}")
                
                for img_file in os.listdir(class_folder):
                    if img_file.endswith('.jpg'):
                        img_path = os.path.join(class_folder, img_file)
                        try:
                            img = image.load_img(img_path, target_size=(331, 331))
                            x = image.img_to_array(img)
                            x = np.expand_dims(x, axis=0)
                            x = preprocess_input(x)
                            features = base_model.predict(x)
                            features = np.squeeze(features)  # Flatten the features
                            X.append(features)
                            Y.append(class_label)
                            
                            processed += 1
                            if processed % 10 == 0:
                                update_progress(f"Processed {processed}/{total_files} images")
                                
                        except Exception as e:
                            update_progress(f"Error processing {img_path}: {str(e)}")
                            continue
            
            # Convert lists to NumPy arrays
            X = np.array(X)
            Y = np.array(Y)

            # Save processed images and labels
            update_progress("Saving processed features to disk...")
            np.save(model_data_path, X)
            np.save(model_label_path_GI, Y)
            
        update_progress("Image Preprocessing Completed")
        update_progress("MobileNetV2 Feature Extraction completed")
        update_progress(f"Feature Dimension: {X.shape}")
    except Exception as e:
        update_progress(f"Error during feature extraction: {str(e)}")
        messagebox.showerror("Error", f"Feature extraction failed: {str(e)}")

@process_in_background
def Train_test_spliting():
    global X, Y, X_train, X_test, y_train, y_test
    if len(X) == 0 or len(Y) == 0:
        messagebox.showerror("Error", "Please extract features first")
        return
        
    text.delete('1.0', END)
    update_progress("Splitting data into training and test sets...")
    
    try:
        X_downsampled = X
        Y_downsampled = Y
        indices_file = os.path.join(model_folder, "shuffled_indices.npy")  
        
        if os.path.exists(indices_file):
            indices = np.load(indices_file)
            X_downsampled = X_downsampled[indices]
            Y_downsampled = Y_downsampled[indices]  
        else:
            indices = np.arange(X_downsampled.shape[0])
            np.random.shuffle(indices)
            np.save(indices_file, indices)
            X_downsampled = X_downsampled[indices]
            Y_downsampled = Y_downsampled[indices]
            
        X_train, X_test, y_train, y_test = train_test_split(X_downsampled, Y_downsampled, test_size=0.2, random_state=42)

        update_progress(f"Input Data Train Size: {X_train.shape}")
        update_progress(f"Input Data Test Size: {X_test.shape}")
        update_progress(f"Output Train Size: {y_train.shape}")
        update_progress(f"Output Test Size: {y_test.shape}")
        update_progress("Data splitting completed. Ready for model training.")
    except Exception as e:
        update_progress(f"Error during data splitting: {str(e)}")
        messagebox.showerror("Error", f"Data splitting failed: {str(e)}")

def performance_evaluation(label, model_name, y_true, y_pred):
    try:
        accuracy = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, average='weighted')  
        rec = recall_score(y_true, y_pred, average='weighted')  
        f1s = f1_score(y_true, y_pred, average='weighted')  
        report = classification_report(y_true, y_pred, target_names=label)

        update_progress(f"{model_name} Accuracy: {accuracy:.4f}")
        update_progress(f"{model_name} Precision: {pre:.4f}")
        update_progress(f"{model_name} Recall: {rec:.4f}")
        update_progress(f"{model_name} F1-score: {f1s:.4f}")
        update_progress(f"{model_name} Classification report\n{report}")

        # Create confusion matrix in a separate thread to not block the GUI
        threading.Thread(target=plot_confusion_matrix, 
                        args=(y_true, y_pred, label, model_name), 
                        daemon=True).start()
        
        # Store metrics for comparison
        return {"name": model_name, "accuracy": accuracy, "precision": pre, 
                "recall": rec, "f1": f1s}
                
    except Exception as e:
        update_progress(f"Error evaluating model: {str(e)}")
        return None

def plot_confusion_matrix(y_true, y_pred, label, model_name):
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label, yticklabels=label)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f"{model_name} Confusion Matrix")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting confusion matrix: {str(e)}")

@process_in_background
def Model_LRC():
    global X_train, X_test, y_train, y_test, Model1
    if X_train is None or y_train is None:
        messagebox.showerror("Error", "Please split data into train/test sets first")
        return
        
    text.delete('1.0', END)
    update_progress("Training Logistic Regression Classifier...")
    
    try:
        # Reshape the data
        num_samples_train, height, width, channels = X_train.shape    
        X_train_reshaped = X_train.reshape(num_samples_train, height * width * channels)
        
        num_samples_test, height, width, channels = X_test.shape    
        X_test_reshaped = X_test.reshape(num_samples_test, height * width * channels)

        model_filename = os.path.join(model_folder, "LRC_model.pkl")
        if os.path.exists(model_filename):
            update_progress("Loading existing LRC model...")
            Model1 = joblib.load(model_filename)
        else:
            update_progress("Training new LRC model...")
            Model1 = LogisticRegression(C=0.01, penalty='l1', solver='liblinear', max_iter=1000)
            Model1.fit(X_train_reshaped, y_train)
            update_progress("Saving LRC model...")
            joblib.dump(Model1, model_filename)     
        
        update_progress("Evaluating model on test data...")
        Y_pred = Model1.predict(X_test_reshaped)
        performance_evaluation(categories, "Logistic Regression Classifier", y_test, Y_pred)
    except Exception as e:
        update_progress(f"Error in LRC model: {str(e)}")
        messagebox.showerror("Error", f"LRC model training failed: {str(e)}")

@process_in_background
def Model_NBC():
    global X_train, X_test, y_train, y_test, Model1
    if X_train is None or y_train is None:
        messagebox.showerror("Error", "Please split data into train/test sets first")
        return
        
    text.delete('1.0', END)
    update_progress("Training Naive Bayes Classifier...")
    
    try:
        # Reshape the data
        num_samples_train, height, width, channels = X_train.shape    
        X_train_reshaped = X_train.reshape(num_samples_train, height * width * channels)
        
        num_samples_test, height, width, channels = X_test.shape    
        X_test_reshaped = X_test.reshape(num_samples_test, height * width * channels)
        
        # Ensure non-negative data for MultinomialNB
        X_train_reshaped = np.abs(X_train_reshaped)
        X_test_reshaped = np.abs(X_test_reshaped)
        
        model_filename = os.path.join(model_folder, "NBC_model.pkl")
        if os.path.exists(model_filename):
            update_progress("Loading existing NBC model...")
            Model1 = joblib.load(model_filename)
        else:
            update_progress("Training new NBC model...")
            Model1 = MultinomialNB()
            Model1.fit(X_train_reshaped, y_train)
            update_progress("Saving NBC model...")
            joblib.dump(Model1, model_filename)     
        
        update_progress("Evaluating model on test data...")
        Y_pred = Model1.predict(X_test_reshaped)
        performance_evaluation(categories, "Naive Bayes Classifier", y_test, Y_pred)
    except Exception as e:
        update_progress(f"Error in NBC model: {str(e)}")
        messagebox.showerror("Error", f"NBC model training failed: {str(e)}")

@process_in_background
def Model_Final():
    global X_train, X_test, y_train, y_test, Model1
    if X_train is None or y_train is None:
        messagebox.showerror("Error", "Please split data into train/test sets first")
        return
        
    text.delete('1.0', END)
    update_progress("Training Random Forest Classifier (Proposed Method)...")
    
    try:
        # Reshape the data
        num_samples_train, height, width, channels = X_train.shape    
        X_train_reshaped = X_train.reshape(num_samples_train, height * width * channels)
        
        num_samples_test, height, width, channels = X_test.shape    
        X_test_reshaped = X_test.reshape(num_samples_test, height * width * channels)
        
        model_filename = os.path.join(model_folder, "RFC_model.pkl")
        if os.path.exists(model_filename):
            update_progress("Loading existing RFC model...")
            Model1 = joblib.load(model_filename)
        else:
            update_progress("Training new RFC model...")
            Model1 = RandomForestClassifier(n_estimators=100, random_state=42)
            Model1.fit(X_train_reshaped, y_train)
            update_progress("Saving RFC model...")
            joblib.dump(Model1, model_filename)     
        
        update_progress("Evaluating model on test data...")
        Y_pred = Model1.predict(X_test_reshaped)
        performance_evaluation(categories, "MobileNetV2 with Random Forest Classifier", y_test, Y_pred)
    except Exception as e:
        update_progress(f"Error in RFC model: {str(e)}")
        messagebox.showerror("Error", f"RFC model training failed: {str(e)}")

def predict():
    global base_model, categories, Model1
    if Model1 is None:
        messagebox.showerror("Error", "Please train a model first")
        return
        
    try:
        filename = filedialog.askopenfilename(initialdir="testImages", 
                                              filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not filename:  # User cancelled
            return
            
        update_progress(f"Predicting class for image: {filename}")
        
        # Load and preprocess image
        img = image.load_img(filename, target_size=(331, 331))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Extract features
        features = base_model.predict(x)
        
        # Reshape features for prediction
        features_reshaped = features.reshape(1, -1)
        
        # Predict class
        preds = Model1.predict(features_reshaped)
        
        if isinstance(preds, (list, np.ndarray)):
            pred_class = int(preds[0])
        else:
            pred_class = int(preds)
               
        # Display the result on the image
        img = cv2.imread(filename)
        img = cv2.resize(img, (800, 400))
        
        class_label = categories[pred_class]
        update_progress(f"Image classified as: {class_label}")

        text_to_display = f'Output Classified as: {class_label}'
        cv2.putText(img, text_to_display, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(f'Output Classified as: {class_label}', img)
        cv2.waitKey(0)
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {str(e)}")


def close():
    if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
        main.destroy()

# Function to create styled buttons with sky blue theme
def create_button(parent, text, command):
    btn = Button(parent, text=text, command=command, bg=BUTTON_COLOR, fg=BUTTON_TEXT_COLOR, 
                 activebackground=HIGHLIGHT_COLOR, activeforeground=BUTTON_TEXT_COLOR, 
                 relief=RAISED, bd=2, width=26, height=1)
    btn.config(font=('Helvetica', 11, 'bold'))
    return btn

# Create a frame for the title with sky blue gradient effect
title_frame = Frame(main, bg=PRIMARY_COLOR, height=100, width=1300)
title_frame.place(x=0, y=0)

# Title with modern font
title_font = ('Helvetica', 22, 'bold')
title = Label(title_frame, text='Smart Waste Classification System', bg=PRIMARY_COLOR, fg='white')
title.config(font=title_font)
title.place(relx=0.5, rely=0.5, anchor=CENTER)

# Subtitle
subtitle_font = ('Helvetica', 12, 'italic')
subtitle = Label(title_frame, text='Using MobileNetV2 for Efficient Waste Sorting', bg=PRIMARY_COLOR, fg='#E0E0E0')
subtitle.config(font=subtitle_font)
subtitle.place(relx=0.5, rely=0.8, anchor=CENTER)

# Create a frame for buttons with clean modern look
button_frame = Frame(main, bg=BG_COLOR, bd=2, relief=RIDGE, width=320, height=550)
button_frame.place(x=30, y=120)

# Frame title
frame_title = Label(button_frame, text="Control Panel", bg=PRIMARY_COLOR, fg="white", width=34)
frame_title.config(font=('Helvetica', 12, 'bold'))
frame_title.pack(pady=5)

# Create section labels
data_section = Label(button_frame, text="Data Processing", bg=SECONDARY_COLOR, fg="white", width=30)
data_section.config(font=('Helvetica', 10, 'bold'))
data_section.pack(pady=5)

# Create buttons with proper spacing
uploadButton = create_button(button_frame, "📁 Upload Dataset", uploadDataset)
uploadButton.pack(pady=5)

processButton = create_button(button_frame, "⚙️ Extract Features", MobileNetV2_feature_extraction)
processButton.pack(pady=5)

splitButton = create_button(button_frame, "✂️ Split Train/Test Data", Train_test_spliting)
splitButton.pack(pady=5)

model_section = Label(button_frame, text="Model Training & Evaluation", bg=SECONDARY_COLOR, fg="white", width=30)
model_section.config(font=('Helvetica', 10, 'bold'))
model_section.pack(pady=5)

lrcButton = create_button(button_frame, "🔄 Run Logistic Regression", Model_LRC)
lrcButton.pack(pady=5)

nbcButton = create_button(button_frame, "🔄 Run Naive Bayes", Model_NBC)
nbcButton.pack(pady=5)

rfcButton = create_button(button_frame, "⭐ Run Proposed Method", Model_Final)
rfcButton.pack(pady=5)


test_section = Label(button_frame, text="Testing & Exit", bg=SECONDARY_COLOR, fg="white", width=30)
test_section.config(font=('Helvetica', 10, 'bold'))
test_section.pack(pady=5)

predictButton = create_button(button_frame, "🔍 Test New Image", predict)
predictButton.pack(pady=5)

exitButton = create_button(button_frame, "❌ Exit Application", close)
exitButton.pack(pady=5)

# Create a frame for text output with clean modern look
text_frame = Frame(main, bg=BG_COLOR, bd=2, relief=RIDGE)
text_frame.place(x=380, y=120, width=880, height=550)

# Text output title
text_title = Label(text_frame, text="Results & Analysis", bg=PRIMARY_COLOR, fg="white")
text_title.config(font=('Helvetica', 12, 'bold'), width=72)
text_title.pack(pady=5)

# Text area for output with scrollbar
text = Text(text_frame, height=30, width=106, bg='white', fg=TEXT_COLOR)
text.config(font=('Consolas', 10))
scroll = Scrollbar(text_frame, command=text.yview)
text.configure(yscrollcommand=scroll.set)

# Use string values for the scrollbar
scroll.pack(side="right", fill="y")
text.pack(side="left", padx=10, pady=(40, 10))

# Create info section at the bottom
info_frame = Frame(main, bg=BG_COLOR, bd=1, relief=GROOVE)
info_frame.place(x=30, y=690, width=1230, height=100)

info_label = Label(info_frame, text="Waste Classification Categories", bg=PRIMARY_COLOR, fg='white')
info_label.config(font=('Helvetica', 12, 'bold'), width=137)
info_label.pack(fill="x")

# Category descriptions (update based on your actual categories)
category_info = """
Recyclable Categories: Paper, Plastic, Glass, Metal | Non-Recyclable Categories: Organic Waste, E-waste, Hazardous Waste | 
Use the control panel to process waste images and train models to accurately classify waste items.
"""
category_label = Label(info_frame, text=category_info, bg=BG_COLOR, fg=TEXT_COLOR, justify=LEFT)
category_label.config(font=('Helvetica', 10))
category_label.pack(pady=10, padx=20, fill="x")

# Initialize application with welcome message
text.insert(END, "Welcome to Smart Waste Classification System\n")
text.insert(END, "-" * 50 + "\n")
text.insert(END, "This application uses deep learning to classify waste images.\n\n")
text.insert(END, "How to use:\n")
text.insert(END, "1. Upload a dataset containing waste images organized in class folders\n")
text.insert(END, "2. Extract features using MobileNetV2\n")
text.insert(END, "3. Split data into training and test sets\n")
text.insert(END, "4. Train and evaluate different classification models\n")
text.insert(END, "5. Test the model on new waste images\n\n")
text.insert(END, "Start by clicking '📁 Upload Dataset'\n")
text.insert(END, "-" * 50 + "\n")

# Configure main window background
main.config(bg=BG_COLOR)
main.mainloop()