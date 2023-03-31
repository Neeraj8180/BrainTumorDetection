{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "3c50e79e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\neera\\anaconda3\\lib\\site-packages\\gradio\\inputs.py:257: UserWarning: Usage of gradio.inputs is deprecated, and will not be supported in the future, please import your component from gradio.components\n",
      "  warnings.warn(\n",
      "C:\\Users\\neera\\anaconda3\\lib\\site-packages\\gradio\\deprecation.py:40: UserWarning: `optional` parameter is deprecated, and it has no effect\n",
      "  warnings.warn(value)\n",
      "C:\\Users\\neera\\anaconda3\\lib\\site-packages\\gradio\\outputs.py:197: UserWarning: Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components\n",
      "  warnings.warn(\n",
      "C:\\Users\\neera\\anaconda3\\lib\\site-packages\\gradio\\deprecation.py:40: UserWarning: The 'type' parameter has been deprecated. Use the Number component instead.\n",
      "  warnings.warn(value)\n",
      "C:\\Users\\neera\\anaconda3\\lib\\site-packages\\gradio\\deprecation.py:40: UserWarning: `capture_session` parameter is deprecated, and it has no effect\n",
      "  warnings.warn(value)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running on local URL:  http://127.0.0.1:7882\n",
      "\n",
      "Could not create share link. Please check your internet connection or our status page: https://status.gradio.app\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div><iframe src=\"http://127.0.0.1:7882/\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": []
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 0s 199ms/step\n",
      "1/1 [==============================] - 0s 103ms/step\n",
      "1/1 [==============================] - 0s 105ms/step\n",
      "1/1 [==============================] - 0s 100ms/step\n",
      "1/1 [==============================] - 0s 105ms/step\n",
      "1/1 [==============================] - 0s 114ms/step\n",
      "1/1 [==============================] - 0s 108ms/step\n",
      "1/1 [==============================] - 0s 106ms/step\n",
      "1/1 [==============================] - 0s 93ms/step\n",
      "1/1 [==============================] - 0s 98ms/step\n"
     ]
    }
   ],
   "source": [
    "import gradio as gr\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import cv2\n",
    "\n",
    "# Load the pre-trained model\n",
    "model = tf.keras.models.load_model('brain_tumor_detection.h5')\n",
    "\n",
    "\n",
    "# Define a function to preprocess the image\n",
    "def preprocess_image(img):\n",
    "    # Resize the image to (224, 224) as expected by the model\n",
    "    img = cv2.resize(img, (224, 224))\n",
    "    # Convert the image to RGB\n",
    "    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "    # Convert the image to a numpy array\n",
    "    img = np.array(img)\n",
    "    # Normalize the image pixels to be between 0 and 1\n",
    "    img = img / 255.0\n",
    "    # Add a batch dimension to the image\n",
    "    img = np.expand_dims(img, axis=0)\n",
    "    return img\n",
    "\n",
    "# Define a function to make predictions on the input image\n",
    "def predict_image(img):\n",
    "    # Preprocess the image\n",
    "    img = preprocess_image(img)\n",
    "    # Make a prediction using the model\n",
    "    pred = model.predict(img)[0][0]\n",
    "    # Return the prediction as a boolean (True for tumor, False for no tumor)\n",
    "    if pred>0.5:\n",
    "        return \"No Tumor Detected, Stay safe Stay happy!!! \"\n",
    "    else:\n",
    "        return \"Tumor Detected, Take Precaution Immediately\"\n",
    "    \n",
    "\n",
    "\n",
    "\n",
    "# Define the Gradio interface\n",
    "inputs = gr.inputs.Image(shape=(224, 224))\n",
    "outputs = gr.outputs.Label(num_top_classes=2)\n",
    "\n",
    "interface = gr.Interface(fn=predict_image, inputs=inputs, outputs=outputs, capture_session=True, title='Brain Tumor Detection', description='It is the Brain Tumor Detection AI model which can predict that in a MRI scaned Img, any Tumor is present or Not. Upload an image of a brain MRI scan to detect if a tumor is present.')\n",
    "# Run the interface\n",
    "interface.launch(share= True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b115fe77",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
