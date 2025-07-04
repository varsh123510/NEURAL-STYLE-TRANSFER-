*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: VARSHA .R

*INTERN ID*: CT06DG2274

*DOMAIN*: ARTIFICIAL INTELLIGENCE

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH

*DESCRIPTION OF TASK*



🖼️ Task 3: Neural Style Transfer 

Task 3 of the CodTech AI Internship introduces you to Neural Style Transfer (NST) — a deep learning-based technique that combines the content of one image with the style of another to produce a stunning new artistic output. This task sits at the intersection of art and AI, showcasing how neural networks can generate human-like creativity.
In simple terms, Neural Style Transfer allows you to take an image (like your selfie or landscape photo) and re-render it in the style of famous paintings (like Van Gogh’s Starry Night or Picasso’s cubism). The model uses Convolutional Neural Networks (CNNs), particularly the pre-trained VGG19 network, to extract both content and style features from the images.

🌟 Objectives of the Task:

Understand the concepts of content representation and style representation using CNN layers.

Build a Python program using TensorFlow/Keras or PyTorch to perform style transfer.

Input: One content image + One style image.

Output: A generated image that blends the content with the chosen artistic style.

🔧 Implementation Details:

Interns begin by selecting two images:

A content image (e.g., a portrait or scenery)

A style image (e.g., a painting or design pattern)

The NST model performs the following:

Extracts content features from intermediate CNN layers (e.g., block5_conv2 in VGG19).

Extracts style features from multiple CNN layers (e.g., block1_conv1 to block5_conv1).

Calculates content loss and style loss.

Optimizes a new image (initially a copy of the content image or noise) to minimize both losses.

Iteratively updates the image using gradient descent to match the content and style goals.

Libraries used typically include:

TensorFlow or PyTorch

OpenCV / PIL for image handling

Matplotlib for visualization

The key formula behind style transfer:
Total Loss = α * Content Loss + β * Style Loss 
Where α and β control how much style or content is prioritized in the final image.

📂 Expected Submission:

Complete Python notebook or .py script with all necessary code.

Input images and the resulting stylized output image.

A short write-up explaining how style transfer works.

Optional enhancements: GUI with Streamlit or ability to choose images dynamically.

💡 Learning Outcomes:

By completing Task 3, interns gain:

An understanding of pre-trained CNN models and feature extraction.

Experience with loss functions, backpropagation, and image optimization.

Practical exposure to generative AI and creative applications of deep learning.

The ability to experiment with artistic styles and build real-world computer vision projects.

Neural Style Transfer is not just a technical project — it’s an opportunity to turn AI into art. Whether you're transforming a selfie into a Van Gogh painting or adding classical textures to modern photography, NST demonstrates the beautiful blend of math, imagination, and machine intelligence.

*OUTPUT*

![Image](https://github.com/user-attachments/assets/a90060e7-074d-4608-9b31-311bd96eb9f6)
