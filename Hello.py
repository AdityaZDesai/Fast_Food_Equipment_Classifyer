from flask import Flask, redirect, url_for, request, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import os.path
from PIL import Image

app = Flask(__name__)

@app.route('/hello/<name>')
def hello_world(name):
    return render_template('welcome.html', name = name)

@app.route('/image/')
def input_image():
    return redirect(url_for('hello_world', name = "bitch"))


@app.route('/')
def register():
    return render_template('login.html')


@app.route('/login/', methods = ['POST'])
def login():
    user = request.files['img']
    image_path = "./testimages/" + user.filename
    user.save(image_path)
    print(user)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ConvNet().to(device)
    x = 'model.pth'
# torch.save(model.state_dict(), file)

    model.load_state_dict(torch.load(x))
    model.eval()

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Same as for your validation data, e.g. Resize, ToTensor, Normalize, ...
    
    

    img = Image.open(user)


    #img = Image.open(image)  # Load image as PIL.Image
    img = img.resize((28,28))

    x = transform(img)  # Preprocess image
    x = x.unsqueeze(0)  # Add batch dimension

    output = model(x)  # Forward pass
    pred = torch.argmax(output, 1)  # Get predicted class if multi-class classification
    #print('Image predicted as **[1]', pred)
    if str(pred.numpy()) == "[0]":
        return render_template('result.html', result= "FLOOR")
    else:
        return render_template('result.html', result= "MICROWAVE")



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 256)            # -> n, 400
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x


@app.route('/compute/<image>')
def compute(image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ConvNet().to(device)
    x = 'model.pth'
# torch.save(model.state_dict(), file)

    model.load_state_dict(torch.load(x))
    model.eval()

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Same as for your validation data, e.g. Resize, ToTensor, Normalize, ...
    
    

    img = Image.open(image)


    #img = Image.open(image)  # Load image as PIL.Image
    img = img.resize((28,28))

    x = transform(img)  # Preprocess image
    x = x.unsqueeze(0)  # Add batch dimension

    output = model(x)  # Forward pass
    pred = torch.argmax(output, 1)  # Get predicted class if multi-class classification
    #print('Image predicted as **[1]', pred)
    if str(pred.numpy()) == "[0]":
        return "FLOOR"
    else:
        return "MICROWAVE"




if __name__ == '__main__':
    app.debug = True
    app.run()
    app.run(debug = True)

