# SiameseNetwork for face similarity discrimination
````
SiameseNetwork
    |--dataset
        |--faces
            |--training
            |--testing
    |--checkpoint
    |--config.py
    |--inference.py
    |--main.py
    |--model.py
    |--utils.py
    |--README.md
````
# dataset
The ORL face data set contains 400 images of 40 different people. 
It was created by the Olivetti Research Laboratory in Cambridge, England between April 1992 and April 1994. 
This data set contains 40 catalogs, each catalog has 10 images, and each catalog represents a different person. 
All images are stored in PGM format, grayscale images, the size of the image width is 92, the height is 112.
 For the images in each category, these images are collected at different times, different lighting, different facial 
 expressions (eyes open/closed, smiling/not smiling) and facial details (with glasses/without glasses) environment of. 
 All the images were taken against a dark, uniform background, and the front face (some with a slight side shift) was taken.

The training folder in the downloaded data set contains images of 37 people, 
and the images of the remaining 3 people are placed in the testing folder for subsequent testing.

ORL face dataset download address
````
https://drive.google.com/file/d/1jC1XCCcgm8Dp7-UWjsb9wv5D6tHYxqd6/view?usp=sharing
````

