# FlapPyBird-Blink

A Flappy Bird Clone made using [python-pygame][pygame] (forked from sourabhv/FlapPyBird)

I tried to implement FlapPyBird with blink as a control instead of space or keyup button using dlib library, inspired by Instagram Filter. This project is done to fulfill the final project of Intelligent Control System class in Tohoku University (Vision-Based Control)
The face is first detected using Histogram of Oriented Gradients (HOG) feature combined with a linear classifier.
Then, the face landmark is predicted using Kazemi's predictor. The blink is then detected by using the Eye Aspect Ratio (EAR).

Feel free to use part or all of the code.
