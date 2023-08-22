# Attempted implementation of CAMOU in PyTorch

(CAMOU: Learning Physical Vehicle Camouflages to Adversarially Attack Detectors in the Wild)[https://openreview.net/pdf?id=SJgEl3A5tm]

I started this project for a research assignment and never finished it due to a shift in research direction. I do not think that the clone network works. 

Main.py allows for the user to take some input image, develop masks on objects in the image, then search for a camouflage that
reduces the detection confidence or makes the objects completely "vanish", at least to the object detector.
