# Table Tennis Scorekeeper

This is the code for an automated YOLOv8-based table tennis scorekeeping system. It was written by Sam Boese, Chris Brown, and John Woods as a term project for the CIS 530 course at Kansas State University taught by Dr. William Hsu.

## Using the Scorekeeping Application

In order to process a video, one first needs to make sure that the input video and the ball and events models are in the same directory as the scorekeeper file. Additionally, the variable `video_path` in the scorekeeper file needs to be changed to the input video name. Once this is done, one can simply start the program by first opening the terminal or another interface that has python installed, and then executing the command `$ python scorekeeper.py `. The resulting output will be a file named `output_video.mp4` in the same directory.

## Generating the Data

The Python data generation scripts assume that you have all the frames extracted from all the videos in a specific location on your computer. Since this is hard to reproduce, .zip files containing the data used to train the YOLOv8 models are also included. Simply unzip and move into a folder called `datasets` in the root directory of the project, and it will be discovered by the YOLO CLI.

## Training the models

Pretrained models already exist in this repository under the `runs` folder. The Beocat training scripts used are also provided under the `scripts `folder. Simply clone this repository onto Beocat, extract the data as described above, and run the commands `$ sbatch train_ball_model.sh` and `$ sbatch train_event_model.sh` to train both models.

## Plotting and Testing

Simply run `$ python plot.py` to generate the results plots for the models, and run `$ python test.py` to perform the t-tests for the models.
