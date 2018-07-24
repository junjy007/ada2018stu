# Advanced Data Analytics 2018 Spring

If you are reading this message on a github webpage and have not install `git`. Please do so following [these instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). Then use `git` to update your local version of the course materials from time to time (I am adding new materials as well as making corrections and updates) by 
```
git pull
```

**Keep your local backup if you are not an expert on git** -- if you have done some experiment and then get your project updated from the server, git won't overwrite your local modification. However, for new users, this could cause considerable confusion due to potential conflication. So if you use my examples as template for your homework, or have done any modification, make sure you manage your local version well.

Please check our [Welcome Video](https://www.youtube.com/watch?v=qeyV4fe3vTs), the mind-map mentioned in the video is [here](https://sketchboard.me/xA4SQKJWSZcd#/).

### About this project
Files are organised in weeks. E.g. W0/ contains materials for the preparation week. W1/ contains Jupyter notebooks, and related documents for study in Week 1. 

* Install Anaconda following instructions on its [download page](https://www.anaconda.com/download/). 

I suggest the command line installer. After downloading the installer. If using Mac/Linux, you may need to grant "executable" attribute to the downloaded install script by symbol "#" means comments, don't enter those stuff. The "\$" symbol is mimicking system prompt, don't type the key by yourself, I would ignore this symbol after this example.
```[bash]
# start a terminal window
# change to the download directory, such as 
$cd ~/Downloads
# now working in the download directory ...
$chmod +x Anaconda3-5.2.0-MacOSX-x86_64.sh #
$./Anaconda3-5.2.0-MacOSX-x86_64.sh
# follow prompt hints to finish installation.
```
When the installation finished, you will have Anaconda at somewhere like
`[HOME-DIRECTORY]/Anaconda3/`
The installer would have asked to add the installation directory to the system path
list. This is achieved though adding an `export PATH=[PATH-TO-Anaconda]:$PATH` in the login script `[HOME-DIRECTORY].bash_profile or .bashrc`. To let it take effect, you may need to open a new terminal window. Try to execute `conda` in the new window, and check if `which python3` prints reasonable path (should point to your Anaconda installation).

After installing Anaconda3, you would have most computational tools we need, i.e. you don't need to worry about packages we will use such as `numpy, scipy, panda` by yourself. To follow the example in welcome message and the first week. Only two extra packages are needed, and they are managed (but not installed by default) by Anaconda as well.

* Install pytorch by 
```
  conda install pytorch torchvision -c pytorch
```
If you have a good gaming notebook with nVidia video cards, you can try pytorch with cuda support -- by choosing the correct cuda version on the [download page](https://pytorch.org) (using conda to install if you can), you will have the corresponding installation command. 

* Install scikit-learn by (needed in Week1, not in the Welcome Demo)
```
  conda install scikit-learn
```


I worked all examples on Python 3.5, since 3.6 got issue with pytorch back in Dec 2017. But I believe Python 3.6 should be fine now (July 2018) for all packages, at least on Mac and Linux. Refer to solutions like [here](https://stackoverflow.com/questions/50185227/problems-installing-and-importing-pytorch) if you encounter version problems. 


### About welcome videos
We have a series of videos to gentlely introduce inexperienced users to data analytics with an example of building a hand-written digit recogniser using deep neural networks. Experienced users can skip some or all clips, but I do recommend everyone to have some hand-on experience before class.


[Part 1](https://www.youtube.com/watch?v=1iyONFB1qsY): Prepare computer environment.

[Part 2](https://www.youtube.com/watch?v=YKFzIifrHcQ): Playing with numbers in Python using a numerical library

[Part 3](https://www.youtube.com/watch?v=lsx2_5poOjU): Load the data: convert images (or whatever information) into numbers

[Part 4](https://www.youtube.com/watch?v=4kTOJoP495k): First step investigation: Visualising the data

[Part 5](https://www.youtube.com/watch?v=lUk6ryYhHZM): Build a model for the data

[Part 6](https://www.youtube.com/watch?v=DxQIOhZGr9c): Adjust the model: optimisation

