## EECS 545 Project

#### Collaboration Workflow
Basic overview of git:
https://www.youtube.com/watch?v=0fKg7e37bQE

##### Initial Steps
1. Fork this repo to your own github account
2. clone the repo to your computer
	1. ```git remote add upstream https://github.com/apolanco115/EECS545-proj.git``` <-- this makes sure your fork is up to date with main project
	2. ```git remote -v``` <-- verify that the previous step was done correctly

#### Contributing code
1. ```git pull --rebase``` <-- so that you are working with the newest version of the code
2. write some code
3. commit your commit your code, and commit often (write useful commit comments)
4. When you feel like you've got working code worth merging with main project, ```git pull --rebase``` <-- this will save you from annoying headaches
5. push your changes
6. create a pull request to *this* repo
7. we'll review together to make sure we don't overwrite our hard work.

### Literature Review References
1. https://deepmind.com/blog/article/deep-reinforcement-learning
2. https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
3. https://arxiv.org/pdf/1602.01783.pdf
