# Automatic scene extraction based on the Hausdorff distance
This python module contains scripts needed to automatically extract similar scenes from naturalistic traffic datasets. This code accompanies the paper 
"Automatic extraction of similar traffic scenes from large naturalistic datasets using the Hausdorff distance - Siebinga, Zgonnikov & Abbink 2022". 

## Installation instructions
This module cannot be used on its own, its should be used as a sub-module of the TraViA visualization software (click 
[here](https://joss.theoj.org/papers/10.21105/joss.03607) for the paper, or [here](https://github.com/tud-hri/travia) for the repository). If you want to 
use this module, first make sure you have a working version of TraViA. You can clone it directly from github using the following command, or fork it first 
and clone your own version.

```
git clone https://github.com/tud-hri/travia.git
```

After cloning TraViA, you can navigate to the travia folder (`cd travia`) and clone this repository as a submodule. Use the following command to clone the 
github version, or create a fork first and then clone your own fork.

```
git submodule add https://github.com/tud-hri/hausdorffsceneextraction.git
```

This submodule has some additional dependencies besides the dependencies of TraViA itself. Please make sure to install all dependencies by running the 
commands below. 

```
pip install -r requirements.txt
cd hausdorffsceneextraction
pip install -r requirements.txt
```

Instruction on how to get the data and how to work with TraViA can be found in the TraViA README file. See the instructions below for how to work with this 
sub-module.

## How to use
Run the script `extract_situations.py` to find situations with similar traffic context to a specific scenario. First select an example of the situation you 
are interested in. You can do this using TraViA itself. Now use the selected dataset id, vehicle id, and frame number in the `main` block of 
`extract_situations.py`. You can use the parameter `datasets_to_search` to exclude part of the highD dataset and speed things up. Finally you can alter the 
arguments in the function `post_process` to determine which plots to generate.  
