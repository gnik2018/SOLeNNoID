# SOLeNNoID

## Solenoid residue detection and classification in protein structures using a U-Net convolutional neural network

## Introduction
SOLeNNoID is a project consisting of scripts and Jupyter notebooks to allow for the detection and classification of solenoid residues in protein structures using a U-Net convolutional neural network approach on a 2D interatomic distance matrix representation of protein structure.

In its most basic form, the **solennoid.py** script processes all available chains of a **.pdb** file and outputs the predictions as a **.csv** and/or **.pdb** file, where the B-factor values are replaced with solenoid residue prediction numerical values. The structure can then be coloured in PyMOL with a simple command.

## User instructions
To get started with SOLeNNoID all you need to do is:
1. Clone this repository:
```
git clone https://github.com/gnik2018/SOLeNNoID.git
```
2. Run the **solennoid.py** script with the **-i** flag and the path to your structure of interest and select the prediction output format using the **-csv** and/or **-pdb** flags:
```
python3 solennoid.py -i example_run/4w8t.pdb -csv  -pdb
```
3. Inspect the output. Example output can be found in the *example_run/* directory. To colour residues according to solenoid class in PyMOL use the following line:
```
spectrum b, grey_magenta_cyan_orange, minimum=0,maximum=3
```

  The colourscheme is as follows:
  * <span style="color:grey"> *grey* </span>: residues predicted as <span style="color:grey"> *non-solenoid* </span>
  * <span style="color:magenta"> *magenta* </span>: residues predicted as <span style="color:magenta"> *beta-solenoid* </span>
  * <span style="color:cyan"> *cyan* </span>: residues predicted as <span style="color:cyan"> *alpha/beta-solenoid* </span>
  * <span style="color:orange"> *orange* </span>: residues predicted as <span style="color:orange"> *alpha-solenoid* </span>
