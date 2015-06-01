Latent Dirichlet Allocation for Substructure Detection of MS2 Peaks in Metabolomics
===================================================================================

This folder contains various codes to run topic modelling via Latent Dirichlet Allocation (LDA) on MS2 peaks produced by liquid-chromatography mass-spectrometry (LC-MS) experiments in metabolomics.

Explanations of important files and folders:

* **lda_cgs.py** contains an implementation of the collapsed Gibbs sampling for LDA.
* **lda_for_fragments.py** contains the code to load the MS2 data matrices, pass them to the LDA model and plot the results. 
* **notebooks** contains various IPython notebooks. This is probably the easiest way to run the project for now. 
* **input** contains all the MS2 data matrices in CSV format that are used as input. 
* **R** contains R-script to do feature extraction from mzXML files to CSV.

Windows Installation Guide
==========================

Setting up the development environment
--------------------------------------

1. First we need to check out the project. Go to https://windows.github.com/ and download the GitHub for Windows client. Install it. This provides a tool to clone our source-code repository.
 
2. During the installation of the GitHub client, **enter the details of your GitHub user account**. If you haven't got one, create it at https://github.com/join. 

3. Using your web browser, head to the repository page at https://github.com/sdrogers/metabolomics_tools. Click on the **Clone in Desktop** button on the right side of the screen. A dialog box will appear asking where to clone the repository. The default location is to place the cloned repository inside the Documents\GitHub folder of the current user. However, you can select any other folder that you want as the location. Click **OK** to proceed.

4. A progress bar will appear to indicate how long the cloning process takes. The cloning process might take a while because of large notebook files, which we probably shouldn't put in the repository ..

5. Implementation of the LDA inference for substructure detection in MS2 peaks project is primarily done in the Python scientific environment, so we need to set-up the Python interpreter and its necessary packages. These are primarily the [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/) packages, alongside with other usual stuff like IPython for interactive console, Matplotlib for plotting, Pandas for data frames etc. If you already have a Python environment (with Numpy/Scipy) installed on your machine, feel free to continue using that. Otherwise, use the Anaconda Python distribution from Continuum Analytics that provides [a one-click installer for all the packages required](https://store.continuum.io/cshop/anaconda/). **Open the link and click download** and select the Windows version to download it. The following steps of this installation guide will be written with the assumption that you are doing a clean installation 

6. **Launch the installer for Anaconda Python** and proceed with the installation process. Accept all the default options and wait for installation to finish. 

7. Upon completion, **launch "IPython (2.7)" from the the newly-created "Anaconda" folder in the Start menu** to verify that installation is successful. You should see something like "Python 2.7 ... " on the first line. Type "exit()" to quit.

8. Now we need to install some of the additional packages used in the project (on top of the usual NumPy/SciPy packages). **Launch "Anaconda Command Prompt" from the "Anaconda" start menu folder.** In the command-prompt window that appears, type the following commands to install the previous LDA implementation that we used (to be removed) and update all packages in the distribution to the latest:

> pip install lda

> conda update --all

> exit

Running the project
-------------------

1. Finally we have reached the stage where we can run stuff!! Under the same start-up menu folder, you can also **launch the "IPython (2.7) Notebook"**. This will open the notebook client in the web browser. You can then navigate to the **Notebooks** folder in the cloned repository location from step (3) in the previous section above. The default is in "Documents\GitHub\metabolomics_tools\justin\notebooks". Click on one of the notebooks to load it. See **lda_results_beer_3_pos_plots.ipynb** for an example of a nicely documented notebook. 

2. From the notebook menu, click **Cell** followed by **Run All** to re-run everything. Alternatively, you can run each cell in the notebook sequentially by first selecting a cell, then pressing Shift-Enter to run that cell and moving on to the next one. 

3. It is also often useful to have some sort of text editor or an integrated development environment to edit and make changes to the source codes etc. I recommend the [PyCharm Community Edition IDE](https://www.jetbrains.com/pycharm/download/) or at least the [Notepad++ text editor](https://notepad-plus-plus.org/).
