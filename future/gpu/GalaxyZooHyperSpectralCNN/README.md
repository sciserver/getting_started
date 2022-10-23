<h1>Deep Learning with GalaxyZoo data on SciServer</h1>

[GalaxyZoo](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo/) is a crowdsourced astronomy project to help with morphological classification of galaxies. This demo will train a deep convolutional neural network to attempt to classify the galaxies automatically.

[PyTorch](https://pytorch.org/) is a deep learning framework that facilitates neural network training and inference with strong GPU acceleration.

To run this example, you will want to create (or use an existing) SciServer Container with PyTorch installed (the "SciServer Essentials" Image) should do.  
<ol>
    <li>Go to the Home tab on your SciServer Dashboard</li>
    <li>Click on Compute from the list of SciServer Apps to open SciServer Compute</li>
    <li>Click the <strong>Create container</strong> button to open the options for starting a virtual computing environment</li>
    <li>Give your new container a name, choose the <strong>GPU Interactive</strong> compute image, and be sure to check the relevant Data Volumes: <strong>Getting Started</strong></li>
    <li>Click Create; your new container will appear in the list</li>
    <li>Click on the name of the container to launch</li>
</ol>

Once you have launched your new container, go to AAS2021Demos/GalaxyZooCNNTraining folder to access the example notebook.
Please copy the contents of this folder to your <strong><code>persistent</code></strong> directory under <strong><code>Storage</code></strong> and have at it!

The notebook you will want to use is "Training GalaxyZoo Hyperspectral Images.ipynb", which will walk you through the training process.
If you have any questions, please email sciserver-helpdesk@jhu.edu.

Have fun!