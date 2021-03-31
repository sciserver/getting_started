<h1>Spectrum Viewer</h1>

http://specviewer.idies.jhu.edu

The Spectrum Viewer is an application that allows the visualization and analysis of SDSS spectra, as well as spectra from other surveys.
It runs as a stand-alone web application, and also within Jupyter Lab.

Capabilities:
<ul>
    <li>Toggling of the model and sky fluxes, and associated errors</li>
    <li>Overlaying of spectral lines, including shifting lines to any redshift value.</li>
    <li>Overlaying of spectral mask regions.</li>
    <li>Smoothing the flux with several kernels. User-defined kernels can be added in the Jupyter Lab version of the app.</li>
    <li>Fitting of the continuum or spectral lines with several models. User-defined models can be added in the Jupyter Lab version of the app.</li>
</ul>   

<h3> Web application</h3>

Accesed at http://specviewer.idies.jhu.edu. <br>
Programatic loading of spectra can be done with a REST API. For example: 

http://specviewer.idies.jhu.edu/?specid=2947691243863304192&catalog=sdss

<h3> Jupyter Lab application </h3>

Soon to be added to the standard SciServer docker images. The advantage over the web application version of the viewer is that users can programatically operate the viewer with python commands from within the notebook's cells (for example, adding multiple spectra at the same time, or including user-defined fitting models or smooting functions.

