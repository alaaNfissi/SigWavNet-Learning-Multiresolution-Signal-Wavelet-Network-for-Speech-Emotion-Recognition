<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />

<div align="center">
  <a href="https://github.com/alaaNfissi/SigWavNet-Fully-Deep-Learning-Multiresolution-Wavelet-Transform-for-Speech-Emotion-Recognition">
    <img src="figures/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">SigWavNet: Learning Multiresolution Signal Wavelet Network for Speech Emotion Recognition</h3>

  <p align="center">
    This paper has been submitted for publication in IEEE Transactions on Affective Computing.
    <br />
   </p>
   <!-- <a href="https://github.com/alaaNfissi/SigWavNet-Fully-Deep-Learning-Multiresolution-Wavelet-Transform-for-Speech-Emotion-Recognition"><strong>Explore the docs »</strong></a> -->
</div>
   

  
<div align="center">

[![view - Documentation](https://img.shields.io/badge/view-Documentation-blue?style=for-the-badge)](https://github.com/alaaNfissi/CNN-n-GRU-for-speech-emotion-recognition/#readme "Go to project documentation")

</div>  


<div align="center">
    <p align="center">
    ·
    <a href="https://github.com/alaaNfissi/SigWavNet-Fully-Deep-Learning-Multiresolution-Wavelet-Transform-for-Speech-Emotion-Recognition/issues">Report Bug</a>
    ·
    <a href="https://github.com/alaaNfissi/SigWavNet-Fully-Deep-Learning-Multiresolution-Wavelet-Transform-for-Speech-Emotion-Recognition/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#abstract">Abstract</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#getting-the-code">Getting the code</a></li>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#reproducing-the-results">Reproducing the results</a></li>
      </ul>
    </li>
    <li>
      <a href="#results">Results</a>
      <ul>
        <li><a href="#on-iemocap-dataset">On IEMOCAP dataset</a></li>
        <li><a href="#on-tess-dataset">On EMO-DB dataset</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABSTRACT -->
## Abstract

<p align="justify"> In the field of human-computer interaction and psychological assessment, speech emotion recognition (SER) plays a vital role in deciphering emotional states from speech signals. Despite advancements, challenges persist due to system complexity, feature distinctiveness issues, and noise interference. This paper introduces a new end-to-end (E2E) deep learning multi-resolution framework for SER, addressing these limitations by extracting meaningful representations directly from raw waveform speech signals. Leveraging the properties of the fast discrete wavelet transform (FDWT), including the cascade algorithm, conjugate quadrature filter, and coefficient denoising, our approach introduces a learnable model for both wavelet bases and denoising through deep learning techniques. The framework incorporates an activation function for learnable asymmetric hard thresholding of wavelet coefficients. By exploring the competency of wavelets in achieving effective localization in time and frequency domains, we integrate one-dimensional dilated convolutional neural networks (1D dilated CNN) with spatial attention layer and bidirectional gated recurrent units (Bi-GRU) with temporal attention layer to capture emotion features and temporal-based characteristics, respectively. By handling variable-length speech without segmentation and eliminating the need for pre- or post-processing, the proposed model demonstrated its efficiency on IEMOCAP and EMO-DB datasets. Results showcase improved performance over existing methods, underscoring the potential of our E2E approach to advancing SER.</p>
<div align="center">
  
![model-architecture][model-architecture]
  
*L-LFDWTB SigWavNet General Architecture*
  
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With
* ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
* ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
* ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
* ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
* ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
* ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started
<p align="justify">
To begin our experiments, we first ensured that our signal has a sampling rate of 16 KHz and is mono-channel in order to standardise our experimental data format.
Each dataset is segmented as follows: 80\% for training, 10\% for validation, and 10\% for testing based on stratified random sampling which entails categorising the whole population into homogenous groupings known as strata. Random samples are then drawn from each stratum unlike basic random sampling which considers all members of a population as equal. With an equal possibility of being sampled, it allows us to generate a sample population that best represents the total population being studied as it is used to emphasise distinctions across groups in a population. A Grid search is then used to find the appropriate hyperparameters. Some hyperparameter optimization approaches are known as "scheduling algorithms". These Trial Schedulers have the authority to terminate troublesome trials early, halt trials, clone trials, and alter trial hyperparameters while they are still running. Thus, the Asynchronous Successive Halving algorithm (ASHA) was picked because of its high performance.
  
We examined four model architectures: CNN-3-GRU, CNN-5-GRU, CNN-11-GRU, and CNN-18-GRU. Each model is run for 100 epochs until it converges using Adam. As we are not using any pretrained model, the weights of each model are started from scratch. The receptive field of our first CNN layer is equal to <em>160</em> which corresponds to <em>(sampling rate / 100)</em> in our case to cover a <em>10-millisecond</em> time span, to be comparable to the window size for many MFCC computations since we transformed all our data to <em>16 KHz</em> representation. All source code used to generate the results and figures in the paper are in
the `CNN-n-GRU_IEMOCAP` and `CNN-n-GRU_TESS` folders. The calculations and figure generation are all run inside [Jupyter notebooks](http://jupyter.org/).
The data preprocessing used in this study is provided in `Data_exploration` folder. See the `README.md` files in each directory for a full description.  
</p>

### Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/alaaNfissi/SigWavNet-Fully-Deep-Learning-Multiresolution-Wavelet-Transform-for-Speech-Emotion-Recognition.git

or [download a zip archive](https://github.com/alaaNfissi/SigWavNet-Fully-Deep-Learning-Multiresolution-Wavelet-Transform-for-Speech-Emotion-Recognition/archive/refs/heads/main.zip).

### Dependencies

<p align="center">

You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.
The required dependencies are specified in the file `requirements.txt`.
We use `conda` virtual environments to manage the project dependencies in
isolation.
Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).
Run the following command to create an `ser-env` environment to create a separate environment:
```sh 
    conda create --name ser-env
```
Activate the environment, this will enable the it for your current terminal session. Any subsequent commands will use software that is installed in the environment:
```sh 
    conda activate ser-env
 ``` 
Use Pip to install packages to Anaconda Environment:
```sh 
    conda install pip
```
Install all required dependencies in it:
```sh
    pip install -r requirements.txt
```
  
</p>

### Reproducing the results

<p align="center">  
  
1. First, you need to download IEMOCAP and TESS datasets:
  * [IEMOCAP official website](https://sail.usc.edu/iemocap/)
  * [TESS official website](https://tspace.library.utoronto.ca/handle/1807/24487)
  
2. To be able to explore the data you need to execute the Jupyter notebook that prepares the `csv` files needed for the experiments.
To do this, you must first start the notebook server by going into the
repository top level and running:
```sh 
    jupyter notebook
```
This will start the server and open your default web browser to the Jupyter
interface. In the page, go into the `Data_exploration` folder and select the
`data_exploration.ipynb` notebook to view/run. Make sure to specify the correct datasets paths on your own machine as described in the notebook.
The notebook is divided into cells (some have text while other have code).
Each cell can be executed using `Shift + Enter`.
Executing text cells does nothing and executing code cells runs the code
and produces it's output.
To execute the whole notebook, run all cells in order.
 
3. After generating the needed `csv` files `IEMOCAP_dataset.csv` and `TESS_dataset.csv`, go to your terminal where the `ser-env` environment was
  activated and go to `CNN-n-GRU_IEMOCAP` folder and choose one of the python files to run the experiment that you want. For example:
```sh  
python iemocap_cnn_3_gru.py
``` 
  _You can do the same thing for the TESS dataset by going to the `CNN-n-GRU_IEMOCAP` and runing one of the python files._

</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Results
<p align="center">  
  
We implemented the proposed architecture CNN-n-GRU in four versions, with n = 3, 5, 11, and 18.
  
</p>

### On IEMOCAP dataset
<p align="center">  
  
Amoung our model’s four versions performance, the best architecture of our model is CNN-18-GRU as it achieves the highest accuracy and F1-score, 
where it reaches 81.3% accuracy and 80.9% F1-score on the IEMOCAP dataset which is better compared to the state of-the-art methods.
The CNN-18-GRU training and validation accuracy over epochs figure shows the evolution of training and validation accuracy of the CNN-18-GRU over 100 epochs. The confusion matrix in CNN-18-GRU confusion matrix figure describes class-wise test results of the CNN18-GRU. 

</p>

CNN-18-GRU training and validation accuracy over epochs            |  CNN-18-GRU confusion matrix
:-----------------------------------------------------------------:|:-----------------------------:
![iemocap_cnn18gru_acc](images/iemocap_cnn18gru_acc.png)  |  ![iemocap_cnn18gru_confusion_matrix_1](images/iemocap_cnn18gru_confusion_matrix_1.png)


### On TESS dataset
<p align="center"> 
  
Amoung our model’s four versions performance, the best architecture of our model is CNN-18-GRU as it achieves the highest accuracy and F1-score, 
where it reaches  99.2% accuracy and 99% F1-score on the TESS dataset which is better compared to the state of-the-art methods.
The CNN-18-GRU training and validation accuracy over epochs figure shows the evolution of training and validation accuracy of the CNN-18-GRU over 100 epochs. The confusion matrix in CNN-18-GRU confusion matrix figure describes class-wise test results of the CNN18-GRU.  

</p>

CNN-18-GRU training and validation accuracy over epochs            |  CNN-18-GRU confusion matrix
:-----------------------------------------------------------------:|:-----------------------------:
![cnn18gru_acc](images/cnn18gru_acc.png)  |  ![cnn18gru_confusion_matrix_1](images/cnn18gru_confusion_matrix.png)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<p align="center">
  
_For more detailed experiments and results you can read the paper._
  
</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Alaa Nfissi - [@LinkedIn](https://www.linkedin.com/in/alaa-nfissi/) - alaa.nfissi@mail.concordia.ca

Github Link: [https://github.com/alaaNfissi/CNN-n-GRU-for-speech-emotion-recognition](https://github.com/alaaNfissi/CNN-n-GRU-for-speech-emotion-recognition)

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[model-architecture]: figures/Aggregated_SigWavNet_V.png


[anaconda.com]: https://anaconda.org/conda-forge/mlconjug/badges/version.svg
[anaconda-url]: https://anaconda.org/conda-forge/mlconjug

[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
