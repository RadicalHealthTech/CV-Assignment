# Radical Health Technical Task

### **What is the fundus?**

The fundus is the interior lining of the eyeball, including the retina (the light-sensitive screen), optic disc (the head of the nerve to the eye), and the macula (the small spot in the retina where vision is keenest). The fundus is the portion of the inner eye that can be seen during an eye examination by looking through the pupil.  
This is what a fundus image looks like:  
<img src="https://upload.wikimedia.org/wikipedia/commons/3/37/Fundus_photograph_of_normal_right_eye.jpg" width=500/>
  
Multiple modalities of diagnostic devices are available for diagnosing eye diseases. Fundus imaging forms the primary source for diagnosing eye diseases. Furthermore, since the eye provides direct non-invasive imaging of arteries and veins, a fundus image can also help diagnose cardio vascular diseases non-invasively.

You can read more about them here:  
<a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3131209/"> Retinal Imaging and Image Analysis </a>  
<a href="https://www.aao.org/eyenet/article/heart-eye-seeing-links"> The Heart and the Eye </a>  
A couple of recent works by Google   
<a href="https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45732.pdf"> [1] </a>, <a href = "https://scholar.google.com/scholar?lr&ie=UTF-8&oe=UTF-8&q=Predicting+Cardiovascular+Risk+Factors+in+Retinal+Fundus+Photographs+using+Deep+Learning+Poplin+Varadarajan+Blumer+Liu+McConnell+Corrado+Peng+Webster"> [2] </a>, <a href="https://scholar.google.com/scholar?lr&ie=UTF-8&oe=UTF-8&q=Deep+learning+for+predicting+refractive+error+from+retinal+fundus+images+Varadarajan+Poplin+Blumer+Angermueller+Ledsam+Chopra+Keane+Corrado+Peng+Webster"> [3] </a>  
Read more about Google's take on Diabetic Retinopathy which heavily popularized the use of DL in Retinal Imaging <a href="https://ai.googleblog.com/2016/11/deep-learning-for-detection-of-diabetic.html"> here</a>

One of the problems in fundus diagnosing is detecting the quality of the fundus image. If the fundus image is not of diagnosible quality, the doctor has to reorder a fundus examination which leads to slower patient treatment time. Un-diagnosible images also pose problems in training learning algorithms.


### Task  
In this notebook you will aim to implement the EyeQual architecture to determine whether or not the fundus image is of diagnosible quality.
You can find the paper <a href="https://bhooi.github.io/papers/eyequal_icmla17.pdf"> here. </a>

You can use the DRIMDB dataset found <a href="http://academictorrents.com/details/99811ba62918f8e73791d21be29dcc372d660305"> here. </a>

We expect you to:

-> **Perform EDA on the data**  
-> **Write relevant data loaders**  
-> **Implement the model from scratch**  
-> **Perform hyperparameter optimization**   
-> **Explain what the model is learning (check out the authors explainability procedure)**  
-> **Criticize the paper explaining it's drawbacks**  
-> **Suggest alternatives / improvements to the model**  

You are free to use any deep learning framework although PyTorch is preferable.


## Submission

To submit, fork this repository, finish and push your notebook on the forked repository then perform a pull request to this repository.
Deadline for the submission is 4 April 2020 11:59 PM.
