# CS482/682 Machine Learning Deep Learning Final Project

# Bridging Domains: Harnessing Ensemble and ADDA for Domain Adaptation
Collaborators: Yinong Zhao, Yuan Gao, Emir Syailendra, Star Liu

Unsupervised domain adaptation concerns the scenario where there are labeled images from the source domain dataset and unlabeled images from the target domain dataset.[2] In the context of deep learning, the task is to learn features from the images of one domain and transfer learned representations to classify unlabeled target domain images. Domain adaptation methods such as Adversarial Discriminative Domain Adaptation (ADDA)[3] address the issue of domain shift by aligning disparate domain features into a common feature space. Additionally, ensemble learning techniques that combine multiple models can also improve classification performance and robustness within a domain adaptation context.[4] Our study implements ADDA and ensemble strategies to enhance the adaptability and accuracy of image classification across various domains.

We used the Office-Home3 [5], 4 datasets curated by Jose Eusebio et al. This dataset contains 15,500 images of 65 classes of objects common to most homes and offices. There are 4 domains of im- ages. The first three domains, Artistic images, Clip art, and Product images (ACP), were considered the source domains on which the models were trained. The real-world images (RWI) were the target source images for testing the performance of domain adaptation.

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)

For details on the architectures, please check out the [Final Report](https://github.com/StarLiu1/MLDL_Domain_Adaptation/blob/main/Final%20Report.pdf). All models and scripts are under /src. 
