# Predicting Lower Back Pain Causes and Mortality Chances Post-Thoracic Surgery
#### _Focus: Machine Learning (K-NN Classifiers), Exploratory Data Analysis_



Spondylolysis is one of the most common causes of lower back pain in regular exercisers and for the elderly. It remains asymptomatic in the majority of people who have it. This is detrimental as spondylolysis can progress to spondylolisthesis, which is defined as anterior displacement of the vertebral body in reference to the bordering vertebral bodies. When you injure your back, many tend to dismiss the asymptomatic sign in belief that you perhaps just got a muscle cramp or deadlifted a little too heavy but that in the end run will heal from it just like any other muscle recovery program for miniscular injuries. What people fail to realize is that there are an uncountable amount of variables that can form spondylolysis and often it is extremely hard to know if you ever have it. At abnormally high grade level for spondylilothesis, you can be very likely to have your lower back discs forever dislodged if not pursuing surgery and this can be life changing in a traumatically negative way -- leaving you in a wheelchair or even to some rare occasions dead. Mortality rate is not zero when talking about no treatment after having this for a while and even with surgery it can be dangerous for as to what side effects will result as this is still a niche field.

Knowing one's level of dorsal health is crucial for not only general surgery-inquiries or general health self-evaluation, but also with knowledge of one's lumbar erosive-eseque features, as you can then determine what your solution procedure should be to get rid of whatever your pain may be. Realizing the potential harm in not knowing if one potenitial has isthmic spondylolisthesis and to what level of concern, I felt the relevancy in pursuing a project on this for both patients in the fitness industry to mitigate putting one's back at further risk and to see if there is a way to predict if a middle-aged person has this asymptomatic condition.



<div>
<img src="https://ars.els-cdn.com/content/image/1-s2.0-S2666548420300111-gr2.jpg" width="500"/>
</div>

---------------------


_Chapters:_
* PART ONE: 'Quantiative' 
    - _Areas of focus revolve around Anatomical Scrutinizing (slip disk), Correlations, Predicting, Classifications_
* PART TWO: 'Categorical'
    - _Areas of focus revolve around Elderly, Correlations, Predicting, Classifications, Discovery_
* PART THREE: 'Prediction'
    - _Areas of focus revolve Machine Learning, Train-Test Split, K-NN Algorithmn, Classification Reports, Accuracy Score, Confusion Matrix
    
* PART FOUR: (Coming Soon) 'Computer Vision' 
    - _Areas of focus revolve around Fitness, Elderly, Gesture Control, Computer Vision, API/GUI, Posture Correctness_


-------------------------------------------------------------------------------------
##### <font color='Blue'> In this project, I will be using two UCI derived repositories. The first dataset is anatomically related and showcasing quantitative biomechanic metrics, while the second set is a rendered symptomatic survey of categorical data displaying pre- and post-surgery questionnaire answers. </font>

##### Both experiments occured in Warsaw and I personally discovered such data from Kaggle's database. The spinal dataset is going to be for the purpose of determining if we can stratify clumps of parcticipant' data into grades of spondylolithesis, as well as to determine association of slip grade to pelvic tilt. The thoracic surgery dataset will be centered around lower-grade spondylolithesis participants, discovering relations (if applicable) to developement of scoliosis as well as seeing if I can find a pattern to predict if a middle-aged person will not only need to attend surgery, but if they would survive long after the procedure considering their pre-operative symptoms. Sometimes, there are risks to take but other times it is not worth the sacrifice. I hope this project will serve well for readers of any age. 

##### <font color = 'green'> ~ For readers new to Jupyter Notebook, please press _ctrl+enter_ to view each cell's results. Happy scrolling! ~  </font>

-------------------------------------------------------------------------------------
<u> Dataset Sources: </u>
* Siddhartha, Manu. “Thoracic Surgery Dataset.” Kaggle, 29 July 2019, \
https://www.kaggle.com/sid321axn/thoraric-surgery
* Hussain, Ali. “Lower Back Pain Symptoms Dataset(Labeled).” Kaggle, 5 Dec. 2017, \
https://www.kaggle.com/alihussain1993/lower-back-pain-symptoms-datasetlabelled
-------------------------------------------------------------------------------------



<u> <b> Research Background </b> </u>
 
 Spinal measurements are clinically significant for a spinal surgeon before suggesting or shortlisting suitable surgical intervention procedure. Traditionally, the spinal surgeon evaluates the condition of the parcticipant prior to a surgical procedure, so as to verify the usefulness and effectiveness of the adopted procedure. In the case of spinal fusion procedures for your L3 vertebrae, where the lower back region prone to disk slips, will the fusion procedure be able to restore the spinal balance is a question for which the answered is obtained through making relevant spinal measurements, including lumbar lordotic curve angle, both segmental and for whole lumbar spine, lumbosacral angle, spinal heights, dimensions of vertebral bodies etc.  

Spondylolysis is defined as an anatomical defect or fracture of the pars interarticularis of the vertebral arch. The pars interarticularis is an isthmus of bone connecting the superior and inferior facet surfaces in the spine at a given level. Spondylolysis occurs at the L5 vertebrae between 85 and 95% of the time and occurs at the L4 vertebrae 5–15% of the time. The two lowest sections on the spinal cord is the L4 followed underneath it with the L5 (Sports Medicine, 3).


<div>
<img src="https://els-jbs-prod-cdn.jbs.elsevierhealth.com/cms/attachment/bd0824f4-5b82-4fa1-9b8c-e0c2eac3c6cc/gr1_lrg.jpg" width="500"/>
</div>

 <font color='Green'> _While Degenerative spondylolisthesis occurs mostly at the L4-L5 (fourth and fifth lumbar vertebrae) level as opposed to its isthmic counterpart, which occurs most often at the lumbosacral level (L5-S1)_ </font>. When isthmic spondylolisthesis results in the irritation or impingement of a nerve root, it can result in sciatica pain. Spondololisthesis may vary in terms of its causuality, but often it is either through excessive external loading (eg. weightlifting) or through age, where disk herniations begin as result of bone erosion. Sciatica refers to pain that radiates along the path of the sciatic nerve, which branches from your lower back through your hips and buttocks and down each leg. It is a condition in which compression of one of its these roots or branches occurs due to excessive loading typically. A sharp shooting pain in the leg, usually on one side, is a very common symptom to look out for.
 
Pelvic incidence, sacral slope and slip percentage have been shown to be important predicting factors for assessing the risk of progression of low and high grade spondylolisthesis. To determine when someone has IS (Isthmic Spondylolisthesis), one can verify through Spino-Pelvic Paramters (PI (Pelvic Incidence), Slip Percentage, and PT (Pelvic Tilt), SS (Sacral slope), LSA (Lumbosacral Angle)) in close scrutiny with the progression of IS -- assesing the differences of parameters of those undergoing thoracic surgery with someone who has IS to better understand the influence IS has on pre & most importantly post-thoracic surgery (Chung et al, 2).

![Image of Spondylolithesis Stages](https://neupsykey.com/wp-content/uploads/2016/09/A324122_1_En_8_Fig1_HTML.gif)

Parcticipants undergo lumbar pedicle screw fixation and fusion for degenerative spondylolisthesis. In aged yet remaninigly accurate study by _Duval-Beaupère_, the SS, PT, and PI remain the top three most cruicial variables to idetnify key patterns to more easily dtermine the state one's spondylolithesis stage is at. There are 4 grades in spondylolithesis, each representing a level of severity and need for surgery before death: Stage I-IV. Grade 1 is when the degree of pelvic tilt is <25% slippage angle. Grade 2 is 25-50% slippage. Grade 3 is 50-75% slippage. Grade 4 is >75% slippage. As degrees of slippage is a percentage, just convert decimal to percentage if given in decimal format. GRADE 3 onwards need surgery.

