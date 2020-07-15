## Project Write-Up
The people counter app is one of the few applications of using a powerful tool such as OpenVino. The purpose of this application is to determine the number of individuals in a video file which can be used to provide information for a variety of surveillance and detection systems. The following code looks at a few models that have been converted and compared for the same purpose.

In order to test out the various differences in the modelling between the different types of setups for detecting people in video files, this project was done both in local and online versions. The models chosen for this particular version included:-
1. Model 1-ssd_mobilenet_v1_coco_2018_01_28.tar.gz
2. Model 2-Faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz
3. Model 3-ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz
All models were converted after being downloaded as per the steps explained in the Intel Edge IOT foundations cours.The models were tested using the COCO dataset that has a rich number of images for people in various poses. Two other models were loaded from the model zoo database from github which varying speeds and COCO parameters.
## References
Due to the other commitments and work that I am taking up at the moment, I had to take references and work on coding structures from multiple sources. I have however reworked the code to fit to my level of understanding and made changes wherever necessary. I have mainly used https://github.com/denilDG/people-counter-app as a reference. This was again mainly done due to the errors that I was facing during my time with algorithm development and had to consult other sources to make sure I wasn't making any mistakes. I am deeply grateful to the community and their continued support so far. 
## Explaining Custom Layers

The models were selected from the model zoo depending on the reviews and recommendations left by users in terms of their accuracy and potential to handle distinctive figures in images.  

The Faster-rcnn model proved to be the most optimal among the three with a reasonable detection and good accuracy. Both the ssd mobilenet models were reasonable in detections below a threshold, they still could not match up to that of the Faster-rcnn.
Since the custom layers are not in the format compatible for OpenVINO, they have to be converted using the following code obtained from the Foundations course:-

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model model/faster_rcnn_resnet50_lowproposals_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_resnet50_lowproposals_coco_2018_01_28_support.json

python mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.json

In industry problems it becomes very important to be able to convert custom layers as your teams might be developing something new or researching on something and your application to work smoothly you would need to know how you could have support for custom layers.

Another common use case would be when using lambda layers. These layers are where you could add an arbitary peice of code to your model implementation. You would need to have support for these kind of layers and custom layers is your way

## Comparing Model Performance

All three models had to be converted using the model optimizer which generated a .xml and .bin file. The Faster-rcnn model showed better results at predicting the correct number of people in the frames when compared to the unconverted forms of the mobile_net models. 

In order to truly gauge the effect of the model performaces, their performance statistics were measured before and after conversion and then compared against a conventional model to check for parameters such as latency, inference time and memory(MB).

| Models Tested                                       | Latency(Before/After Conversion) (microseconds)      | Memory(Before/After Conversion)|
|____________________________________________________________________________________________________________________________
| MODEL 1-ssd_mobilenet_v1_coco (plain TF)            | 55/60                                                | 29/26                         |
| MODEL 2-Faster_rcnn_resnet50_lowproposals           | 157/165                                              | 365/359                       |
| MODEL 3-ssd_mobilenet_v2_quantized_coco(plain TF)   | 52/64                                                | 67/65                         |


The accuracy of the pre-conversion models was normal to the expected limit at best and improved after conversion. The minimum, average and maximum inference time of the models improved after the conversion when compared to pre conversion accuracies.

The estimated CPU overhead of the pre-conversion models constituted nearly 60% per core cpu while the CPU overhead of the post conversion model lowered to around 45% depending on the network needs and costs of employing cloud for edge app deployment.

## Assess Model Use Cases

The people counter application can be used in several spaces where determining people numbers is crucial to the functioning of the environment or the process. Potential use cases include intruder alert systems, elevators in hospitals to safeguard people numbers, industrial settings to prevent hazards by limiting the number of people through alerts and number of people gathered near fire exits to avoid mishaps.

Each of these use cases would be useful because they would remove the component of human intervention and solely focus on a highly complex yet easy to implement model that supervisors and organizations can keep track of whenever necessary. Such implementations would be especially useful for the health, military, pharmaceuticals, manufacturing and safety industry.
## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these include variations and discrepancies produced on the imaging frames. The lighting and camera focal length affect the methods with which the system generates pixelated lines to distinguish between the various elements in the frames, causing overlaps and errors. For SSD and bounding box algorithm based methods, a high model accuracy is necessary for models that work on the edge to prevent mishaps and failures. Imaging size equally plays an important role to improve or reduce the model accuracy since higher resolution images are more likely to produce better models but must undergo greater processing times and steps before being deployed. Insufficient lighting and image blurriness can also cause great drops in model accuracy due to the deviation from a normal image with properly captured objects and elements.

## Model Research

Model 1: ssd_mobilenet_v1_coco---->[http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz]
The model was converted into the appropriate form using the aforementioned argument in the previous sections. While the model wasn't exactly as resonant and adaptable as the rcnn model, it was still able to detect people numbers to a certain degree. Some results have been stored as images in the images folder for reference. Changing thresholds and other parameters did not yield much changes to the results.

Model 2: Faster_rcnn_resnet50_lowproposals_coco------>[http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz]
Better than the other models and able to generate a good schema with bounding boxes and a proper counting algorithm for the number of people in a frame. Improvements can still be made.

Model 3: ssd_mobilenet_v2_quantized_300x300_coco----->http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz]
Similar issues with Model 1 but still better in performance. Some biases seen in detecting people wearing clothing that matches the surroundings and slight delays in bounding boxes to appear around the people in the frame. Counting algorithm however still worked fine.

## Conclusion

Tensorflow models are better when it comes to building model accurate predictions of people detection in moving frames, considering the inference and the CPU overhead combined with the latency. I am currently running additional models for other types on my local machine and will be updating the results in a separate Github repo. The results in the form of snapshots have been stored in the images folder of the system with a small comparison of how they differ and appear in terms of detecting and counting people. Conversions are ultimately necessary to improve upon models and reduce some of the loads from the system when running servers and such systems.
