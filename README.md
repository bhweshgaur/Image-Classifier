# Deep Learning 
## Project: Image Classifier
### Install
This project requires **Python 3.x** and the following Python libraries installed:
- Numpy
- Json
- Matplotlib
- Tensorflow

You can run this project either on iPython Notebook or Colab Notebook. I recommend using Colab Notebook, because in iPython Notebook you might get error while loading dataset due to heavy traffic.

Either of the notebook require **%pip install tfds-nightly** command to run before loading data.

### Code
Code is provided in the `Project_Image_Classifier_Project.ipynb` notebook file. `Model.h5 and label_file.zip` contains *model.h5* (loaded model), *label_map.json* (mapped flower labels) and few images for testing the model. `predict.py` is used to run the project through terminal. `Project_Image_Classifier_Project.html` file is the same code in html format.

### Run
You can run notebook file by simply opening you terminal and writing 

```bash
jupyter-notebook Project_Image_Classifier_Project.ipynb
```

To run application file, open terminal and write 
```bash
python predict.py <image_path> model.h5 <value of K(integer)>
```

where K is the no. of most likely flowers you want model to predict.

### Output
This project will predict the most likely K(integer) number of images of the input photo.
