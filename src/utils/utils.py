import os

def save_prediction(predictions, image_file, path):
	"""Saves a prediction of an image from a model to a certain path
	
	Args:
		image_file (str): String containing the filename of the image, eg. verse007.mha
		predictions (list): Python list containing 4-tuples with the predictions in form of
							(vertebrae_label, x, y, z)
		path (str): String containing path to which the save the predictions
	"""
	
	save_file = convert_file_extension_to_txt(image_file)
	
	with open(os.path.join(path, save_file), 'w') as f:
		for prediction in predictions:
			f.write(str(prediction) + "\n")


def load_prediction(image_file, path):
	"""Load a prediction of a model for a certain image
	
	Args:
		image_file (str): String containing the filename of the image, eg. verse007.mha
		path (str): String containing path to where the predictions are saved and should be loaded from
		
	Returns:
		A Python list containing 4-tuples as elements in form of (vertebrae_label, x, y, z)
	"""
	locations = []
	save_file = convert_file_extension_to_txt(image_file)
	
	with open(os.path.join(path, save_file), 'r') as f:
		for line in f:
			locations.append(eval(line.strip()))
			
	return locations


def convert_file_extension_to_txt(image_file):
	"""Convert and a file extension to .txt

	Args:
		image_file (str): String containing the image file

	Returns:
		Initial string where extension is changed to '.txt'
		
	Examples:
		>>> convert_file_extension_to_txt('verse007.mha')
		 'verse007.txt'

	"""
	return image_file.split('.')[0] + '.txt'