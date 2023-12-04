"""
This is a main script to run the hate meme detector pipeline. 
"""

class HateMemeDetector:
    def __init__(self, image_caption, text_extractor, text_analyzer, clip_extractor, hate_classifier):
        self.image_caption = image_caption
        self.text_extractor = text_extractor
        self.text_analyzer = text_analyzer
        self.clip_extractor = clip_extractor
        self.hate_classifier = hate_classifier
    
    def run(self, image_path):
        # get image caption


        # get text on image using OCR and clean the image from text.
         
        
        # concatenate the image caption and text on image then analyze them.

        # get the features from Clip model.

        # concatenate the features from Clip model and the features from text analysis then classify them.

        # return the result.

        pass