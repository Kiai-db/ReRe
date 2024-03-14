from Segment_Ingr import getobjects
from classify_ingr import classify_ingr
from generator_basic import generator
from rottendetector import rottenCNN
from PIL import Image
import cv2


#Alex
Ingredientsimage = Image.open("ingredients.png")

image = cv2.cvtColor(Ingredientsimage, cv2.COLOR_BGR2RGB)  

pil_image_array = getobjects(image)

ingredients = classify_ingr(pil_image_array)   #ingredient_i = (image, pred_class)


#Husain
rottenClass = []
for ingredient in ingredients:
    result = rottenCNN(ingredient(0))
    rottenClass.append((result,ingredient(0)))



