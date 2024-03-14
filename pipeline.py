from Segment_Ingr import getobjects
from classify_ingr import classify_ingr
# from generator_basic import recipe_generator
from rottendetector import rottenCNN
from PIL import Image
import cv2
import numpy as np

#Alex
Ingredientsimage = Image.open("TestImages/MicrosoftTeams-image.png")

# Convert PIL Image to NumPy array before using cv2.cvtColor
image = cv2.cvtColor(np.array(Ingredientsimage), cv2.COLOR_BGR2RGB)

pil_image_array = getobjects(image)

ingredients = classify_ingr(pil_image_array)   #ingredient_i = (image, pred_class)

#Husain
Class_Per_Ingr = []
Class_Ingr = []
for ingredient in ingredients:
    result = rottenCNN([ingredient[0]])
    Class_Per_Ingr.append((result[0], result[1], ingredient[0]))
    Class_Ingr.append((ingredient[0], result[0]))
        
    if result[1] > 0:  
        print(f"{ingredient[0]} is {result[1]}% rotten.")
        if result[1] > 50:
            print("Bin the ingredient.")
        else:
                print("Ensure to remove the rotten part of the fruit, it is salvageable!")

#recipe = recipe_generator(Class_Ingr, "easy", "mexican")
#print(recipe)
