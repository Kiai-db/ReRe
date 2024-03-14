from Segment_Ingr import getobjects
from classify_ingr import classify_ingr
from generator_basic import recipe_generator
from rottendetector import rottenCNN
from PIL import Image
import cv2


#Alex
Ingredientsimage = Image.open("TestImages/three-rotten-apples-on-white-260nw-264253049.jpeg")

image = cv2.cvtColor(Ingredientsimage, cv2.COLOR_BGR2RGB)  

pil_image_array = getobjects(image)

ingredients = classify_ingr(pil_image_array)   #ingredient_i = (image, pred_class)


#Husain
Class_Per_Ingr = []
Class_Ingr = []
for ingredient in ingredients:
    result = rottenCNN(ingredient(0))
    Class_Per_Ingr.append((result(0),result(1),ingredient(0)))
    Class_Ingr.append((ingredient(0),result(0)))
    if result(1) != "0%":
        print(ingredient(0), " is ",result(1), " rotten. ")
        if (result(1)(0)) > 50:
            print("Bin the ingredient.")
        else:
            print("Ensure to remove the rotten part of the fruit, it is salvagable!")

recipe = recipe_generator(Class_Ingr, "easy", "mexican")

print(recipe)