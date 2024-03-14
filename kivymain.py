from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.logger import Logger
from plyer import filechooser, camera 
from PIL import Image
import cv2
import numpy as np

from Segment_Ingr import getobjects
from classify_ingr import classify_ingr
from rottendetector import rottenCNN

class MainApp(App):
    def build(self):
        root_layout = BoxLayout(orientation='vertical')
        
        # Top buttons for camera and upload
        top_buttons = BoxLayout(size_hint_y=None, height='48dp')
        take_picture_btn = Button(text='Take Picture')
        take_picture_btn.bind(on_press=self.take_picture)
        top_buttons.add_widget(take_picture_btn)
        
        upload_image_btn = Button(text='Upload Image')
        upload_image_btn.bind(on_press=self.upload_image)
        top_buttons.add_widget(upload_image_btn)
        root_layout.add_widget(top_buttons)
        
        # Spacer to push bottom buttons to the bottom
        spacer = BoxLayout(size_hint_y=1)
        root_layout.add_widget(spacer)
        
        # Bottom buttons
        bottom_buttons = BoxLayout(size_hint_y=None, height='48dp')
        for i in range(1, 5):
            btn = Button(text=f'Button {i}')
            bottom_buttons.add_widget(btn)
        root_layout.add_widget(bottom_buttons)
        
        return root_layout

    def take_picture(self, instance):
        try:
            camera.take_picture(filename='picture.jpg',
                                on_complete=self.on_picture_taken)
        except NotImplementedError:
            Logger.warning('Camera: This feature is not implemented on your platform.')

    def on_picture_taken(self, filename):
        Logger.info(f'Picture taken and saved to {filename}')

    def upload_image(self, instance):
        try:
            filechooser.open_file(on_selection=self.on_file_chosen)
        except NotImplementedError:
            Logger.warning('Filechooser: This feature is not implemented on your platform.')

    def on_file_chosen(self, selection):
        if selection:
            Logger.info(f'Selected: {selection[0]}')

    def on_file_chosen(self, selection):
        if selection:
            filepath = selection[0]
            Logger.info(f'Selected: {filepath}')
            self.process_image(filepath)

    def process_image(self, image_path):
        # Load and process the image
        try:
            Ingredientsimage = Image.open(image_path)
            image = cv2.cvtColor(np.array(Ingredientsimage), cv2.COLOR_BGR2RGB)
            pil_image_array = getobjects(image)
            ingredients = classify_ingr(pil_image_array)
            Logger.info(f'Ingredients Classified: {ingredients}')
            
            for ingredient in ingredients:
                result = rottenCNN([ingredient[0]])
                if result[1] > 0:
                    status = f"{ingredient[1]} is {result[1]}% rotten."
                    if result[1] > 50:
                        Logger.info(status + " Bin the ingredient.")
                    else:
                        Logger.info(status + " Ensure to remove the rotten part of the fruit, it is salvageable!")
        except Exception as e:
            Logger.error(f'Error processing image: {e}')

if __name__ == '__main__':
    MainApp().run()