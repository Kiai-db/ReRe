import tkinter as tk
from tkinter import ttk
import settings
import openai
import time


class IngredientsApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Select Ingredients")

        # Initialize ingredients list
        self.ingredientsList = settings.ingredientsList

        self.ingredients = []

        self.menu_frame = tk.Frame(self.master)
        self.menu_frame.grid(row=0, column=0, padx=20, pady=20)

        self.ingredients_label = tk.Label(self.menu_frame, text="Select Ingredients:")
        self.ingredients_label.grid(row=0, column=0, columnspan=2, sticky=tk.W)

        # Checkboxes
        self.tomato_var = tk.IntVar()
        self.tomato_check = tk.Checkbutton(self.menu_frame, text="Tomato", variable=self.tomato_var, command=self.update_ingredients)
        self.tomato_check.grid(row=1, column=0, sticky=tk.W)

        self.chicken_var = tk.IntVar()
        self.chicken_check = tk.Checkbutton(self.menu_frame, text="chicken", variable=self.chicken_var, command=self.update_ingredients)
        self.chicken_check.grid(row=2, column=0, sticky=tk.W)

        self.lettuce_var = tk.IntVar()
        self.lettuce_check = tk.Checkbutton(self.menu_frame, text="Lettuce", variable=self.lettuce_var, command=self.update_ingredients)
        self.lettuce_check.grid(row=3, column=0, sticky=tk.W)

        self.cheese_var = tk.IntVar()
        self.cheese_check = tk.Checkbutton(self.menu_frame, text="Cheese", variable=self.cheese_var, command=self.update_ingredients)
        self.cheese_check.grid(row=4, column=0, sticky=tk.W)

        self.onion_var = tk.IntVar()
        self.onion_check = tk.Checkbutton(self.menu_frame, text="Onion", variable=self.onion_var, command=self.update_ingredients)
        self.onion_check.grid(row=5, column=0, sticky=tk.W)

        self.beef_var = tk.IntVar()
        self.beef_check = tk.Checkbutton(self.menu_frame, text="beef", variable=self.beef_var, command=self.update_ingredients)
        self.beef_check.grid(row=6, column=0, sticky=tk.W)

        self.Pepper_var = tk.IntVar()
        self.Pepper_check = tk.Checkbutton(self.menu_frame, text="Pepper", variable=self.Pepper_var, command=self.update_ingredients)
        self.Pepper_check.grid(row=7, column=0, sticky=tk.W)

        self.difficulty_label = tk.Label(self.menu_frame, text="Select Difficulty:")
        self.difficulty_label.grid(row=0, column=2, sticky=tk.W, padx = 10)  
        # Dropdown list
        self.dropdown_var = tk.StringVar()
        self.dropdown = ttk.Combobox(self.menu_frame, textvariable=self.dropdown_var)
        self.dropdown['values'] = ('easy', 'intermediate', 'challenging')  # Add your options here
        self.dropdown.grid(row=1, column=2, sticky=tk.E, padx = 10)

        # Dropdown for cuisine
        self.type_label = tk.Label(self.menu_frame, text="Cuisine Type:")
        self.type_label.grid(row=0, column=3, sticky=tk.W, padx = 10)

        self.cuisine_var = tk.StringVar()
        self.cuisine = ttk.Combobox(self.menu_frame, textvariable=self.cuisine_var)
        self.cuisine['values'] = ('Italian', 'French', 'Chinese', 'Japanese', 'Indian', 'Mexican', 'Thai', 'Greek', 'Spanish', 'Lebanese', 'Korean', 'Vietnamese', 'Moroccan', 'Turkish', 'American', 'Brazilian', 'German', 'Russian', 'Ethiopian', 'Jamaican')  # Add your options here
        self.cuisine.grid(row=1, column=3, sticky=tk.E, padx = 10)

        self.submit_button = tk.Button(self.master, text="Submit", command=self.display_ingredients_list)
        self.submit_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        self.ingredients_text = tk.Text(self.master, height=10, width=30)
        self.ingredients_text.grid(row=0, column=1, padx=10, pady=10)

    def update_ingredients(self):
        # Clear the ingredients list
        self.ingredients.clear()

        # Checkboxes
        if self.tomato_var.get() == 1:
            self.ingredients.append("Tomato")
        if self.lettuce_var.get() == 1:
            self.ingredients.append("Lettuce")
        if self.cheese_var.get() == 1:
            self.ingredients.append("Cheese")
        if self.onion_var.get() == 1:
            self.ingredients.append("Onion")
        if self.beef_var.get() == 1:
            self.ingredients.append("beef")
        if self.Pepper_var.get() == 1:
            self.ingredients.append("Pepper")
        if self.chicken_var.get() == 1:
            self.ingredients.append("chicken")


        # Update ingredients list display
        self.display_ingredients()

    def display_ingredients(self):
        self.ingredients_text.delete('1.0', tk.END)
        for ingredient in self.ingredients:
            self.ingredients_text.insert(tk.END, ingredient + "\n")

    def display_ingredients_list(self):
        # Update selected ingredients
        self.update_ingredients()

        # Add selected ingredients to ingredientsList
        self.ingredientsList.extend(self.ingredients)

        # Get the selected difficulty level from the dropdown
        selected_difficulty = self.dropdown_var.get()

        # Get the selected cuisine type from the dropdown
        selected_cuisine = self.cuisine_var.get()

        # Open AI completion window
        self.openai_completion_window(selected_difficulty, selected_cuisine)

    def openai_completion_window(self, difficulty, cuisine):
        completion_message = self.generate_completion_message(difficulty, cuisine)
        completion_window = tk.Toplevel(self.master)
        completion_window.title("AI Completion Result")

        completion_text = tk.Text(completion_window, height=40, width=200, wrap=tk.WORD)
        completion_text.pack(padx=40, pady=40)
        completion_text.insert(tk.END, completion_message)

    def generate_completion_message(self, difficulty, cuisine):
        time.sleep(0.1)
        list_as_string = ', '.join(str(element) for element in self.ingredientsList[:-1])
        list_as_string += f", and {self.ingredientsList[-1]}"
        ingredients = f"The list contains: {list_as_string}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a chef's assistant, skilled in creating recipes"},
                {"role": "user", "content": f"Tell me a {difficulty} {cuisine} recipe with " + ingredients}
            ]
        )
        return response['choices'][0]['message']['content']


def main():
    root = tk.Tk()
    app = IngredientsApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
