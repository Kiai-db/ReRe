import settings
import openai
import time

class IngredientsApp:
    def __init__(self):
        # Initialize ingredients list
        self.ingredientsList = settings.ingredientsList

        self.ingredients = []

    def update_ingredients(self, selections):
        # Clear the ingredients list
        self.ingredients.clear()

        # Check selected ingredients
        for ingredient, selected in selections.items():
            if selected:
                self.ingredients.append(ingredient)

    def display_ingredients_list(self, selections, difficulty, cuisine):
        # Update selected ingredients
        self.update_ingredients(selections)

        # Add selected ingredients to ingredientsList
        self.ingredientsList.extend(self.ingredients)

        # Open AI completion window
        return self.generate_completion_message(difficulty, cuisine)

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

def generator():
    app = IngredientsApp()

    # Example selections
    selections = {
        "Tomato": True,
        "Lettuce": False,
        "Cheese": True,
        "Onion": True,
        "beef": False,
        "Pepper": True,
        "chicken": False
    }

    difficulty = "easy"
    cuisine = "Italian"

    completion_message = app.display_ingredients_list(selections, difficulty, cuisine)
    print(completion_message)

if __name__ == "__main__":
    main()
