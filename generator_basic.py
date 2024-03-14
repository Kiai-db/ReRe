import settings
import openai
import time

def recipe_generator(ingredients_ripeness, difficulty, cuisine):
    class IngredientsApp:
        def __init__(self):
            # Initialize ingredients list
            self.ingredientsList = settings.ingredientsList
            self.ingredients = []

        def update_ingredients(self, ingredients_ripeness):
            # Clear the ingredients list
            self.ingredients.clear()

            # Check selected ingredients
            for ingredient, ripeness in ingredients_ripeness:
                ripeness_str = self.convert_ripeness_to_string(ripeness)
                if ripeness > 0:  # Assuming ripeness is greater than 0 indicates selection
                    self.ingredients.append((ingredient, ripeness_str))

        def display_ingredients_list(self, ingredients_ripeness, difficulty, cuisine):
            # Update selected ingredients
            self.update_ingredients(ingredients_ripeness)

            # Add selected ingredients to ingredientsList
            self.ingredientsList.extend(self.ingredients)

            # Open AI completion
            return self.generate_completion_message(difficulty, cuisine)

        def generate_completion_message(self, difficulty, cuisine):
            time.sleep(0.1)
            list_as_string = ', '.join(str(element[0]) + " (" + str(element[1]) + ")" for element in self.ingredientsList[:-1])
            list_as_string += f", and {self.ingredientsList[-1][0]} ({self.ingredientsList[-1][1]})"
            ingredients = f"The list contains: {list_as_string}"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a chef's assistant, skilled in creating recipes"},
                    {"role": "user", "content": f"Tell me a {difficulty} {cuisine} recipe with " + ingredients + "but exclude any rotten ingredients and try to prioritise using overripe ingredients where possible without comprimising the integrity of the recipe"}
                ]
            )
            return response['choices'][0]['message']['content']

        def convert_ripeness_to_string(self, ripeness):
            if ripeness == 0:
                return "overripe"
            elif ripeness == 1:
                return "ripe"
            elif ripeness == 2:
                return "rotten"
            elif ripeness == 3:
                return "unripe"
            else:
                return ""

    app = IngredientsApp()
    completion_message = app.display_ingredients_list(ingredients_ripeness, difficulty, cuisine)
    return completion_message

def main():
    # Example tuple of ingredients and ripeness values
    ingredients_ripeness = [
        ("Tomato", 0),   # Not selected
        ("Lettuce", 4),  # Selected
        ("Cheese", 2),   # Selected
        ("Onion", 2),    # Not selected
        ("Beef", 1),     # Selected
        ("Pepper", 3),   # Not selected
        ("Chicken", 2)   # Selected
    ]

    difficulty = "easy"
    cuisine = "chinese"

    completion_message = recipe_generator(ingredients_ripeness, difficulty, cuisine)
    print(completion_message)

if __name__ == "__main__":
    main()
