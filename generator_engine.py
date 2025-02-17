from crewai import Agent, Crew
from langchain.llms import OpenAI

# Initialize LLM (You can replace with another provider if needed)
llm = OpenAI(model_name="gpt-4")

# Step 1: Collect User Inputs
def get_user_input():
    print("\nWelcome to the AI-Powered Recipe Recommender!")
    
    weight = float(input("Enter your current weight (kg): "))
    body_fat = float(input("Enter your current body fat percentage: "))
    goal = input("Enter your goal (bulking/cutting/maintenance): ").strip().lower()
    groceries = input("Enter available groceries (comma-separated): ").strip().split(",")

    return {
        "weight": weight,
        "body_fat": body_fat,
        "goal": goal,
        "groceries": [item.strip().lower() for item in groceries]
    }

user_data = get_user_input()

# Step 2: Define CrewAI Agents

# User Profile Agent
user_agent = Agent(
    name="User Profile Manager",
    role="Gathers user metrics and stores preferences",
    goal="Analyze user's fitness goal and nutritional needs",
    backstory="A fitness expert AI that helps people plan meals",
    llm=llm
)

# Ingredient Matching Agent
ingredient_agent = Agent(
    name="Ingredient Matcher",
    role="Finds matching recipes based on available groceries",
    goal="Ensure recipes fit the user's available ingredients",
    backstory="A smart AI chef that adapts to user constraints",
    llm=llm
)

# Recipe Recommendation Agent
recipe_agent = Agent(
    name="Recipe Recommender",
    role="Filters and ranks recipes based on nutritional needs",
    goal="Suggest top recipes for user goals",
    backstory="A dietitian AI that helps users cook healthier",
    llm=llm
)

# Step 3: Define Task & Crew
task = f"""
The user has provided the following details:
- Weight: {user_data['weight']} kg
- Body Fat Percentage: {user_data['body_fat']}%
- Fitness Goal: {user_data['goal']}
- Available Groceries: {', '.join(user_data['groceries'])}

Your task is to recommend a recipe that:
1. Aligns with their fitness goal (bulking/cutting/maintenance).
2. Uses available groceries or suggests suitable alternatives.
3. Avoids any allergens if detected.
4. Provides an estimated calorie and macronutrient breakdown.
"""


crew = Crew(
    agents=[user_agent, ingredient_agent, recipe_agent],
    task=task
)

# Step 4: Run CrewAI Workflow
print(task)
result = crew.kickoff()
print("\nRecommended Recipe:\n", result)
