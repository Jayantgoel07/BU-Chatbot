import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load the CSV file containing questions and answers
def load_faq_data(csv_file):
    df = pd.read_csv(csv_file)
    questions = df['Question'].tolist()
    answers = df['Answer'].tolist()
    return questions, answers

# Function to get the closest matching question
def find_closest_question(user_input, questions):
    # Create a TF-IDF Vectorizer to convert text to vectors
    vectorizer = TfidfVectorizer(stop_words='english')

    # Combine the user input with the list of questions (to vectorize them together)
    all_questions = questions + [user_input]

    # Transform the text into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(all_questions)

    # Compute the cosine similarity between the user input and all the questions
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Find the index of the most similar question
    closest_index = cosine_sim.argmax()
    return closest_index

# Main function to run the chatbot
def chatbot():
    # Step 2: Upload the CSV file
    csv_file = "/Users/jayantgoel/Developer/Chatbot/faq_dataset.csv"

    # Load the questions and answers from the CSV file
    questions, answers = load_faq_data(csv_file)

    print("Chatbot: Hello! How can I assist you today?")
    print("Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")

        # If user wants to exit
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        # Find the closest matching question
        closest_index = find_closest_question(user_input, questions)

        # Return the corresponding answer
        print(f"Chatbot: {answers[closest_index]}")

if __name__ == "__main__":
    chatbot()
