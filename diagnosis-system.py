import pandas as pd
from sentence_transformers import SentenceTransformer, util
pd.set_option('display.max_columns', None)

# Load the data
df = pd.read_csv('Diseases_Symptoms.csv')
# print(df.head())

# Initialize a Sentence Transformer model to generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for each condition's symptoms
df['Symptom_Embedding'] = df['Symptoms'].apply(lambda x: model.encode(x))

# Function to find matching condition based on input symptoms
def find_condition_by_symptoms(input_symptoms):
    # Generate embedding for the input symptoms
    input_embedding = model.encode(input_symptoms)
    
    # Calculate similarity scores with each condition
    df['Similarity'] = df['Symptom_Embedding'].apply(lambda x: util.cos_sim(input_embedding, x).item())
    
    # Find the most similar condition
    best_match = df.loc[df['Similarity'].idxmax()]
    return best_match['Name'], best_match['Treatments']

# Sample input and output
symptoms = "Fever, sore throat, and fatigue"
condition, treatments = find_condition_by_symptoms(symptoms)

print("Symptoms:", symptoms)
print("Condition:", condition)
print("Recommended Treatments:", treatments)