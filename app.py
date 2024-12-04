from sentence_transformers import SentenceTransformer, util

# Load the MiniLM model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define an array of sentences
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast dark fox leaps across a sleepy canine.",
    "The weather is sunny and warm today.",
    "The forecast predicts a bright and hot day."
]

# Create embeddings for each sentence
embeddings = model.encode(sentences, convert_to_tensor=True)

# Calculate pairwise cosine similarity
similarity_matrix = util.cos_sim(embeddings, embeddings)

# Display the similarity scores
print("Sentence Similarity Scores:")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(f"Similarity between \"{sentences[i]}\" and \"{sentences[j]}\": {similarity_matrix[i][j]:.4f}")
