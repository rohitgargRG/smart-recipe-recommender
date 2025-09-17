from google.colab import drive
drive.mount('/content/drive')

!pip install gradio sentence-transformers openpyxl

import pandas as pd
import gradio as gr
import torch
import pickle
import os
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
import numpy as np

# ================== Recipe Recommender Class ==================
class RecipeRecommender:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.df = self._load_and_preprocess()
        self.embeddings_path = "embeddings.pkl"
        self._prepare_embeddings()
        self.train_df, self.test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        self._evaluate()

    def _load_and_preprocess(self):
        df = pd.read_excel(self.data_path)
        df.fillna('', inplace=True)
        df['RecipeName'] = df['RecipeName'].str.lower()
        df['Ingredients'] = df['Ingredients'].str.lower()
        df['Diet'] = df['Diet'].str.lower()
        df['Allergens'] = df['Allergens'].str.lower()
        df['TotalTimeInMins'] = pd.to_numeric(df['TotalTimeInMins'], errors='coerce').fillna(0).astype(int)
        return df

    def _prepare_embeddings(self):
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'rb') as f:
                self.df['embedding'] = pickle.load(f)
        else:
            texts = (self.df['RecipeName'] + " " + self.df['Ingredients']).tolist()
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            embeddings = [emb.cpu().numpy() for emb in embeddings]
            self.df['embedding'] = embeddings
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(self.df['embedding'], f)

    def recommend(self, query_text, candidate_df, top_k=5):
        user_embedding = self.model.encode(query_text.lower(), convert_to_tensor=True).cpu()
        candidate_embeddings = [torch.tensor(emb) for emb in candidate_df['embedding']]
        sims = util.cos_sim(user_embedding, torch.stack(candidate_embeddings))
        candidate_df = candidate_df.copy()
        candidate_df['similarity'] = sims[0]
        return candidate_df.sort_values(by='similarity', ascending=False).head(top_k)

    def recommend_ui(self, recipe_name, ingredients, total_time, allergens, diet):
        filtered_df = self.df.copy()
        print("Initial recipes count:", len(filtered_df))

        if recipe_name:
            filtered_df = filtered_df[filtered_df['RecipeName'].str.contains(recipe_name.lower(), regex=False, na=False)]
            print(f"After recipe_name filter ({recipe_name}):", len(filtered_df))

        if diet:
            filtered_df = filtered_df[filtered_df['Diet'].str.contains(diet.lower(), regex=False, na=False)]
            print(f"After diet filter ({diet}):", len(filtered_df))

        if total_time:
            try:
                max_time = int(total_time)
                filtered_df = filtered_df[filtered_df['TotalTimeInMins'] <= max_time]
                print(f"After total_time filter (<= {max_time} mins):", len(filtered_df))
            except ValueError:
                print("⚠️ Invalid total_time input ignored")

        if allergens:
            for allergen in [a.strip() for a in allergens.split(',') if a.strip()]:
                filtered_df = filtered_df[~filtered_df['Allergens'].str.contains(allergen, regex=False, na=False)]
                print(f"After allergen filter ({allergen}):", len(filtered_df))

        if filtered_df.empty:
            return "🚫 No matching recipes found. Try adjusting filters.", self.eval_results

        # Build query text
        query_text = (recipe_name + " " if recipe_name else "") + ingredients
        top_results = self.recommend(query_text, filtered_df, top_k=5)

        # Format result
        output = ""
        for i, row in top_results.iterrows():
            output += f"### 🍽️ {row['RecipeName'].title()} ({row['TotalTimeInMins']} min | {row['Diet'].title()})\n"
            output += f"- **Ingredients:** {row['Ingredients']}\n"
            if row['Allergens']:
                output += f"- ⚠️ **Allergens:** {row['Allergens']}\n"
            output += "---\n"

        return output, self.eval_results


    def _evaluate(self, top_k=5):
        mrr_list, p_at_k_list, r_at_k_list = [], [], []

        for _, row in self.test_df.iterrows():
            query = row['RecipeName'] + " " + row['Ingredients']
            true_ingredients = set([x.strip() for x in row['Ingredients'].split(',')])

            results = self.recommend(query, self.train_df, top_k=top_k)

            # Compute hits
            hits = 0
            rank = 0
            for idx, rec_row in enumerate(results.itertuples(), 1):
                pred_ingredients = set([x.strip() for x in rec_row.Ingredients.split(',')])
                if len(true_ingredients & pred_ingredients) > 0:
                    hits += 1
                    if rank == 0:
                        rank = idx  # first relevant item

            precision = hits / top_k
            recall = hits / len(true_ingredients) if true_ingredients else 0
            mrr = 1 / rank if rank > 0 else 0

            p_at_k_list.append(precision)
            r_at_k_list.append(recall)
            mrr_list.append(mrr)

        self.eval_results = f"""### 📊 Model Evaluation (on 20% test set)
- **Precision@5:** {np.mean(p_at_k_list):.2f}
- **Recall@5:** {np.mean(r_at_k_list):.2f}
- **MRR:** {np.mean(mrr_list):.2f}
"""
# ================== Launch Gradio UI ==================
def launch_app():
    recommender = RecipeRecommender("/content/drive/MyDrive/recipe recommender project/IndianFoodDataset_Allergens_finallll.xlsx")

    def interface(recipe_name, ingredients, total_time, allergens, diet):
        return recommender.recommend_ui(recipe_name, ingredients, total_time, allergens, diet)

    gr.Interface(
        fn=interface,
        inputs=[
            gr.Textbox(label="🍛 Recipe Name (optional)"),
            gr.Textbox(label="🧂 Ingredients (comma-separated)"),
            gr.Textbox(label="⏱️ Max Total Time (in minutes)", placeholder="e.g., 30"),
            gr.Textbox(label="🚫 Allergens to Avoid (comma-separated)", placeholder="e.g., dairy, nuts"),
            gr.Dropdown(["", "vegetarian", "non-vegetarian"], label="🥗 Diet Preference")
        ],
        outputs=[
            gr.Markdown(label="📋 Top Recipe Recommendations"),
            gr.Markdown(label="📈 Model Evaluation Metrics")
        ],
        title="🥘 Smart Recipe Recommender (ML + SBERT)",
        description="Enter preferences to get intelligent recipe suggestions. Powered by SBERT & Semantic Similarity.",
        theme="default"
    ).launch()

launch_app()
