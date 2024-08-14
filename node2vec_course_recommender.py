import networkx as nx
from node2vec import Node2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from course_data import get_all_classes
import pickle
import os

class Node2VecCourseRecommender:
    def __init__(self, dimensions=64, walk_length=30, num_walks=200, workers=4):
        self.graph = self.build_graph()
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.model = None
        self.embeddings = None

    def build_graph(self):
        G = nx.Graph()  # Changed to undirected graph
        for course1, course2 in get_all_classes():
            G.add_edge(course1, course2)
        return G

    def train(self):
        print("Training Node2Vec model...")
        node2vec = Node2Vec(self.graph, dimensions=self.dimensions, walk_length=self.walk_length, 
                            num_walks=self.num_walks, workers=self.workers)
        self.model = node2vec.fit(window=10, min_count=1, batch_words=4)
        self.embeddings = {node: self.model.wv[node] for node in self.graph.nodes()}
        print("Training complete.")

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.model, self.embeddings, self.graph), f)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.model, self.embeddings, self.graph = pickle.load(f)
        print(f"Model loaded from {filename}")

    def get_or_train_model(self, filename):
        if os.path.exists(filename):
            self.load_model(filename)
        else:
            self.train()
            self.save_model(filename)

    def get_recommendations(self, seed_courses, num_recommendations=5):
        if self.model is None:
            raise ValueError("Model hasn't been trained or loaded.")

        seed_embeddings = np.array([self.embeddings[course] for course in seed_courses])
        all_courses = list(self.embeddings.keys())
        all_embeddings = np.array([self.embeddings[course] for course in all_courses])
        similarities = cosine_similarity(seed_embeddings.mean(axis=0).reshape(1, -1), all_embeddings).flatten()
        
        course_similarities = list(zip(all_courses, similarities))
        course_similarities.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for course, similarity in course_similarities:
            if course not in seed_courses:
                recommendations.append((course, similarity))
                if len(recommendations) == num_recommendations:
                    break
        
        return recommendations

    def get_course_info(self, course):
        return {
            "related_courses": list(self.graph.neighbors(course)),
        }

    def get_all_courses(self):
        return list(self.graph.nodes())

if __name__ == "__main__":
    recommender = Node2VecCourseRecommender()
    recommender.get_or_train_model('node2vec_model.pkl')
    print("Model ready for use.")