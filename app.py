from flask import Flask, render_template, request, jsonify
from node2vec_course_recommender import Node2VecCourseRecommender
import networkx as nx
import random

app = Flask(__name__)

# Load or train the model
recommender = Node2VecCourseRecommender()
recommender.get_or_train_model('node2vec_model.pkl')

# Generate a color map for course prefixes
def generate_color_map(courses):
    prefixes = set(course[:3] for course in courses)
    return {prefix: f"#{random.randint(0, 0xFFFFFF):06x}" for prefix in prefixes}

color_map = generate_color_map(recommender.get_all_courses())

@app.route('/')
def index():
    all_courses = recommender.get_all_courses()
    return render_template('index.html', courses=all_courses, color_map=color_map)

@app.route('/graph_data')
def graph_data():
    nodes = [{"id": node, "label": node, "color": color_map[node[:3]]} for node in recommender.graph.nodes()]
    edges = [{"from": edge[0], "to": edge[1]} for edge in recommender.graph.edges()]
    return jsonify({"nodes": nodes, "edges": edges})

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_courses = request.json['courses']
    recommendations = recommender.get_recommendations(selected_courses, num_recommendations=5)
    
    result = []
    for course, similarity in recommendations:
        course_info = recommender.get_course_info(course)
        result.append({
            'course': course,
            'similarity': f"{similarity:.4f}",
            'related_courses': course_info['related_courses']
        })
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)