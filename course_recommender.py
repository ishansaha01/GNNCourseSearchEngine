import networkx as nx
from collections import Counter
from course_data import get_all_classes

class CourseRecommender:
    def __init__(self):
        self.graph = self.build_graph()
        self.pagerank = nx.pagerank(self.graph)
    
    def build_graph(self):
        G = nx.DiGraph()
        G.add_edges_from(get_all_classes())
        return G
    
    def get_recommendations(self, seed_courses, num_recommendations=5):
        # Compute personalized PageRank
        personalization = {node: 1 for node in seed_courses}
        personalized_pagerank = nx.pagerank(self.graph, personalization=personalization)
        
        # Combine global and personalized PageRank scores
        combined_scores = {node: 0.1 * self.pagerank.get(node, 0) + 0.9 * personalized_pagerank.get(node, 0) 
                           for node in self.graph.nodes()}
        
        # Sort and filter recommendations
        recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        filtered_recommendations = [(course, score) for course, score in recommendations 
                                    if course not in seed_courses and self.is_available(course, seed_courses)]
        
        return filtered_recommendations[:num_recommendations]
    
    def is_available(self, course, completed_courses):
        prerequisites = set(self.graph.predecessors(course))
        return prerequisites.issubset(set(completed_courses))

    def get_course_info(self, course):
        return {
            "prerequisites": list(self.graph.predecessors(course)),
            "leads_to": list(self.graph.successors(course)),
            "pagerank_score": self.pagerank.get(course, 0)
        }

    def analyze_graph(self):
        print(f"Total number of courses: {self.graph.number_of_nodes()}")
        print(f"Total number of prerequisites relationships: {self.graph.number_of_edges()}")
        print(f"Number of connected components: {nx.number_weakly_connected_components(self.graph)}")

        degree_sequence = [d for n, d in self.graph.degree()]
        print(f"Average number of connections per course: {sum(degree_sequence) / len(degree_sequence):.2f}")
        print(f"Course with highest number of prerequisites: {max(self.graph.in_degree(), key=lambda x: x[1])}")
        print(f"Course that is a prerequisite for the most courses: {max(self.graph.out_degree(), key=lambda x: x[1])}")

if __name__ == "__main__":
    # Initialize the recommender
    recommender = CourseRecommender()

    # Analyze the graph
    recommender.analyze_graph()

    # Example usage
    seed_courses = ["COS217", "ECE206", "COS306"]
    recommendations = recommender.get_recommendations(seed_courses)

    print("\nTop recommended courses:")
    if recommendations:
        for course, score in recommendations:
            print(f"{course}: {score:.4f}")
            try:
                course_info = recommender.get_course_info(course)
                print(f"  Prerequisites: {', '.join(course_info['prerequisites'])}")
                print(f"  Leads to: {', '.join(course_info['leads_to'])}")
                print(f"  PageRank score: {course_info['pagerank_score']:.4f}")
            except Exception as e:
                print(f"  Error retrieving course info: {e}")
            print()
    else:
        print("No recommendations found for the given seed courses.")