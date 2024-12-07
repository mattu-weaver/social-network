"""
Creates random networks to analyse connections.
"""
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
from .Page import Page

class FriendshipPage(Page):
    """
    Class for the friendship page.
    """
    def __init__(self):
        super().__init__('The Friendship Paradox')
        self.graph = None
        self.num_nodes = 0
        self.num_edges = 0
        self.friends_count = {}
        self.friends_of_friends = {}
        self.average_friends = 0.0
        self.avg_friends_of_friends = 0.0
        self.v = 0.0
        self.graph_model = None


    def render_page(self):
        """ 
        Renders the friendship page.
        """
        super().render_page()
        st.write('Your friends have more friends than you do (on average)')

        self.graph_model = st.sidebar.selectbox(
            "Select a graph model:",
            options=["Erdős–Rényi (Random)", "Barabási–Albert (Scale-Free)"]
        )

        self.num_nodes = st.sidebar.slider("Number of People (Nodes):",
            min_value=5, max_value=200, value=15)
        self.num_edges = st.sidebar.slider("Number of Friendships (Edges):",
            min_value=5, max_value=800, value=30)

        self.graph = self.generate_random_network()
        self.friends_count = self.calculate_friends_count()
        self.friends_of_friends = self.calculate_friends_of_friends(self.friends_count)
        self.average_friends = np.mean(list(self.friends_count.values()))
        self.avg_friends_of_friends = np.mean(list(self.friends_of_friends.values()))

        fig = self.visualise_network()
        st.pyplot(fig)

        col1, col2 = st.columns(2)

        with col1:
            self.draw_mean_trend()

        with col2:
            self.display_violin_plot()

        st.sidebar.write(f'Each friend has an average of {self.average_friends:.1f} connections.')
        st.sidebar.write(f"Each friend's friend has an average of {self.avg_friends_of_friends:.1f} connections.")


    def generate_random_network(self):
            """
            Generates a random network based on the specified graph model.

            Returns:
                nx.Graph: A generated graph with nodes labeled by their names.
            """
            # Validate inputs
            if self.num_nodes <= 0:
                raise ValueError("Number of nodes must be greater than 0.")
            if self.num_edges < 0:
                raise ValueError("Number of edges must be non-negative.")
            
            # Create the graph based on the selected model
            if self.graph_model == "Erdős–Rényi (Random)":
                graph = nx.gnm_random_graph(self.num_nodes, self.num_edges)
            elif self.graph_model == "Barabási–Albert (Scale-Free)":
                m = max(1, min(self.num_edges // self.num_nodes, self.num_nodes - 1))
                graph = nx.barabasi_albert_graph(self.num_nodes, m)
            else:
                raise ValueError(f"Unsupported graph model: {self.graph_model}")

            # Assign names to nodes
            nx.set_node_attributes(graph, {node: f"P{node}" for node in graph.nodes}, "name")
            return graph
    

    def calculate_friends_count(self):
        """
        Calculates the number of friends (neighbours) for each node in the graph.

        Returns:
            dict: A dictionary where keys are node IDs and values are the count of neighbours.
        """
        if self.graph is None:
            raise ValueError("The graph object does not exist")

        return {node: len(self.graph[node]) for node in self.graph}


    def calculate_friends_of_friends(self, friends_count):
        """
        Calculate the average number of friends each person's friends have.
        """
        if self.graph is None:
            raise ValueError("The graph object does not exist")

        friends_of_friends = {}
        for node in self.graph.nodes:
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                avg_friends_of_friends = np.mean([friends_count[neighbor] for neighbor in neighbors])
                friends_of_friends[node] = avg_friends_of_friends
            else:
                friends_of_friends[node] = 0
        return friends_of_friends
    
    def visualise_network(self):
        """
        Visualizes the network graph with nodes colored based on their degree centrality.
        Dynamically adjusts text color for better readability.
        
        Returns:
            matplotlib.figure.Figure: The generated plot figure.
        """
        if self.graph is None:
            raise ValueError("The graph object does not exist")

        # Calculate degree centrality
        degree_centrality = nx.degree_centrality(self.graph)

        # Assign node colors based on centrality
        node_color = [degree_centrality[node] for node in self.graph.nodes]
        cmap = plt.cm.viridis  # Colormap for node coloring

        # Calculate node degrees
        node_degrees = {node: len(list(self.graph.neighbors(node))) for node in self.graph.nodes}

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, node_color=node_color, with_labels=False, ax=ax,
                cmap=cmap, node_size=300, edge_color='gray')

        # Add labels with dynamic text color
        for node, (x, y) in pos.items():
            node_color_value = degree_centrality[node]
            rgb_color = cmap(node_color_value)  # Map centrality to RGB
            luminance = 0.2126 * rgb_color[0] + 0.7152 * rgb_color[1] + 0.0722 * rgb_color[2]  # Improved luminance calc
            text_color = 'white' if luminance < 0.5 else 'black'  # Dynamic text color based on luminance
            ax.text(x, y, str(node_degrees[node]), fontsize=12, ha='center', va='center', color=text_color)

        # Add color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Degree Centrality', fontsize=10, color='black')
        cbar.ax.yaxis.set_tick_params(color='black')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklines'), color='black')

        # Set title and layout
        ax.set_title(f"Social Network [{self.graph_model}]", fontsize=14, color='black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        plt.tight_layout(pad=2)

        # Add a border around the plot
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        return fig

    def display_violin_plot(self):
        """
        Displays a violin plot for friends and friends of friends distributions
        and adds horizontal lines at the median values.
        """
        # Prepare data for violin plot
        data = {
            'Friends': list(self.friends_count.values()),
            'Friends of Friends': list(self.friends_of_friends.values())
        }
        df = pd.DataFrame(data).melt(var_name='Metric', value_name='Count')

        # Calculate medians for each metric
        medians = df.groupby('Metric')['Count'].median()

        # Create violin plot
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.violinplot(x='Metric', y='Count', data=df, palette='muted', ax=ax)

        # Add horizontal lines for medians
        for metric, median in medians.items():
            ax.axhline(y=median, color='red', linestyle='--', linewidth=3,
                       label=f'{metric} Median: {median:.2f}')
            ax.text(-0.3, median,  # Adjust position to the left
                    f"{metric} Median: {median:.2f}",
                    color='black', fontsize=18, ha='left', va='center')

        # Add chart elements
        ax.set_title('Distributions of Friends and Friends of Friends connections', fontsize=20)
        ax.set_xlabel('Metric', fontsize=18)
        ax.set_ylabel('Count', fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)


    def draw_mean_trend(self):
        """
        Draws a trend chart comparing the means of friends and friends of friends.
        """
        # Calculate the mean values
        mean_friends = np.mean(list(self.friends_count.values()))
        mean_friends_of_friends = np.mean(list(self.friends_of_friends.values()))
        
        # Prepare data for the trend
        trend_data = pd.DataFrame({
            'Metric': ['Friends', 'Friends of Friends'],
            'Mean Count': [mean_friends, mean_friends_of_friends]
        })

        # Create a figure object
        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot a bar chart or line chart to show the trend
        sns.barplot(data=trend_data, x='Metric', y='Mean Count', palette='muted', ax=ax)
        
        # Add data labels for clarity
        for i, value in enumerate(trend_data['Mean Count']):
            ax.text(i, value - 1.0, f'{value:.2f}', ha='center', fontsize=18)

        # Customize the plot
        ax.set_title('Mean Comparison: Friends vs Friends of Friends', fontsize=20)
        ax.set_xlabel('Metric', fontsize=18)
        ax.set_ylabel('Mean Count', fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        # Adjust layout
        plt.tight_layout()

        # Explicitly pass the figure to Streamlit
        st.pyplot(fig)
