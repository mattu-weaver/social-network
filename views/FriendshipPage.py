
import pandas as pd
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""
Renders the other page.
"""
import streamlit as st
from .Page import Page

class FriendshipPage(Page):
    """
    Class for the Other page.
    """
    def __init__(self):
        super().__init__('The Friendship Paradox')
        self.graph = None
        self.num_nodes = 0
        self.num_edges = 0
        self.friends_count = 0
        self.friends_of_friends = {}
        self.average_friends = 0.0
        self.v = 0.0


    def render_page(self):
        """ 
        Renders the welcome page from which other views may be selected.
        """
        super().render_page()
        st.write('Your friends have more friends than you do (on average)')

        self.graph_model = st.sidebar.selectbox(
            "Select a graph model:",
            options=["Erdős–Rényi (Random)", "Barabási–Albert (Scale-Free)"]
        )

        self.num_nodes = st.sidebar.slider("Number of People (Nodes):", min_value=5, max_value=200, value=15)
        self.num_edges = st.sidebar.slider("Number of Friendships (Edges):", min_value=5, max_value=800, value=30)

        self.graph = self.generate_random_network()
        self.friends_count = self.calculate_friends_count()
        self.friends_of_friends = self.calculate_friends_of_friends()
        self.average_friends = np.mean(list(self.friends_count.values()))
        self.average_friends_of_friends = np.mean(list(self.friends_of_friends.values()))
        
        

        fig = self.visualise_network()
        st.pyplot(fig)

        col1, col2 = st.columns(2)

        with col1:
            fig = self.draw_mean_trend()

        with col2:
            self.display_violin_plot()


        st.sidebar.write(f'Each friend has an average of {self.average_friends:.1f} friends.')
        st.sidebar.write(f"Each friend's friend has an average of {self.average_friends_of_friends:.1f} friends.")


    def generate_random_network(self):
        if self.graph_model == "Erdős–Rényi (Random)":
            graph = nx.gnm_random_graph(self.num_nodes, self.num_edges)
        elif self.graph_model == "Barabási–Albert (Scale-Free)":
            graph = nx.barabasi_albert_graph(self.num_nodes, self.num_edges // self.num_nodes)
        for node in graph.nodes:
            graph.nodes[node]['name'] = f"P{node}"
        return graph
    
    def calculate_friends_count(self):
        return {node: len(list(self.graph.neighbors(node))) for node in self.graph.nodes}

    # Calculate average number of friends each person's friends have
    def calculate_friends_of_friends(self):
        friends_of_friends = {}
        for node in self.graph.nodes:
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                avg_friends_of_friends = np.mean([self.friends_count[neighbor] for neighbor in neighbors])
                friends_of_friends[node] = avg_friends_of_friends
            else:
                friends_of_friends[node] = 0
        return friends_of_friends
    
    def visualise_network(self):
        """
        Visualises the network graph with nodes coloured based on their degree centrality.
        """
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size explicitly

        # Define node positions
        pos = nx.spring_layout(self.graph, seed=42)

        # Calculate degree centrality
        centrality = nx.degree_centrality(self.graph)

        # Map centrality values to a colour scale
        centrality_values = list(centrality.values())
        node_colours = [plt.cm.viridis(value) for value in centrality_values]

        # Draw the graph
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            labels=nx.get_node_attributes(self.graph, 'name'),
            node_color=node_colours,
            edge_color="gray",
            node_size=500,
            font_size=8,
            ax=ax
        )

        # Add a colour bar for reference
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(centrality_values), vmax=max(centrality_values)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Degree Centrality', fontsize=10)

        # Set title and layout
        ax.set_title(f"Social Network [{self.graph_model}]", fontsize=14)
        plt.tight_layout(pad=2)  # Ensure padding consistency

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
        ax.set_title('Violin Plot: Distributions of Friends and Friends of Friends', fontsize=20)
        ax.set_xlabel('Metric', fontsize=18)
        ax.set_ylabel('Count', fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)


    def draw_kde_plot(self):
        """
        Draws a KDE plot for friends and friends of friends counts using Seaborn.
        """
        # Prepare data for Seaborn
        data = {
            'Friends': list(self.friends_count.values()),
            'Friends of Friends': list(self.friends_of_friends.values())
        }
        df = pd.DataFrame(data).melt(var_name='Metric', value_name='Count')

        # Create a figure object
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create the KDE plot
        sns.kdeplot(data=df[df['Metric'] == 'Friends'], x='Count', label='Friends', color='blue', linewidth=2, ax=ax)
        sns.kdeplot(data=df[df['Metric'] == 'Friends of Friends'], x='Count', label='Friends of Friends', color='orange', linewidth=2, ax=ax)

        # Customize plot
        ax.set_title('KDE Plot: Friends and Friends of Friends', fontsize=16)
        ax.set_xlabel('Count', fontsize=18)
        ax.set_ylabel('Density', fontsize=18)

        # Add legend
        ax.legend(title='Metric')

        # Adjust layout
        plt.tight_layout()

        # Explicitly pass the figure to Streamlit
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
