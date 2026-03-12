# LA prerequisites
**Adjacency matrix raised to power** 
```python
nx.adjacency_matrix( G: networkx.Graph ) -> scipy.sparse.spmatrix
```
$\boxed{A^n_{ij} = \# \text{ n length walks } v_i \rightarrow v_j}$
Proof : $A^{k+1}_{ij} = \sum_{l}{A^{k}_{il}A_{lj}}$
`Katz index` : Weighted sum over all path lengths from $v_i \rightarrow v_j$.
$S_{ij} = \sum_{l=1}^{\infty}{\beta^l A^l_{ij}}$.
$$\mathbb{S} = \sum_{l=1}^{\infty} \beta^l \mathbb{A}^l = (\mathbb{I} - \beta \mathbb{A})^{-1} - \mathbb{I}$$
**Laplacian matrix** $L = D - A$, $D$ being degree matrix ($D = \text{diag}(A\mathbb{1})$, with $\mathbb{1}$ being column vector with all ones, equivalently $D_{ij} = c_d(i)\delta_{ij}$ with $\delta$ being Kronecker delta) and $A$ is adjacency matrix.
Laplacian is `symmetric` (have real orthogonal eigenvectors) and `positive semi-definite`.
$x^\top L x = \sum_{(i,j) \in E} (x_i - x_j)^2 \geq 0$
```python
nx.laplacian_matrix( G: networkx.Graph ) -> scipy.sparse.spmatrix
```
**Spectrum** $$\text{spec}(G) = \begin{bmatrix} \lambda_1 & \cdots & \lambda_t \\
m_1 & \cdots & m_t
\end{bmatrix}$$ where $m_i$ is multiplicity of eigenvalue $\lambda_i$. Spectrum of a complete graph is 
$$\text{spec}(K_n) = \begin{bmatrix} n-1 & -1 \\
1 & n-1
\end{bmatrix}$$
**Eigenvector corresponding to largest eigenvalue**
Consider $\lambda_1 \gt \lambda_2 \gt \cdots \lambda_n$ and corresponding eigenvectors as $v_1, v_2, \cdots, v_n$.
We can express our initial guess $c_0$ as a linear combination of all the eigenvectors of $A$ ($v_1, v_2, \dots, v_n$):

$$c_0 = \alpha_1 v_1 + \alpha_2 v_2 + \dots + \alpha_n v_n$$

When we multiply by $A^t$, the linearity of matrix multiplication gives us:

$$A^t c_0 = \alpha_1 \lambda_1^t v_1 + \alpha_2 \lambda_2^t v_2 + \dots + \alpha_n \lambda_n^t v_n$$

Now, factor out the dominant eigenvalue $\lambda_1^t$:

$$c_ t = A^t c_0 = \lambda_1^t \left[ \alpha_1 v_1 + \alpha_2 \left( \frac{\lambda_2}{\lambda_1} \right)^t v_2 + \dots + \alpha_n \left( \frac{\lambda_n}{\lambda_1} \right)^t v_n \right] \approx \alpha_1 \lambda_1^t v_1$$ So, $c_t || v_1$, i.e., $$\boxed{\left( \lim_{t \rightarrow \infty} A^t c_0 \right) \, || \, v_1}$$

# Traditional Graph ML
After extracting features from the graph using the below mentioned techniques, we can train appropriate ML model (like Random forest / SVM / Neural Nets) over the extracted features.
## G(V,E) -> **node** level features -> node level prediction
Given $G(V, E)$ learn the function $f : V \rightarrow \mathbb{R}$.
### Degree centrality
$c_d(i) = D_{ii}$
### Eigenvector centrality
Consider centrality of the neighbors $c_e(i) = \frac{1}{\lambda} \sum_{j \in N(i)} c_e(j)$  i.e., $c_e(i) = \frac{1}{\lambda}\sum_j{A_{ij}c_e(j)}$ or $Ac_e = \lambda c_e$.
The largest eigenvalue $\lambda_{\max}$ is always positive and unique (Perron-Frobenius theorem). Corresponding eigenvector $c_{\max}$ is used for centrality. So, $\boxed{c_e = \lim_{t \rightarrow \infty} A^t c_0}$, i.e., start with an initial guess $c_0$ and stop when $||c_{t+1} - c_t|| \lt \epsilon$, where $c_{t+1} = Ac_t$ (or we can normalize for better numerical results as $c_{t+1} = \frac{Ac_t}{||Ac_t||}$).
#### PageRank (Brin-Page 1998)
Rank webpages by the number of in-links and the recursive importance of the referencing webpage.
$$r_j^{(t+1)} = \sum_{i \rightarrow j} \beta \frac{r_i^{(t)}}{d_i} + (1 - \beta) \frac{1}{N}$$, $d_i$ being out-degree of node $i$. Solve by power iteration as explained before $r^{(t+1)} = G r^{(t)}$, $G = \beta M + (1 - \beta) \left[ \frac{1}{N} \right]_{N \times N}$ , stop when $||r^{(t+1)} - r^t||_1 < \epsilon$. Make sure the columns of stochastic matrix $M$ sum to 1 (to tackle *dead-ends*), using $\beta$ to tackle *spider-traps* : with prob $\beta$ follow a link at random and with prob $1 - \beta$ it jumps to some random page.
#### Personalized PageRank (PPR)
Teleport probability is not uniform.
#### Random walks with restarts
Teleport back to starting node $S = \{  Q \}$.
### Katz centrality
Centrality of the neighbors and a small constant
$c_k = \alpha A c_k + \beta$, or $c_k = (I - \alpha A)^{-1} \beta$.
### Betweenness centrality
A node is important if it lies on many shortest paths between other nodes
$$c_b(v) = \sum_{s \neq v \neq t}\frac{\#(\text{shortest paths between s and t that contain v})}{\#(\text{shortest paths between s and t})}$$
### Clustering coefficient
Measures how connected v's neighboring nodes are
$$c_c(v) = \frac{\#(\text{edges among neighboring nodes})}{\binom{c_d(v)}{2}}$$
### Graphlet degree vector (GDV)
Count vector of graphlets rooted at a given node $\rightarrow f_G$.
### Usage example
```python
import networkx as nx
G = nx.karate_club_graph()
club_labels = nx.get_node_attributes(G, 'club')
degree = nx.degree(G)

# centrality metrics
degree_cent = nx.degree_centrality(G)
eigen_cent = nx.eigenvector_centrality(G, max_iter=100) # Uses the power iteration method
katz_cent = nx.katz_centrality(G, alpha=0.1, beta=1.0) # alpha is the attenuation factor; beta is the weight
between_cent = nx.betweenness_centrality(G)
clust_coeff = nx.clustering(G)
cliques = nx.number_of_cliques(G)
pagerank = nx.pagerank(G)
hub, auth = nx.hits(G)

# data preprocessing for putting into ML model
data = [list(x.values()) for x in (degree_cent, eigen_cent, katz_cent, pagerank)]
data.append(list(club_labels.values()))
data_arr = np.array(data)
df = pd.DataFrame(data_arr.T, columns=['degree_cent', 'eigen_cent', 'katz_cent', 'pagerank', 'club_labels'])
df.club_labels = df.club_labels.apply(lambda x : 0 if x == 'Mr. Hi' else 1)
X = df.drop(columns=['club_labels'])
y = df.club_labels
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
```
## G(V,E) -> **edge** level features -> edge level prediction
Given $G(V, E)$ learn $f : V^2 \rightarrow \mathbb{R}$ , here $E \subseteq V^2$.
Possible tasks : 
- remove a set of random edges and aim to predict them
- Given graph on edges up to time $t_0$, produce a ranked list L of edges not in $G[t_0, t_0']$ that are predicted to appear in $G[t_1, t_1']$.
### Distance-based features
$S_{ij} = \text{shortest distance b/w } v_i \text{ and } v_j$. 
Limitation this does not capture the degree of neighborhood overlap.
### Local Neighborhood overlap
Captures # neighboring nodes shared between two nodes $v_i$ and $v_j$.
**Jaccard Index:**

$$S_{ij} = \frac{|\mathcal{N}(v_i) \cap \mathcal{N}(v_j)|}{|\mathcal{N}(v_i) \cup \mathcal{N}(v_j)|}$$
**Sorensen Index:**

$$S_{ij} = \frac{2|\mathcal{N}(v_i) \cap \mathcal{N}(v_j)|}{c_d(i) + c_d(j)}$$
**Salton Index (Cosine Similarity):**

$$S_{ij} = \frac{|\mathcal{N}(v_i) \cap \mathcal{N}(v_j)|}{\sqrt{c_d(i)c_d(j)}}$$
**Resource Allocation (RA) Index:**

$$S_{ij} = \sum_{u \in \mathcal{N}(v_i) \cap \mathcal{N}(v_j)} \frac{1}{c_d(u)}$$
**Adamic-Adar (AA) Index:**

$$S_{ij} = \sum_{u \in \mathcal{N}(v_i) \cap \mathcal{N}(v_j)} \frac{1}{\log(c_d(u))}$$
Limitation of local neighborhood overlap is $S_{ij} = 0$ when $\mathcal{N}(v_i) \cap \mathcal{N}(v_j) = \phi$.
### Global neighborhood overlap
**Katz index** -> counts the number of paths of all lengths between a given pair of nodes
$$\mathbb{S} = \sum_{l=1}^{\infty} \beta^l \mathbb{A}^l = (\mathbb{I} - \beta \mathbb{A})^{-1} - \mathbb{I}$$
**Leicht, Holme and Newman (LHN) Similarity** 
$S_{ij} = \sum_{l=0}^{\infty} \beta^l \frac{A_{ij}^l}{\mathbb{E}[A_{ij}^l]} + \delta_{ij}$ Simplifying we get
$$S = 2\lambda_1 D^{-1} (I - \frac{\beta A}{\lambda_1})^{-1} D^{-1}$$
## G(V,E) -> **graph** level features -> graph level prediction
Use case: When we need to classify whole graphs (e.g., "Is this molecule toxic or safe?") but don't have enough data to train a deep learning model.

Design Kernel instead of feature vectors. Kernel matrix K is positive semidefinite and there exists a feature representation $\phi$ such that $K(G, G') = \phi(G)^\top \phi(G')$. Once the kernel is defined, kernel SVM can be used to make the predictions.
### Graphlet kernel
$h_G = \frac{f_G}{f_G^\top \mathbb{1}}$ (normalized GDV), using this define $K(G, G') = h_G^\top h_{G'}$. However counting graphlets is NP-hard. 
### Weisfeiler-Lehman kernel
Technique called color refinement. $c^{(k+1)}(v) = \text{HASH}(c^{(k)}(v), \{ c^{(k)}(u)\}_{u \in N(v)})$.
### Random-walk kernel

### Shortest-path graph kernel

## Graph clustering (community detection) -> Unsupervised

# Node embedding
$f : V \rightarrow \mathbb{R}^d$ Map nodes into an embedding space, so that similarity in embeddings between nodes indicates their similarity in the network, i.e., $\text{similarity}(u, v)_{\text{original graph}} \approx z_u^\top z_v$.
So we have two components : **Encoder** ($z$) and **similarity** function. [[#A Comprehensive Survey of **Graph Embedding** Problems, Techniques, and Applications]]
## Shallow encoding
$\text{ENC}(v) = z_v = Z v$, $Z \in \mathbb{R}^{d \times |V|}, v \in \mathbb{1}^{|V|}$ (indicator vector with all zeros except a one in column indicating node v). Disadvantage: too many parameters in $Z$, not scalable.
Use random walks to define similarity, i.e., $z_u^\top z_v \approx \text{prob. that u and v co-occur on a random walk over the graph}$. The objective is 
$$\mathcal{L} = - \sum_{u \in V} \sum_{v \in N_R(u)} \log(P(v | z_u))$$ where $N_R(u)$ is the probability of node $u$ by the random walk strategy $R$ and $P(v | z_u) = \alpha \exp(z_u^\top z_v)$ (softmax), $\alpha$ being normalizing constant.
*Negative sampling approximation*: $$\log(P(v|z_u)) \approx \log (\sigma(z_u^\top z_v)) - \sum_{i: n_i \backsim P_V }\log (\sigma(z_u^\top z_{n_i}))$$

### DeepWalk
### node2vec
Develop biased 2nd order random walk $R$ to generate network neighborhood $N_R(u)$ of node $u$.
- compute random walk probabilities
- simulate $r$ random walks of length $l$ starting from each node $u$.
- optimize the objective using SGD.

# Graph embedding
$f : G \rightarrow \mathbb{R}^d$ embed a subgraph or the entire graph into embedding space.
### Sum over node embeddings
$z_G = \sum_{v \in G} z_v$.
### Using "virtual node"
$z_G = \text{node embedding of virtual node}$.
### Anonymous walk embedding
States in anonymous walks correspond to the index of the first time we visited the node in a random walk. 
- Simulate anonymous walks $w_i$ of $l$ steps and record their counts. Represent the graph as a probability distribution over these walks.
or 
- Embed anonymous walks, concatenate their embeddings to get a graph embedding.



# GNN
Use when we have large complex data. GNN learns features and tasks simultaneously using message passing. Every node "talks" to its neighbors to update its own representation. If node A is connected to node B, node A's feature vector $h_A$ is updated using $h_B$. 

# Papers
## A Comprehensive Survey of **Graph Embedding**: Problems, Techniques, and Applications 
HongYun Cai , Vincent W. Zheng , and Kevin Chen-Chuan Chang, 2017, IEEE
### 1. Methods of Graph Embeddings
Graph embedding techniques map graph data into low-dimensional spaces while preserving structural and property information. The main methods include:
- **Matrix Factorization:** Represents graph characteristics (like node proximity or Laplacian matrices) as a matrix and factorizes it to obtain low-dimensional embeddings. Use when the adjacency matrix $W \notin [0,1]^{m \times m}$, i.e., edges have non-binary weights. 
	- *Graph laplacian eigenmap*
	$\{ x_i \}_{i=1}^m \subset \mathbb{R}^n$  and its embeddings $\{ z_i \}_{i=1}^m \subset \mathbb{R}^d$ with $d << n$ . Consider $z_i = A x_i$, $$A^* = \text{argmin} \sum_{i,j} ||Ax_i - Ax_j||^2 W_{ij} = \text{argmin } 2 \text{ tr}(A(XLX^\top)A^\top)$$ subject to $A(XDX^\top)A^\top = I$.  Or, simplifies to $\text{argmin } tr(A(XWX^\top)A^\top)$ subject to $A(XDX^\top)A^\top = I$. $$A^* = \begin{bmatrix} v_1^\top \\ v_2^\top \\ . \\ . \\ v_d^\top \end{bmatrix}$$ with $v_1, v_2, \cdots, v_n$ being eigenvectors corresponding to $\lambda_1 \leq \lambda_2 \leq \cdots \lambda_n$ of this generalized eigenvalue equation: $(XWX^\top)v_i = \lambda_i (XDX^\top)v_i$.
	- *Node proximity*
	$W \approx Y (Y^c)^\top$. Find rank $d$ decomposition of $W$ (use SVD).  
	```python
# Solve the generalized eigenvalue problem M1 * v = lambda * M2 * v 
# scipy's eigh returns eigenvalues in ascending order (lambda_1 <= lambda_2 <= ...) 
eigenvalues, eigenvectors = scipy.linalg.eigh(M1, M2)
A = eigenvectors[:, :d].T # Shape: (d, n)
embeddings = A @ X # (d, m)

# method 2
U, Sigma, VT = scipy.sparse.linalg.svds(W, k=d) # svds returns singular values in ascending order; reverse them to get the largest first 
idx = np.argsort(Sigma)[::-1] 
U = U[:, idx] 
Sigma = Sigma[idx]
embeddings = U @ np.diag(np.sqrt(Sigma)) # (m, d)
	```
- **Deep Learning:** Utilizes neural networks to encode graph structures.
	- random walks (to sample paths and learn context like *DeepWalk*, *node2vec*, *metapath2vec*)
	```python
import networkx as nx
from karateclub import Node2Vec, DeepWalk

G = nx.karate_club_graph()

# 1. DeepWalk
dw_model = DeepWalk(walk_number=10, walk_length=80, dimensions=64)
dw_model.fit(G)
deepwalk_embeddings = dw_model.get_embedding()

# 2. Node2Vec
n2v_model = Node2Vec(walk_number=10, walk_length=80, p=1.0, q=1.0, dimensions=64)
n2v_model.fit(G)
node2vec_embeddings = n2v_model.get_embedding()

print("DeepWalk embedding shape:", deepwalk_embeddings.shape)
	```
	- without random walks (such as *SDNE* (autoencoder), *GCN*, *GNN*)
	```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Load standard Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Define a 2-layer Graph Convolutional Network (GCN)
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First GCN layer + ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        # Second GCN layer to output classes/embeddings
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN()
out = model(data)
print("Output shape (Nodes x Classes):", out.shape)
	```
- **Edge Reconstruction:** Learns embeddings by optimizing objective functions that preserve edge connections.
	- maximizing the probability of reconstructing an edge
	- minimizing distance-based loss
	- margin-based ranking losses (Knowledge graph)
	```python
from pykeen.pipeline import pipeline

# Run a pipeline to train TransE (minimizing distance/margin-based loss) 
# on the standard Nations dataset
result = pipeline(
    dataset='Nations',
    model='TransE',
    training_kwargs=dict(num_epochs=5),  # Keep epochs low for example purposes
    random_seed=1234,
)

# Extract entity and relation embeddings
entity_embeddings = result.model.entity_representations[0](indices=None)
print("Entity embeddings tensor shape:", entity_embeddings.shape)
	```
- **Graph Kernels:** Decomposes graphs into atomic substructures (such as graphlets, subtree patterns, or random walks) and uses kernel functions to measure the similarity between these substructures.
	- Graphlet
	- Weisfeiler-Lehman subtree
	- Random walks
- **Generative Models:** Embeds graphs into latent spaces by learning the underlying data distribution to reconstruct or generate graph structures.
### 2. Matching Methods to Graphs
Different input graphs carry different types of information, dictating the choice of embedding method:
- **Homogeneous Graphs:** These contain only one type of node and edge, focusing purely on structural connections. *Matrix factorization* and *Deep Learning with random walks* are heavily applied here to preserve topological proximity.
- **Heterogeneous Graphs:** These graphs (such as **Knowledge Graphs**) contain multiple types of nodes and edges. *Edge Reconstruction* techniques (like minimizing distance-based loss) or Deep Learning using *metapath-guided random walks* are typically applied to capture the rich semantic relationships.
- **Graphs with Auxiliary Information:** These include node labels, attributes, or edge features. Deep Learning *without random walks* is most commonly applied to smoothly integrate both the graph topology and the rich attribute information into the embedding.
- **Graphs Constructed from Non-relational Data:** Data like images or text can be transformed into graphs and embedded, often using *Matrix Factorization* or *Deep Learning*.
### 3. Matching Methods to Problems
The output of graph embedding is task-driven and aligns with specific analytical problems:
- **Node Embedding:** Maps individual nodes to vectors. It is widely applied for problems like node classification, clustering, and node recommendation.
- **Edge Embedding:** Maps pairs of nodes (edges) to vectors. It is primarily applied for link prediction tasks and knowledge graph completion.
- **Hybrid Embedding:** Combines different graph components (like nodes and edges). It is used for semantic proximity search and extracting cohesive subgraphs.
- **Whole-Graph Embedding:** Represents an entire graph as a single vector. It is essential for graph classification and similarity problems, such as comparing the properties of proteins or molecular structures.
### 4. Comparing Multiple Methods for the Same Problem
When multiple methods can be applied to a specific problem, their strengths and weaknesses govern the choice:
- **Node/Edge Embedding (Matrix Factorization vs. Deep Learning):** For tasks like node classification on large networks, *Matrix Factorization effectively preserves basic graph properties but often struggles with non-linear patterns and scalability*. Conversely, Deep Learning with random walks easily scales to massive graphs and captures rich neighborhood contexts. If the graph contains node attributes, Deep Learning without random walks outperforms the others because it dynamically fuses both structural and attribute data.  
- **Whole-Graph Embedding (Graph Kernels vs. Deep Learning):** For graph classification problems, Graph Kernels offer a rigorous mathematical approach by explicitly comparing substructures. However, they rely on manually crafted heuristic rules and suffer from high computational complexity. Deep Learning approaches overcome this by automatically learning graph representations in a data-driven manner, providing better scalability and often yielding superior predictive performance on complex graphs.

## A survey of geometric graph neural networks: data structures, models and applications 
Jiaqi HAN, Jiacheng CEN
