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




# Geometric graphs
A geometric graph is defined as $\vec{\mathcal{G}}(A, H, \vec{X})$, where $A \in [0,1]^{N \times N}$ is the adjacency matrix, $H \in \mathbb{R}^{N \times C_h}$ is the node feature matrix, and $X \in \mathbb{R}^{N \times 3}$ are the 3D coordinates of all nodes.
In a standard graph, the data lives purely in a **topological space**. The primary information is relational (who is connected to whom), and the model only needs to care about the graph's connectivity structure and abstract node/edge features.

In a geometric graph (such as a 3D molecule, a point cloud, or an N-body physics system), nodes also possess **spatial coordinates** in a Euclidean space (e.g., $\mathbb{R}^3$). The paradigm shift is that the network must now respect the physical laws and symmetries of that space. It is no longer enough to just know that two nodes are connected; the model must understand their spatial relationship while strictly adhering to the laws of physics—specifically, Euclidean symmetries $E(n)$ or $SE(n)$.

Here is how these symmetries are encoded and how the embedding architectures adapt to this shift.
### 1. Encoding Symmetries: Permutation, Translation, and Orthogonal Transformations
When building models for geometric graphs, the architecture must encode specific symmetries as either **invariant** (the output does not change when the input transforms) or **equivariant** (the output transforms in the exact same mathematical way as the input).
- **Permutation:** Changing the order of the nodes in the input matrix should not change the graph's output.
    - _How it is encoded:_ Just like in standard GNNs, permutation invariance is achieved by using symmetric aggregation functions (like `sum`, `mean`, or `max`) when pooling messages from neighboring nodes.
- **Translation:** Shifting the entire graph in space (e.g., moving a molecule 5 units to the right) should not change its scalar properties (like energy) or its internal relative structure.
    - _How it is encoded:_ Instead of feeding absolute coordinates $x_i$ into the neural network, the architecture uses **relative displacement vectors** ($x_i - x_j$). If the whole system is translated by a vector $c$, the shift cancels out: $(x_i + c) - (x_j + c) = x_i - x_j$.
- **Orthogonal Transformation (Rotation and Reflection):** Rotating a geometric graph in 3D space should not alter invariant properties (like the type of molecule) but must strictly rotate equivariant properties (like atomic forces or velocity vectors) by the exact same rotation matrix.
    - _How it is encoded (Invariant):_ To make the model blindly invariant to rotations, the architecture computes the $L_2$ norm (Euclidean distance) of the relative vectors: $||x_i - x_j||^2$. Distances do not change when an object is rotated.
    - _How it is encoded (Equivariant):_ To make the model equivariant, the architecture performs scalar multiplications on the relative displacement vectors. Since scalars are invariant to rotation, multiplying a relative vector by a learned scalar preserves the vector's rotational direction.
### 2. How the Embedding Architecture Differs
Standard GNNs (like GCNs or GraphSAGE) only update hidden feature vectors $h_i$ (representing semantic information like atom type or charge). Geometric GNNs must handle both the hidden features $h_i$ and the geometric coordinates $x_i$.
This leads to two primary types of geometric architectures:
#### A. Invariant Graph Neural Networks (e.g., SchNet)
These architectures only output invariant embeddings. They discard the directional information of the coordinates and rely purely on distances.
- **Message Passing:** The message from node $j$ to node $i$ is a function of their hidden states and the **distance** between them.
    $$m_{ij} = \phi_m(h_i, h_j, ||x_i - x_j||)$$
- **Update:** The hidden state is updated based on the aggregated messages. The coordinates $x_i$ are never updated.
    $$h_i' = \phi_h(h_i, \sum_{j} m_{ij})$$
- _Limitation:_ Because they only use distances, they cannot distinguish between certain chiral structures (mirror images) or complex angular geometries.
#### B. Equivariant Graph Neural Networks (e.g., EGNN)
These architectures maintain two parallel streams of embeddings: one for invariant node features ($h_i$) and one for equivariant coordinate vectors ($x_i$). As the network gets deeper, it continuously updates both the features and the physical structure.
- **Message Passing:** The message is computed using the hidden features and the squared distance.
    $$m_{ij} = \phi_e(h_i, h_j, ||x_i - x_j||^2)$$
- **Coordinate Update (Equivariant Stream):** The model predicts a force or shift for the coordinates by applying a learned scalar function $\phi_x$ to the invariant message, and multiplying it by the direction vector $(x_i - x_j)$.
    $$x_i' = x_i + \sum_{j} (x_i - x_j) \phi_x(m_{ij})$$
    Because $(x_i - x_j)$ rotates equivariantly and $\phi_x(m_{ij})$ is a rotation-invariant scalar, the entire update is mathematically guaranteed to be $E(n)$ equivariant.
- **Feature Update (Invariant Stream):** The hidden features are updated just like a standard GNN, but informed by the geometric messages.
    $$h_i' = \phi_h(h_i, \sum_{j} m_{ij})$$
# Papers
## A Comprehensive Survey of **Graph Embedding**: Problems, Techniques, and Applications 
HongYun Cai , Vincent W. Zheng , and Kevin Chen-Chuan Chang, 2017, IEEE
### 1. Methods of Graph Embeddings
Graph embedding techniques map graph data into low-dimensional spaces while preserving structural and property information. The main methods include:
- **Matrix Factorization:** Use when the adjacency matrix $W \notin [0,1]^{m \times m}$, i.e., edges have non-binary weights. Example usage: graph constructed from non-relational data by encoding the pairwise relations between instances.
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
# 1. DeepWalk
class DeepWalk:
    def __init__(self, graph, embedding_dim=128, walk_length=80, num_walks=10, window_size=5):
        """
        graph: dict of adjacency lists {node: [neighbors]}
        """
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.embeddings = None
    
    def random_walk(self, start_node):
        walk = [start_node]
        for _ in range(self.walk_length - 1):
            current = walk[-1]
            neighbors = self.graph.get(current, [])
            if len(neighbors) == 0:
                break
            walk.append(random.choice(neighbors))
        return walk
    
    def generate_walks(self):
        walks = []
        nodes = list(self.graph.keys())
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk(node))
        return walks
    
    def train(self, lr=0.025, epochs=1):
        walks = self.generate_walks()
        num_nodes = len(self.graph)
        
        # Initialize embeddings
        self.embeddings = np.random.randn(num_nodes, self.embedding_dim) * 0.01
        
        # Simple Skip-gram implementation
        for epoch in range(epochs):
            for walk in walks:
                for i, node in enumerate(walk):
                    # Get context window
                    start = max(0, i - self.window_size)
                    end = min(len(walk), i + self.window_size + 1)
                    
                    for j in range(start, end):
                        if i != j:
                            context_node = walk[j]
                            # Update embeddings (simplified)
                            self.embeddings[node] += lr * self.embeddings[context_node]
        
        return self.embeddings


# 2. node2vec
class Node2Vec:
    def __init__(self, graph, embedding_dim=128, walk_length=80, num_walks=10, 
                 p=1.0, q=1.0, window_size=5):
        """
        graph: dict of adjacency lists {node: [neighbors]}
        p: return parameter (controls likelihood of returning to previous node)
        q: in-out parameter (controls BFS vs DFS)
        """
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window_size = window_size
        self.embeddings = None
    
    def get_alias_nodes(self, node):
        """Precompute transition probabilities for neighbors"""
        neighbors = self.graph.get(node, [])
        if len(neighbors) == 0:
            return [], []
        
        unnormalized_probs = [1.0] * len(neighbors)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [p / norm_const for p in unnormalized_probs]
        
        return neighbors, normalized_probs
    
    def biased_random_walk(self, start_node):
        walk = [start_node]
        
        for _ in range(self.walk_length - 1):
            current = walk[-1]
            neighbors = self.graph.get(current, [])
            
            if len(neighbors) == 0:
                break
            
            if len(walk) == 1:
                # First step: uniform random
                walk.append(random.choice(neighbors))
            else:
                prev = walk[-2]
                probs = []
                
                for neighbor in neighbors:
                    if neighbor == prev:
                        # Return to previous node
                        probs.append(1.0 / self.p)
                    elif neighbor in self.graph.get(prev, []):
                        # Neighbor of previous node (BFS)
                        probs.append(1.0)
                    else:
                        # Not connected to previous node (DFS)
                        probs.append(1.0 / self.q)
                
                # Normalize and sample
                probs = np.array(probs)
                probs = probs / probs.sum()
                next_node = np.random.choice(neighbors, p=probs)
                walk.append(next_node)
        
        return walk
    
    def generate_walks(self):
        walks = []
        nodes = list(self.graph.keys())
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.biased_random_walk(node))
        return walks
    
    def train(self, lr=0.025, epochs=1):
        walks = self.generate_walks()
        num_nodes = len(self.graph)
        
        # Initialize embeddings
        self.embeddings = np.random.randn(num_nodes, self.embedding_dim) * 0.01
        
        # Simple Skip-gram implementation
        for epoch in range(epochs):
            for walk in walks:
                for i, node in enumerate(walk):
                    start = max(0, i - self.window_size)
                    end = min(len(walk), i + self.window_size + 1)
                    
                    for j in range(start, end):
                        if i != j:
                            context_node = walk[j]
                            self.embeddings[node] += lr * self.embeddings[context_node]
        
        return self.embeddings


# 3. metapath2vec
class MetaPath2Vec:
    def __init__(self, graph, node_types, metapath, embedding_dim=128, 
                 walk_length=100, num_walks=10, window_size=5):
        """
        graph: dict of adjacency lists {node: [neighbors]}
        node_types: dict mapping {node: type}
        metapath: list of node types defining the meta-path, e.g., ['Author', 'Paper', 'Author']
        """
        self.graph = graph
        self.node_types = node_types
        self.metapath = metapath
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.embeddings = None
    
    def metapath_random_walk(self, start_node):
        walk = [start_node]
        
        for step in range(self.walk_length - 1):
            current = walk[-1]
            # Determine expected next node type based on meta-path
            next_type_idx = (step + 1) % len(self.metapath)
            expected_type = self.metapath[next_type_idx]
            
            # Get neighbors of the expected type
            neighbors = self.graph.get(current, [])
            valid_neighbors = [n for n in neighbors if self.node_types.get(n) == expected_type]
            
            if len(valid_neighbors) == 0:
                break
            
            walk.append(random.choice(valid_neighbors))
        
        return walk
    
    def generate_walks(self):
        walks = []
        # Only start from nodes matching the first type in meta-path
        start_type = self.metapath[0]
        start_nodes = [n for n, t in self.node_types.items() if t == start_type]
        
        for _ in range(self.num_walks):
            random.shuffle(start_nodes)
            for node in start_nodes:
                walks.append(self.metapath_random_walk(node))
        
        return walks
    
    def train(self, lr=0.025, epochs=1):
        walks = self.generate_walks()
        num_nodes = len(self.graph)
        
        # Initialize embeddings
        self.embeddings = np.random.randn(num_nodes, self.embedding_dim) * 0.01
        
        # Simple Skip-gram implementation
        for epoch in range(epochs):
            for walk in walks:
                for i, node in enumerate(walk):
                    start = max(0, i - self.window_size)
                    end = min(len(walk), i + self.window_size + 1)
                    
                    for j in range(start, end):
                        if i != j:
                            context_node = walk[j]
                            self.embeddings[node] += lr * self.embeddings[context_node]
        
        return self.embeddings
```

- without random walks (such as *SDNE* (autoencoder), *GCN*, *GraphSAGE*, *GNN*)
```python
# 1. SDNE (Structural Deep Network Embedding)
class SDNE(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128]):
        super(SDNE, self).__init__()
        # Encoder layers
        self.encoder1 = nn.Linear(input_dim, hidden_dims[0])
        self.encoder2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        
        # Decoder layers
        self.decoder1 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.decoder2 = nn.Linear(hidden_dims[0], input_dim)
    
    def encode(self, adj_matrix):
        x = F.relu(self.encoder1(adj_matrix))
        embedding = F.relu(self.encoder2(x))
        return embedding
    
    def decode(self, embedding):
        x = F.relu(self.decoder1(embedding))
        reconstruction = self.decoder2(x)
        return reconstruction
    
    def forward(self, adj_matrix):
        embedding = self.encode(adj_matrix)
        reconstruction = self.decode(embedding)
        return embedding, reconstruction


# 2. GCN (Graph Convolutional Network)
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, node_features, adj_matrix):
        # Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
        deg = adj_matrix.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm_adj = deg_inv_sqrt.view(-1, 1) * adj_matrix * deg_inv_sqrt.view(1, -1)
        
        # First layer
        x = torch.mm(norm_adj, node_features)
        x = F.relu(self.fc1(x))
        
        # Second layer
        x = torch.mm(norm_adj, x)
        x = self.fc2(x)
        return x


# 3. Generic GNN (Message Passing Neural Network)
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.message_fc = nn.Linear(input_dim, hidden_dim)
        self.update_fc = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, node_features, edge_index):
        # edge_index: [2, num_edges] tensor with source and target nodes
        num_nodes = node_features.size(0)
        
        # Message passing
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        messages = self.message_fc(node_features[src_nodes])
        
        # Aggregate messages by destination node
        aggregated = torch.zeros(num_nodes, messages.size(1), device=node_features.device)
        aggregated.index_add_(0, dst_nodes, messages)
        
        # Update node representations
        combined = torch.cat([node_features, aggregated], dim=1)
        x = F.relu(self.update_fc(combined))
        x = self.output_fc(x)
        return x

# 4. GraphSAGE
class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, node_features, sampled_neighbors):
        aggregated = torch.mean(node_features[sampled_neighbors], dim=1)
        x = F.relu(self.fc1(aggregated))
        x = self.fc2(x)
        return x
        
# USAGE
input_dim = 5
hidden_dim = 16
output_dim = 2
model = GraphSAGE(input_dim, hidden_dim, output_dim)

## Setting up loss functions and optimization algorithms
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

## training
for epoch in range(num_epochs):
    loss_accumulated = 0.0
    for node in G.nodes():
        for _ in range(num_samples):
            # Randomly sampled adjacent nodes
            sampled_neighbors = random.sample(list(G.neighbors(node)), num_neighbors)
            sampled_neighbors = torch.tensor(sampled_neighbors)
            
            # forward pass
            logits = model(torch.tensor(node_features[node], dtype=torch.float32),
                           sampled_neighbors)
            
            # Loss Calculation
            loss = criterion(logits.view(1, -1), torch.tensor([labels[node]], dtype=torch.long))
            loss_accumulated += loss.item()
            
            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss_accumulated}")

## inference
predicted_labels = []
true_labels = []
for node in G.nodes():
    sampled_neighbors = list(G.neighbors(node))
    logits = model(torch.tensor(node_features[node], dtype=torch.float32), sampled_neighbors)
    predicted_label = torch.argmax(logits).item()
    true_label = labels[node][0]
    predicted_labels.append(predicted_label)
    true_labels.append(true_label)

## Accuracy Rating
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy}")
	
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
### E(n) Equivariant Graph Neural Networks
To forecast the positions of the particles at a future time step ($t + \Delta t$), the network must shift the coordinates. The researchers modified the standard coordinate update equation by adding a specific velocity term:

$$x_i^{l+1} = x_i^l + \underbrace{\sum_{j \neq i} (x_i^l - x_j^l) \phi_x(m_{ij})}_{\text{Force/Interaction Shift}} + \underbrace{v_i \phi_v(h_i^l)}_{\text{Momentum/Velocity Shift}}$$

