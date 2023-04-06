class: middle, center, title-slide

# Deep Learning

Lecture 9: Graph neural networks

<br><br>
Prof. Gilles Louppe<br>
[g.louppe@uliege.be](mailto:g.louppe@uliege.be)

???

R: geometric DL course https://geometricdeeplearning.com/lectures/ => Lec 5 and 6
R: https://twitter.com/thomaskipf/status/1499470015455805451?t=1jy4X1fhk_vo1rjSO-2YFQ&s=03
R: https://twitter.com/xbresson/status/1629898461402497024?t=quHYF-x9RNFnVsLqfd264A&s=03 Bresson
R: https://twitter.com/CSProfKGD/status/1638742528588673024?t=FWeRSkQOoOTF59cPu7YOqQ&s=03
R: https://arxiv.org/abs/2301.08210 review by Petar Veličković

---

# Today

- Graphs 
- Graph neural networks
- Applications

---

motivation
    traffic networks
    social networks
    molecular graphs
    protein graphs
    scene graphs
    -> graphs are everywhere
graph-structured data

---

graph basics
- nodes
- edges
- adjacency matrix
- features

---

tasks on graphs
- graph prediction (want invariance to node order)
- node prediction (want equivariance to node order)
- edge prediction (want equivariance to node order)

---

Permutation invariance at the graph level (f)
- Regular NNs are sensitive to the order of the nodes, but graphs do not have a canonical ordering
- Formally, we want permutation invariance, i.e. f(X, A) = f(XP, PAP^T) for all permutations P

---

Permutation equivariance at the node level (F)
- We want permutation equivariance, i.e. F(PX, PAP^T) = PF(X, A) for all permutations P
- For F(X) = [h1, ..., H_|V|], h_i = phi(x_i, set(x_j's)), this can be achieved by shared 
  application of a permutation-invariant (of set(x_j's)) function on each node (2301.08210 near eq 7 or Slide 37 AIMS Lec 5)
        Prove it  (similar to convolutions for which shuffling the input pixels within a local receptive field does not change the output of the kernel)

---

Locality
- h_i = phi(x_i,  N(x_i))
- Why locality? 
    Inductive bias: the topology encodes the structure of the data, and information about a node is often most relevant to its close neighbors rather than distant ones
    Computational efficiency: locality allows us to process the graph in parallel, and to avoid storing the entire graph in memory

---

Layers
- Message-passing (1 round)
- Convolutional
    derive as a special case of message-passing
- Attentional
    derive as a special case of message-passing

---

Blueprint for building graph neural networks:
- Sequential composition => Propagation of information across the graph => The effective neighborhood of a node grows with the depth of the network
    (similarly to CNNs in which the effective receptive field grows with the depth of the network)
- Pooling layers
    - Coarsening
    - Global pooling

---

Special cases / The graph is unknown
- No edge <=> sets => deep sets
- Fully connected graph => self-attention => make the equivalence explicit!
- Sequential graph => RNN

---

Applications
- Point-GNN; Point clouds, LIDAR, etc
- Chemistry/molecular/drug discovery
- Learning to simulate physics
     Encoder-process-decoder architecture
- Balthazar power grid case

---

Geometric deep learning

---

class: end-slide, center
count: false

The end.
