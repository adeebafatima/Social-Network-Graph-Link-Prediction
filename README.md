# Social-Network-Graph-Link-Prediction

### Problem statement: 
Given a directed social graph, have to predict missing links to recommend users (Link Prediction in graph)

### Data Overview
Taken data from facebook's recruting challenge on kaggle https://www.kaggle.com/c/FacebookRecruiting  
data contains two columns source and destination each edge in graph 
    - Data columns (total 2 columns):  
    - source_node         int64  
    - destination_node    int64  
    
### Mapping the problem into supervised learning problem:
Generated training samples of good and bad links from given directed graph and for each link extract features like 
<ul>
<li>Jaccard Distance for followees</li>
<li>Jaccard Distance for followers</li>
<li>Cosine Distance for followees</li>
<li>Cosine Distance for followers</li>
<li>No. of followers of source node</li>
<li>No. of followees of source node</li>
<li>No. of followees of destination node</li>
<li>Intersection of followers of source and destination node</li>
<li>Intersection of followees of source and destination node</li>
<li>Adar Index</li>
<li>Follows Back</li>
<li>Same community</li>
<li>Shortest path</li>
<li>Page Rank</li>
<li>Kartz Centrality</li>
<li>HITS Score</li>
<li>Weight Feature</li>
<li>SVD</li>
<li>Preferential attachment for followees</li>
<li>Preferential attachment for followers</li>
<li>SVD Dot Product</li>
</ul>

Trained ML model(Random Forest and Xgboost) based on these features to predict link. 
- Reference papers:  
    - https://www.cs.cornell.edu/home/kleinber/link-pred.pdf
    - https://www3.nd.edu/~dial/publications/lichtenwalter2010new.pdf
    
### Business objectives and constraints:  
- No low-latency requirement.
- Probability of prediction is useful to recommend highest probability links

### Performance metric for supervised learning:  
- Both precision and recall is important so F1 score is good choice
- Confusion matrix

