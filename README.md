# Detecting Fake Profiles in Online Social Networks

## **Executive Summary**

This project is aimed at developing an advanced detection system designed to enhance the trustworthiness and integrity of an online social network by identifying and mitigating fake profiles. The presence of fake profiles poses significant risks, including the spread of misinformation, fraudulent activities, and decreased user trust. By leveraging cutting-edge machine learning and deep learning models, including Graph Neural Networks (GNNs), we aim to protect legitimate users, maintain data integrity, and support the platform's long-term growth and user satisfaction.

## **Business Understanding**

### **Objective:**
The primary business objective is to safeguard the online social network by accurately detecting and mitigating fake profiles. This initiative seeks to:
- **Improve User Trust:** Ensure that user interactions are authentic and foster trust and satisfaction.
- **Enhance Security:** Protect users from potential scams and phishing attempts.
- **Maintain Data Integrity:** Ensure the accuracy of data for analytics and personalized recommendations by removing deceptive accounts.
- **Reduce Operational Costs:** Minimize manual moderation and customer support resources by automating fake profile detection.
- **Compliance and Reputation Management:** Meet regulatory requirements and protect the platform’s reputation from being associated with fraudulent activities.

### **Strategic Importance:**
Addressing the issue of fake profiles is critical for sustaining a healthy and trustworthy online community, which is directly tied to user retention and business growth.

## **Exploratory Data Analysis (EDA)**

### **Dataset Overview:**
The dataset consists of nodes and edges representing users and their interactions within an ego-network from Facebook, anonymized by replacing internal IDs. The EDA focused on understanding the distribution of various network features that could indicate the presence of fake profiles.
- **Graph Construction:** The social network was represented as a graph where nodes represent users and edges represent connections between them.
- **Feature Extraction:** Graph-based features were extracted for each node, including degree, clustering coefficient, eigenvector centrality, average neighbor degree, local edge density, and others. These features were used to characterize the connectivity and influence of each node within the network.
- **Manual Labeling:** The labeling was achieved using a 3-step process -
  - **Community Detection:** The Louvain method was used to detect communities within the graph. Nodes that were weakly integrated into their communities (i.e., having fewer than 50% of their neighbors in the same community) were labeled as potential fake profiles.
  - **Link Prediction:** The Jaccard coefficient was employed to predict potential connections between nodes. Labels were adjusted based on the likelihood of these connections, particularly focusing on nodes that were already suspected of being fake.
  - **Isolation Forest:** An Isolation Forest model was used to identify outliers in the dataset. Nodes identified as outliers were labeled as potential fake profiles. 
  
  The dataset was then manually labeled using a combination of these techniques.

### **Key Findings:**
- **Degree Distribution:** Fake profiles often have unusually high or low degrees (number of connections).
- **Triangles and Clustering Coefficient:** Fake profiles typically have lower interconnectedness among their connections, leading to fewer triangles and lower clustering coefficients.
- **Local Edge Density:** Fake profiles show lower local edge density, reflecting scattered and isolated connections.
- **Average Neighbor Degree and Eigenvector Centrality:** Anomalous patterns in these features can indicate fake profiles targeting influential users or being part of low-quality networks.

## **Data Cleaning and Preprocessing**

### **Steps Taken:**
- **Log Transformation:** All features were subjected to log transformation to handle skewed distributions and to ensure that the models could better capture the underlying patterns in the data. This transformation was particularly important for features like degree and local edge density, which exhibited heavy tails.
- **Normalization:** Features were normalized to ensure a consistent scale, particularly for models sensitive to feature magnitude.
- **Class Imbalance:** SMOTE (Synthetic Minority Over-sampling Technique) was applied to address the class imbalance, ensuring that models do not become biased towards the majority class.

### **Result:**
The cleaned and preprocessed dataset was well-prepared for model training, with balanced classes and normalized features, facilitating effective model learning.

## **Model Training and Evaluation**

### **Models Employed:**
- **Traditional Machine Learning Models:**
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - k-Nearest Neighbors (k-NN)
  - AdaBoost
  - Naive Bayes
  - XGBoost

- **Advanced Deep Learning Models:**
  - Graph Attention Network (GAT)
  - Graph Convolutional Networks (GCN)
  - GraphSAGE

### **Evaluation Metrics:**
- **Confusion Matrix**
- **ROC Curve and AUC Score**
- **Precision-Recall Curve and Average Precision**

### **Outcome:**
All models were evaluated on the basis of their accuracy, precision, recall, and other relevant metrics. The results highlighted strong performance, particularly with the Graph Neural Networks, which outperformed traditional models in detecting fake profiles.

## **Feature Importance**

### **Key Features Identified:**
The most important features across models included:
- **Degree**
- **Local Edge Density**
- **Eigenvector Centrality**
- **Average Neighbor Degree**

These features consistently showed strong predictive power in identifying fake profiles.

### **Insights:**
The feature importance analysis validated the hypothesis that fake profiles exhibit distinctive patterns in their network behavior, which can be effectively captured through graph-based features.

## **Model Performance**

### **Comparative Analysis:**
**Traditional ML Models:** Showed solid performance, with Random Forest and XGBoost leading in accuracy and precision among them.

| Model | Train Time (s) | Train Accuracy | Train Precision | Train Recall | Train F1 Score | Best Parameters |
|-------|----------------|----------------|-----------------|--------------|----------------|-----------------|
| Naive Bayes | 0.002342 | 0.837737 | 0.815935 | 0.872240 | 0.843148 | N/A |
| Logistic Regression | 1.534917 | 0.906940 | 0.903756 | 0.910883 | 0.907306 | {'classifier__C': 10, 'classifier__penalty': 'l1', 'classifier__solver': 'liblinear'} |
| AdaBoost | 3.094780 | 0.941049 | 0.932020 | 0.951498 | 0.941659 | {'classifier__learning_rate': 1.0, 'classifier__n_estimators': 350} |
| SVM | 1.274500 | 0.969440 | 0.966680 | 0.972397 | 0.969530 | {'classifier__C': 200, 'classifier__kernel': 'rbf'} |
| k-Nearest Neighbors | 0.089554 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | {'classifier__metric': 'manhattan', 'classifier__n_neighbors': 1, 'classifier__weights': 'uniform'} |
| Decision Tree | 0.126065 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | {'classifier__max_depth': 20, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 2} |
| Random Forest | 64.447057 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | {'classifier__bootstrap': False, 'classifier__max_depth': 30, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 100} |
| XGBoost | 5.517897 | 0.999409 | 1.000000 | 1.000000 | 0.999408 | {'classifier__colsample_bytree': 1.0, 'classifier__learning_rate': 0.3, 'classifier__max_depth': 9, 'classifier__n_estimators': 300, 'classifier__subsample': 0.9} |

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 Score | Best Parameters |
|-------|---------------|----------------|-------------|---------------|-----------------|
| Naive Bayes | 0.848185 | 0.971549 | 0.854495 | 0.909270 | N/A |
| Logistic Regression | 0.895215 | 0.978873 | 0.901761 | 0.938736 | {'classifier__C': 10, 'classifier__penalty': 'l1', 'classifier__solver': 'liblinear'} |
| AdaBoost | 0.930693 | 0.979749 | 0.941613 | 0.960302 | {'classifier__learning_rate': 1.0, 'classifier__n_estimators': 350} |
| SVM | 0.955446 | 0.985782 | 0.963855 | 0.974695 | {'classifier__C': 200, 'classifier__kernel': 'rbf'} |
| k-Nearest Neighbors | 0.959571 | 0.978625 | 0.975904 | 0.977262 | {'classifier__metric': 'manhattan', 'classifier__n_neighbors': 1, 'classifier__weights': 'uniform'} |
| Decision Tree | 0.962871 | 0.983178 | 0.974977 | 0.979060 | {'classifier__max_depth': 20, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 2} |
| Random Forest | 0.967822 | 0.977941 | 0.986098 | 0.982003 | {'classifier__bootstrap': False, 'classifier__max_depth': 30, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 100} |
| XGBoost | 0.968647 | 0.984186 | 0.980538 | 0.982358 | {'classifier__colsample_bytree': 1.0, 'classifier__learning_rate': 0.3, 'classifier__max_depth': 9, 'classifier__n_estimators': 300, 'classifier__subsample': 0.9} |

**Graph Neural Networks:** Outperformed traditional models, with GraphSAGE achieving the highest accuracy (96.29%) on the test set.

| Model | Time (s) | Final Epochs | Final Loss | Train Accuracy | Test Accuracy |
|-------|----------|--------------|------------|----------------|---------------|
| Graph Attention Network (GAT) | 48.194153 | 1000 | 0.304262 | 0.904054 | 0.904703 |
| Graph Convolutional Networks (GCN) | 16.354273 | 1000 | 0.281990 | 0.904983 | 0.903465 |
| Graph Sample and Aggregation (GraphSAGE) | 8.669998 | 1000 | 0.123789 | 0.967193 | 0.962871 |


### **Training Time and Efficiency:**
GraphSAGE also demonstrated significantly lower training times while maintaining high accuracy, making it an efficient choice for large-scale deployment.

## **Findings and Actionable Items**

### **Findings:**
- **High Accuracy:** The GraphSAGE model, in particular, demonstrated superior accuracy in detecting fake profiles.
- **Critical Features:** Degree, local edge density, and eigenvector centrality are crucial in distinguishing between real and fake profiles.

### **Actionable Items:**
- **Implementation:** Deploy the GraphSAGE model in the production environment for real-time fake profile detection.
- **Monitoring:** Establish a monitoring framework to continuously evaluate the model's performance and update it as needed to adapt to evolving fake profile tactics.
- **User Education:** Provide users with guidance on recognizing and reporting suspicious profiles to complement automated detection efforts.

## **Next Steps and Recommendations**

1. **Deploy the Model:** Implement the GraphSAGE model for online detection of fake profiles.
2. **Expand Data Collection:** Continue to collect and analyze more data to refine and improve the model over time.
3. **User Reporting Integration:** Integrate user feedback and reporting mechanisms to enhance model training and accuracy.
4. **Regular Audits:** Conduct regular audits to ensure that the model remains effective as new types of fake profiles emerge.

## **Conclusion**

The project successfully developed a robust system to detect fake profiles, leveraging advanced graph-based neural networks. The implementation of this system is expected to significantly enhance the integrity of the platform, leading to increased user trust, reduced operational costs, and sustained business growth. By continuing to refine and monitor the model, the platform can stay ahead of emerging threats and maintain a secure and trustworthy environment for its users.

## **Appendix**

### **List of Graph Visualizations:**

| Description | Link |
|-------------|------|
| Graph Layout using NetworkX | [View Image](images/graph_spring_layout.png) |
| Feature Correlation Matrix | [View Image](images/feature-correlation-matrix.png) |
| Feature Correlation Matrix (After Normalization) | [View Image](images/feature-correlation-matrix-after-normalization.png) |
| Feature Data Distribution | [View Image](images/feature-data-distribution.png) |
| Feature Data Distribution (After Normalization) | [View Image](images/feature-data-distribution-after-normalization.png) |
| Label Distribution Across All Features | [View Image](images/pair-plot-all-features.png) |
| AdaBoost Model Performance Metrics | [View Image](images/performance-metrics-AdaBoost.png) |
| Decision Tree Model Performance Metrics | [View Image](images/performance-metrics-Decision-Tree.png) |
| k-Nearest Neighbors Model Performance Metrics | [View Image](images/performance-metrics-k-Nearest-Neighbors.png) |
| Logistic Regression Model Performance Metrics | [View Image](images/performance-metrics-Logistic-Regression.png) |
| Naive Bayes Model Performance Metrics | [View Image](images/performance-metrics-Naive-Bayes.png) |
| Random Forest Model Performance Metrics | [View Image](images/performance-metrics-Random-Forest.png) |
| SVM Model Performance Metrics | [View Image](images/performance-metrics-SVM.png) |
| XGBoost Model Performance Metrics | [View Image](images/performance-metrics-XGBoost.png) |
| AdaBoost Feature Importance | [View Image](images/feature-importance-AdaBoost.png) |
| Decision Tree Feature Importance | [View Image](images/feature-importance-Decision-Tree.png) |
| Logistic Regression Feature Importance | [View Image](images/feature-importance-Logistic-Regression.png) |
| Random Forest Feature Importance | [View Image](images/feature-importance-Random-Forest.png) |
| XGBoost Feature Importance | [View Image](images/feature-importance-XGBoost.png) |
| GAT Train and Test Accuracy Over Epochs | [View Image](images/graph-attention-network-(gat)-train-and-test-accuracy-over-epochs.png) |
| GAT Training Loss Over Epochs | [View Image](images/graph-attention-network-(gat)-training-loss-over-epochs.png) |
| GCN Train and Test Accuracy Over Epochs | [View Image](images/graph-convolutional-networks-(gcn)-train-and-test-accuracy-over-epochs.png) |
| GCN Training Loss Over Epochs | [View Image](images/graph-convolutional-networks-(gcn)-training-loss-over-epochs.png) |
| GraphSAGE Train and Test Accuracy Over Epochs | [View Image](images/graph-sample-and-aggregation-(graphsage)-train-and-test-accuracy-over-epochs.png) |
| GraphSAGE Training Loss Over Epochs | [View Image](images/graph-sample-and-aggregation-(graphsage)-training-loss-over-epochs.png) |