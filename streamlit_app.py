import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Set page configuration
st.set_page_config(page_title="House Price Classification", layout="wide")
st.title('House Price Classification Analysis')

# Create tabs
tab1, tab2, tab3 = st.tabs(["Data Processing & Training", "Model Performance", "Make Prediction"])

with tab1:
    st.header("Data Processing and Model Training")
    
    # Load data
    df = pd.read_csv('house_prices_1.csv')
    st.write("### Data Sample")
    st.dataframe(df.head())

    # Create price categories with binary classification
    df['price_category'] = pd.qcut(df['price'], q=2, labels=['low', 'high'])

    # Initialize imputer early
    imputer = SimpleImputer(strategy='mean')

    # Feature Engineering bölümünü güncelleyelim
    # Temel ve anlamlı özellikler oluşturalım
    df['age'] = 2015 - df['yr_built']  # Basit yaş hesabı

    # Renovasyon özelliklerini basitleştirelim
    df['has_renovation'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
    df['years_since_renovation'] = df.apply(
        lambda x: 2015 - x['yr_renovated'] if x['yr_renovated'] > 0 else x['age'],
        axis=1
    )

    # Alan bazlı özellikleri basitleştirelim
    df['total_sqft'] = df['sqft_living'] + df['sqft_lot']
    df['living_lot_ratio'] = df['sqft_living'] / df['sqft_lot'].replace(0, 1)
    df['basement_ratio'] = df['sqft_basement'] / df['sqft_living'].replace(0, 1)

    # Oda bazlı özellikleri basitleştirelim
    df['bath_bed_ratio'] = df['bathrooms'] / df['bedrooms'].replace(0, 1)
    df['avg_room_size'] = df['sqft_living'] / (df['bedrooms'] + df['bathrooms']).replace(0, 1)

    # Kalite göstergeleri
    df['condition_age_interaction'] = df['condition'] * df['age']
    df['view_waterfront_interaction'] = df['view'] * df['waterfront']

    # Güncellenmiş özellik listesi - polynomial features'ları kaldırdık
    features = [
        # Temel özellikler
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition',
        
        # Zaman bazlı özellikler
        'age', 'has_renovation', 'years_since_renovation',
        
        # Alan bazlı özellikler
        'total_sqft', 'living_lot_ratio', 'basement_ratio',
        
        # Oda bazlı özellikler
        'bath_bed_ratio', 'avg_room_size',
        
        # Kalite göstergeleri
        'condition_age_interaction', 'view_waterfront_interaction'
    ]

    # Correlation analizi
    def remove_highly_correlated_features(df, threshold=0.8):
        correlation_matrix = df.corr().abs()
        upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return df.drop(columns=to_drop)

    # Yüksek korelasyonlu özellikleri kaldır
    X = df[features]
    X = remove_highly_correlated_features(X, threshold=0.8)
    features = X.columns.tolist()

    # Veri ön işleme
    X = df[features]
    y = df['price_category']

    # Remove rows where y is NaN
    valid_indices = ~y.isna()
    X = X[valid_indices]
    y = y[valid_indices]

    # Create new imputer for final feature matrix
    final_imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(
        final_imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X_imputed, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    st.write("### Selected Features:")
    st.write(selected_features)

    # Split and Scale with selected features
    X = X_imputed[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Veri dengeleme fonksiyonunu düzeltelim
    def balance_dataset(X, y, sampling_strategy=0.8):
        # Sınıfları belirle
        majority_class = y.value_counts().index[0]
        minority_class = y.value_counts().index[1]
        
        # Dataframe'leri oluştur
        df_majority = pd.DataFrame(X[y == majority_class])
        df_minority = pd.DataFrame(X[y == minority_class])
        
        # Majority class örnek sayısını hesapla
        n_minority = len(df_minority)
        n_majority = min(
            len(df_majority),  # Mevcut majority örnek sayısı
            int(n_minority / sampling_strategy)  # Hedeflenen örnek sayısı
        )
        
        # Downsample majority class
        df_majority_downsampled = resample(
            df_majority, 
            replace=False,
            n_samples=n_majority,
            random_state=42
        )
        
        # Combine minority class with downsampled majority class
        X_balanced = pd.concat([df_majority_downsampled, df_minority])
        y_balanced = pd.concat([
            pd.Series([majority_class] * len(df_majority_downsampled)), 
            pd.Series([minority_class] * len(df_minority))
        ])
        
        return X_balanced, y_balanced

    # Sınıf dağılımını kontrol et ve dengele
    st.write("### Class Distribution Before Balancing:")
    st.write(y.value_counts(normalize=True))

    # Veri setini dengele
    X_balanced, y_balanced = balance_dataset(X_selected, y, sampling_strategy=0.8)

    st.write("### Class Distribution After Balancing:")
    st.write(y_balanced.value_counts(normalize=True))

    # 3. Threshold değerini ayarlayalım
    def predict_with_custom_threshold(model, X, threshold=0.5):
        proba = model.predict_proba(X)
        return 'high' if proba[0][1] >= threshold else 'low'

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(
            C=0.1,
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=4,
            min_samples_split=15,
            min_samples_leaf=8,
            class_weight='balanced',
            random_state=42
        ),
        'SVM': SVC(
            C=0.1,
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=15,
            random_state=42
        ),
        'Naive Bayes': GaussianNB(
            priors=[0.5, 0.5]
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=11,
            weights='distance',
            metric='manhattan'
        )
    }

    # Setup k-fold cross validation
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Dictionary to store results
    results = {}
    
    # Training Progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Train and evaluate models
    for idx, (name, model) in enumerate(models.items()):
        status_text.text(f'Training {name}...')
        
        # Perform k-fold cross-validation
        scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='accuracy')
        st.write(f"{name} CV Scores: {scores}")
        st.write(f"{name} Average CV Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        progress_bar.progress((idx + 1)/len(models))

    status_text.text('Training Complete!')

    # Feature engineering kısmında (mevcut feature'lardan sonra)
    def add_unsupervised_features(X, n_components=3, n_clusters=3):
        # PCA uygula
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(X)
        
        # K-Means uygula
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Yeni feature'ları DataFrame'e ekle
        X_new = pd.DataFrame(X, columns=X_imputed.columns)  # Orijinal sütun isimlerini kullan
        
        # PCA bileşenlerini ekle
        pca_df = pd.DataFrame(
            pca_features,
            columns=[f'pca_component_{i+1}' for i in range(n_components)],
            index=X_new.index
        )
        X_new = pd.concat([X_new, pca_df], axis=1)
        
        # Cluster labels'ı ekle
        X_new['cluster_label'] = cluster_labels
        
        # Cluster merkezine olan uzaklığı ekle
        X_new['distance_to_cluster_center'] = np.min(
            kmeans.transform(X),
            axis=1
        )
        
        return X_new, pca.explained_variance_ratio_

    # Feature selection öncesinde unsupervised features ekle
    st.write("### Adding Unsupervised Learning Features")

    # Normalize data for unsupervised learning
    scaler_unsup = StandardScaler()
    X_scaled_unsup = scaler_unsup.fit_transform(X_imputed)

    # Add unsupervised features
    X_with_unsup, explained_var_ratio = add_unsupervised_features(
        X_scaled_unsup,
        n_components=3,
        n_clusters=3
    )

    # Show explained variance ratio for PCA
    st.write("#### PCA Explained Variance Ratio:")
    explained_var_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(explained_var_ratio))],
        'Explained Variance Ratio': explained_var_ratio,
        'Cumulative Variance Ratio': np.cumsum(explained_var_ratio)
    })
    st.dataframe(explained_var_df)

    # Show cluster distribution
    st.write("#### Cluster Distribution:")
    cluster_dist = pd.DataFrame(X_with_unsup['cluster_label'].value_counts()).reset_index()
    cluster_dist.columns = ['Cluster', 'Count']
    st.bar_chart(cluster_dist.set_index('Cluster'))

    # Update feature selection with new features
    X = X_with_unsup
    selector = SelectKBest(score_func=f_classif, k=12)  # k değerini artırdık
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()

    st.write("### Selected Features (including unsupervised features):")
    st.write(selected_features)

with tab2:
    st.header("Model Performance Analysis")
    
    # GridSearchCV için parametre gridleri
    param_grids = {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'max_iter': [1000, 2000],
            'class_weight': ['balanced', None]
        },
        'Decision Tree': {
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [5, 10, 15, 20],
            'min_samples_leaf': [4, 8, 12]
        },
        'SVM': {
            'C': [0.01, 0.1, 1],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [2, 3, 4]
        },
        'Naive Bayes': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        },
        'KNN': {
            'n_neighbors': [5, 7, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    }
    
    # GridSearchCV için tuning kodu
    tuned_models = {}
    tuned_results = {}
    
    # Perform hyperparameter tuning
    with st.spinner("Performing Hyperparameter Tuning..."):
        for name, model in models.items():
            if name in param_grids:
                grid_search = GridSearchCV(
                    model,
                    param_grids[name],
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1
                )
                grid_search.fit(X_train_scaled, y_train)
                tuned_models[name] = grid_search.best_estimator_
                
                # Evaluate tuned model
                y_pred = grid_search.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                
                tuned_results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'best_params': grid_search.best_params_
                }
    
    # Create DataFrames for both before and after tuning
    before_metrics_df = pd.DataFrame({
        'Model': results.keys(),
        'Accuracy': [results[model]['accuracy'] for model in results],
        'Precision': [results[model]['precision'] for model in results],
        'Recall': [results[model]['recall'] for model in results],
        'F1 Score': [results[model]['f1'] for model in results]
    }).set_index('Model')

    after_metrics_df = pd.DataFrame({
        'Model': tuned_results.keys(),
        'Accuracy': [tuned_results[model]['accuracy'] for model in tuned_results],
        'Precision': [tuned_results[model]['precision'] for model in tuned_results],
        'Recall': [tuned_results[model]['recall'] for model in tuned_results],
        'F1 Score': [tuned_results[model]['f1'] for model in tuned_results]
    }).set_index('Model')
    
    # Display tables side by side
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Performance Metrics (Before Tuning)")
        st.dataframe(
            before_metrics_df.style.format("{:.3f}"),
            use_container_width=True
        )
    
    with col2:
        st.write("### Performance Metrics (After Tuning)")
        st.dataframe(
            after_metrics_df.style.format("{:.3f}"),
            use_container_width=True
        )
    
    # Visualization - Always show both before and after
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy Plot
        st.write("### Accuracy Comparison")
        fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
        
        data = []
        for name in tuned_results.keys():
            data.extend([
                {'Model': name, 'Accuracy': results[name]['accuracy'], 'Type': 'Before Tuning'},
                {'Model': name, 'Accuracy': tuned_results[name]['accuracy'], 'Type': 'After Tuning'}
            ])
        df_plot = pd.DataFrame(data)
        sns.barplot(data=df_plot, x='Model', y='Accuracy', hue='Type')
        
        plt.xticks(rotation=45)
        plt.title('Model Accuracy Comparison')
        plt.tight_layout()
        st.pyplot(fig_acc)
    
    with col2:
        # F1 Score Plot
        st.write("### F1 Score Comparison")
        fig_f1, ax_f1 = plt.subplots(figsize=(10, 6))
        
        data = []
        for name in tuned_results.keys():
            data.extend([
                {'Model': name, 'F1 Score': results[name]['f1'], 'Type': 'Before Tuning'},
                {'Model': name, 'F1 Score': tuned_results[name]['f1'], 'Type': 'After Tuning'}
            ])
        df_plot = pd.DataFrame(data)
        sns.barplot(data=df_plot, x='Model', y='F1 Score', hue='Type')
        
        plt.xticks(rotation=45)
        plt.title('Model F1 Score Comparison')
        plt.tight_layout()
        st.pyplot(fig_f1)

    # Precision ve Recall plotları için yeni bir satır
    col3, col4 = st.columns(2)
    
    with col3:
        # Precision Plot
        st.write("### Precision Comparison")
        fig_prec, ax_prec = plt.subplots(figsize=(10, 6))
        
        data = []
        for name in tuned_results.keys():
            data.extend([
                {'Model': name, 'Precision': results[name]['precision'], 'Type': 'Before Tuning'},
                {'Model': name, 'Precision': tuned_results[name]['precision'], 'Type': 'After Tuning'}
            ])
        df_plot = pd.DataFrame(data)
        sns.barplot(data=df_plot, x='Model', y='Precision', hue='Type')
        
        plt.xticks(rotation=45)
        plt.title('Model Precision Comparison')
        plt.tight_layout()
        st.pyplot(fig_prec)
    
    with col4:
        # Recall Plot
        st.write("### Recall Comparison")
        fig_recall, ax_recall = plt.subplots(figsize=(10, 6))
        
        data = []
        for name in tuned_results.keys():
            data.extend([
                {'Model': name, 'Recall': results[name]['recall'], 'Type': 'Before Tuning'},
                {'Model': name, 'Recall': tuned_results[name]['recall'], 'Type': 'After Tuning'}
            ])
        df_plot = pd.DataFrame(data)
        sns.barplot(data=df_plot, x='Model', y='Recall', hue='Type')
        
        plt.xticks(rotation=45)
        plt.title('Model Recall Comparison')
        plt.tight_layout()
        st.pyplot(fig_recall)

    # Confusion Matrices
    st.write("### Confusion Matrices")
    conf_col1, conf_col2, conf_col3 = st.columns(3)
    
    cols = [conf_col1, conf_col2, conf_col3]
    for idx, (name, model) in enumerate(models.items()):
        with cols[idx % 3]:
            st.write(f"#### {name}")
            y_pred = model.predict(X_test_scaled)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name} Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            st.pyplot(fig_cm)

    # Mevcut görselleştirmelerden sonra en iyi model karşılaştırması ekleyelim
    st.write("### Best Model Comparison")
    
    # En iyi modeli belirle (tüm metriklerin ortalamasına göre)
    def calculate_overall_score(model_results):
        return (model_results['accuracy'] + 
                model_results['precision'] + 
                model_results['recall'] + 
                model_results['f1']) / 4

    # Before tuning için en iyi model
    before_scores = {name: calculate_overall_score(model_data) 
                    for name, model_data in results.items()}
    best_model_before = max(before_scores.items(), key=lambda x: x[1])[0]

    # After tuning için en iyi model
    after_scores = {name: calculate_overall_score(model_data) 
                   for name, model_data in tuned_results.items()}
    best_model_after = max(after_scores.items(), key=lambda x: x[1])[0]

    # En iyi modellerin tüm metriklerini karşılaştıran plot
    fig_best, ax_best = plt.subplots(figsize=(12, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    before_values = [
        results[best_model_before]['accuracy'],
        results[best_model_before]['precision'],
        results[best_model_before]['recall'],
        results[best_model_before]['f1']
    ]
    after_values = [
        tuned_results[best_model_after]['accuracy'],
        tuned_results[best_model_after]['precision'],
        tuned_results[best_model_after]['recall'],
        tuned_results[best_model_after]['f1']
    ]

    x = np.arange(len(metrics))
    width = 0.35

    rects1 = ax_best.bar(x - width/2, before_values, width, 
                        label=f'Before Tuning ({best_model_before})')
    rects2 = ax_best.bar(x + width/2, after_values, width, 
                        label=f'After Tuning ({best_model_after})')

    ax_best.set_ylabel('Score')
    ax_best.set_title('Best Model Performance Comparison')
    ax_best.set_xticks(x)
    ax_best.set_xticklabels(metrics)
    ax_best.legend()

    # Bar değerlerini göster
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax_best.annotate(f'{height:.3f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    st.pyplot(fig_best)

    # En iyi modeller hakkında bilgi
    st.write(f"""
    #### Best Models Summary:
    - Before Tuning: **{best_model_before}** (Average Score: {before_scores[best_model_before]:.3f})
    - After Tuning: **{best_model_after}** (Average Score: {after_scores[best_model_after]:.3f})
    """)

with tab3:
    st.header("Make Predictions")
    
    # Create a form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            bedrooms = st.number_input('Bedrooms', min_value=1, max_value=10, value=3)
            bathrooms = st.number_input('Bathrooms', min_value=1.0, max_value=10.0, value=2.0)
            sqft_living = st.number_input('Living Space (sqft)', min_value=500, max_value=15000, value=2000)
            sqft_lot = st.number_input('Lot Size (sqft)', min_value=500, max_value=100000, value=5000)
            floors = st.number_input('Floors', min_value=1, max_value=4, value=1)
            sqft_above = st.number_input('Above Ground sqft', min_value=0, max_value=10000, value=1500)
        
        with col2:
            waterfront = st.selectbox('Waterfront', [0, 1])
            view = st.selectbox('View', [0, 1, 2, 3, 4])
            condition = st.selectbox('Condition', [1, 2, 3, 4, 5])
            sqft_basement = st.number_input('Basement (sqft)', min_value=0, max_value=5000, value=0)
            yr_built = st.number_input('Year Built', min_value=1900, max_value=2014, value=2000)
            yr_renovated = st.number_input('Year Renovated', min_value=0, max_value=2014, value=0)

        submitted = st.form_submit_button("Predict")

    # Move placeholders after the form
    prediction_status = st.empty()
    prediction_results = st.empty()

    # Prediction logic remains the same
    if submitted:
        with prediction_status:
            with st.spinner('Making predictions...'):
                # Create base input data
                input_data = pd.DataFrame({
                    'bedrooms': [bedrooms],
                    'bathrooms': [bathrooms],
                    'sqft_living': [sqft_living],
                    'sqft_lot': [sqft_lot],
                    'floors': [floors],
                    'waterfront': [waterfront],
                    'view': [view],
                    'condition': [condition],
                    'sqft_basement': [sqft_basement],
                    'yr_built': [yr_built],
                    'yr_renovated': [yr_renovated]
                })
                
                # Add engineered features
                input_data['age'] = 2015 - input_data['yr_built']
                input_data['has_renovation'] = input_data['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
                input_data['years_since_renovation'] = input_data.apply(
                    lambda x: 2015 - x['yr_renovated'] if x['yr_renovated'] > 0 else x['age'],
                    axis=1
                )
                
                # Alan bazlı özellikler
                input_data['total_sqft'] = input_data['sqft_living'] + input_data['sqft_lot']
                input_data['living_lot_ratio'] = input_data['sqft_living'] / input_data['sqft_lot'].replace(0, 1)
                input_data['basement_ratio'] = input_data['sqft_basement'] / input_data['sqft_living'].replace(0, 1)
                
                # Oda bazlı özellikler
                input_data['bath_bed_ratio'] = input_data['bathrooms'] / input_data['bedrooms'].replace(0, 1)
                input_data['avg_room_size'] = input_data['sqft_living'] / (input_data['bedrooms'] + input_data['bathrooms']).replace(0, 1)
                
                # Kalite göstergeleri
                input_data['condition_age_interaction'] = input_data['condition'] * input_data['age']
                input_data['view_waterfront_interaction'] = input_data['view'] * input_data['waterfront']
                
                try:
                    # Select only the features that were used in training
                    input_data = input_data[selected_features]
                    
                    # Scale the input
                    input_scaled = scaler.transform(input_data)
                    
                    # Create results string
                    results_markdown = "### Model Predictions:\n\n"
                    for name, model in models.items():
                        # Normal prediction
                        prediction = predict_with_custom_threshold(model, input_scaled, threshold=0.5)
                        probability = model.predict_proba(input_scaled)
                        
                        # Confidence değerini düzelt
                        confidence = probability[0][1] if prediction == 'high' else probability[0][0]
                        
                        results_markdown += f"**{name}:**\n"
                        results_markdown += f"- Predicted Category: {prediction}\n"
                        results_markdown += f"- Confidence: {confidence*100:.2f}%\n"
                        results_markdown += f"- Raw Probabilities: Low: {probability[0][0]*100:.2f}%, High: {probability[0][1]*100:.2f}%\n"
                        results_markdown += "---\n"
                    
                    # Display results using the placeholder
                    prediction_results.markdown(results_markdown)
                    prediction_status.empty()
                    
                except Exception as e:
                    prediction_status.empty()
                    prediction_results.error(f"Error in prediction: {str(e)}")
