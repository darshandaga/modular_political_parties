from pathlib import Path

from matplotlib import pyplot

from political_party_analysis.loader import DataLoader
from political_party_analysis.dim_reducer import DimensionalityReducer
from political_party_analysis.visualization import scatter_plot

if __name__ == "__main__":

    data_loader = DataLoader()
    print("Original DataFrame Info:")
    print(f"Shape: {data_loader.party_data.shape}")
    print(f"Columns: {list(data_loader.party_data.columns)}")
    print("\nFirst 5 rows:")
    print(data_loader.party_data.head())
    
    # Data pre-processing step
    print("\n" + "="*50)
    print("PREPROCESSING DATA...")
    print("="*50)
    
    # Call the preprocess_data method
    preprocessed_data = data_loader.preprocess_data()
    
    print(f"\nAfter preprocessing:")
    print(f"Shape: {data_loader.party_data.shape}")
    print(f"Index: {data_loader.party_data.index.names}")
    print("\nFirst 5 rows of preprocessed data:")
    print(data_loader.party_data.head())

    # Dimensionality reduction step
    print("\n" + "="*50)
    print("DIMENSIONALITY REDUCTION...")
    print("="*50)
    
    # Apply PCA to reduce dimensions to 2D
    dim_reducer = DimensionalityReducer("PCA", data_loader.party_data)
    reduced_dim_data = dim_reducer.transform()
    print("--------------------------------")
    print(reduced_dim_data)
    print("--------------------------------")
    print(f"\nDimensionality reduction results:")
    print(f"Original shape: {data_loader.party_data.shape}")
    #print(f"Reduced shape: {reduced_dim_data.shape}")
    
    # Print explained variance ratio
    explained_variance = dim_reducer.get_explained_variance_ratio()
    if explained_variance is not None:
        print(f"Explained variance ratio:")
        for i, var in enumerate(explained_variance):
            print(f"  PC{i+1}: {var:.3f} ({var*100:.1f}%)")
        print(f"Total explained variance: {sum(explained_variance):.3f} ({sum(explained_variance)*100:.1f}%)")
    
    print(f"\nFirst 5 rows of reduced data:")
    print(reduced_dim_data.head())
    
    # Show feature importance (loadings) for interpretation
    feature_importance = dim_reducer.get_feature_importance()
    if feature_importance is not None:
        print(f"\nTop 5 features contributing to each component:")
        for col in feature_importance.columns:
            print(f"\n{col}:")
            top_features = feature_importance[col].abs().sort_values(ascending=False).head(5)
            for feature, loading in top_features.items():
                print(f"  {feature}: {feature_importance.loc[feature, col]:.3f}")

    # Plot dim reduced data
    pyplot.figure(figsize=(10, 8))
    splot = pyplot.subplot()
    scatter_plot(
        reduced_dim_data,
        color="r",
        splot=splot,
        label="dim reduced data",
    )
    pyplot.title("Political Parties - PCA Dimensionality Reduction")
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "dim_reduced_data.png"]))
    print(f"\nDimensionality reduction plot saved to: plots/dim_reduced_data.png")

    # Density estimation/distribution modelling step
    print("\n" + "="*50)
    print("DENSITY ESTIMATION...")
    print("="*50)
    
    # Import the DensityEstimator
    from political_party_analysis.estimator import DensityEstimator
    from political_party_analysis.visualization import plot_density_estimation_results
    
    # Get the original feature names before dimensionality reduction
    original_feature_names = data_loader.party_data.columns.tolist()
    
    # Create density estimator with reduced dimensional data
    density_estimator = DensityEstimator(
        data=reduced_dim_data,
        dim_reducer=dim_reducer,
        high_dim_feature_names=original_feature_names
    )
    
    # Fit Gaussian Mixture Model
    gmm_model = density_estimator.fit_gaussian_mixture()
    print(f"Fitted Gaussian Mixture Model with {density_estimator.n_components} components")
    
    # Get cluster assignments for all data points
    cluster_assignments = density_estimator.get_cluster_assignments()
    print(f"Cluster assignments shape: {cluster_assignments.shape}")
    
    # Sample 10 parties from the distribution
    sampled_parties, sample_labels = density_estimator.sample_from_distribution(n_samples=10)
    print(f"\nSampled 10 parties from the distribution:")
    print(f"Sampled data shape: {sampled_parties.shape}")
    print("First 5 sampled parties:")
    print(sampled_parties.head())
    
    # Map sampled parties back to high-dimensional space
    high_dim_samples = density_estimator.map_to_high_dimension(sampled_parties)
    if high_dim_samples is not None:
        print(f"\nMapped samples back to high-dimensional space:")
        print(f"High-dimensional samples shape: {high_dim_samples.shape}")
        print("First 5 high-dimensional samples:")
        print(high_dim_samples.head())
    
    # Get means and covariances for plotting
    means, covariances = density_estimator.get_means_and_covariances()
    print(f"\nGaussian components:")
    print(f"Number of means: {len(means)}")
    print(f"Number of covariances: {len(covariances)}")

    # Plot density estimation results here
    plot_density_estimation_results(
        X=reduced_dim_data,
        Y_=cluster_assignments,
        means=means,
        covariances=covariances,
        title="Political Parties - Density Estimation (Gaussian Mixture Model)"
    )
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "density_estimation.png"]))

    # Plot left and right wing parties here
    pyplot.figure()
    splot = pyplot.subplot()
    
    # Create a simple left/right classification based on cluster assignments
    # We'll use the cluster assignments from the density estimation
    print("\n" + "="*50)
    print("PLOTTING LEFT/RIGHT WING PARTIES...")
    print("="*50)
    
    # Plot different clusters with different colors to represent political orientations
    colors = ['red', 'blue', 'green']
    labels = ['Left-wing', 'Center', 'Right-wing']
    
    for i in range(density_estimator.n_components):
        cluster_mask = cluster_assignments == i
        cluster_data = reduced_dim_data[cluster_mask]
        
        if not cluster_data.empty:
            splot.scatter(
                cluster_data.iloc[:, 0],
                cluster_data.iloc[:, 1],
                c=colors[i % len(colors)],
                s=30,
                label=labels[i % len(labels)],
                alpha=0.7
            )
    
    splot.set_xlabel("1st Principal Component")
    splot.set_ylabel("2nd Principal Component")
    splot.legend()
    splot.set_aspect("equal", "box")
    pyplot.title("Left/Right Wing Parties (Based on Clustering)")
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "left_right_parties.png"]))
    print("Left/right wing parties plot saved to: plots/left_right_parties.png")

    # Plot finnish parties here
    print("\n" + "="*50)
    print("PLOTTING FINNISH PARTIES...")
    print("="*50)
    
    from political_party_analysis.visualization import plot_finnish_parties
    
    pyplot.figure(figsize=(10, 8))
    splot = pyplot.subplot()
    plot_finnish_parties(reduced_dim_data, splot=splot)
    pyplot.title("Finnish Political Parties")
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "finnish_parties.png"]))
    print("Finnish parties plot saved to: plots/finnish_parties.png")

    print("Analysis Complete")
