import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re
import datetime
import json
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models


# ---------------------------
# Helpers
# ---------------------------
def safe_convert(df, col, convert_func, **kwargs):
    if col in df.columns:
        df[col] = df[col].apply(convert_func, **kwargs)
    return df


def safe_fill(df, col, fill_value, dtype='category'):
    if col in df.columns:
        df[col] = df[col].fillna(fill_value).astype(dtype)
    return df


def safe_datetime(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
def load_data(filename, columns):
    return pd.read_csv(filename, usecols=columns)


def parse_delay(val):
    if pd.isnull(val):
        return np.nan
    match = re.search(r'(\d+)', str(val))
    return float(match.group(1)) if match else np.nan


def process_numeric(df):
    if 'Number_Of_Students_On_The_Bus' in df.columns:
        df['Number_Of_Students_On_The_Bus'] = pd.to_numeric(df['Number_Of_Students_On_The_Bus'], errors='coerce')
    if 'How_Long_Delayed' in df.columns:
        df['How_Long_Delayed'] = df['How_Long_Delayed'].apply(parse_delay)
    return df


def process_categorical(df, cat_cols):
    for col in cat_cols:
        safe_fill(df, col, 'Unknown')
    return df


def process_dates(df, date_cols):
    for col in date_cols:
        safe_datetime(df, col)
    return df


def normalize_numerics(df, num_cols):
    existing = [col for col in num_cols if col in df.columns]
    if existing:
        scaler = MinMaxScaler()
        df[existing] = scaler.fit_transform(df[existing].fillna(0))
    return df


def preprocess_data(filename):
    cols = [
        'School_Year', 'Busbreakdown_ID', 'Run_Type', 'Bus_No', 'Route_Number', 'Reason',
        'Schools_Serviced', 'Occurred_On', 'Created_On', 'Boro', 'Bus_Company_Name',
        'How_Long_Delayed', 'Number_Of_Students_On_The_Bus',
        'Has_Contractor_Notified_Schools', 'Has_Contractor_Notified_Parents',
        'Have_You_Alerted_OPT', 'Informed_On', 'Incident_Number', 'Last_Updated_On',
        'Breakdown_or_Running_Late', 'School_Age_or_PreK'
    ]
    df = load_data(filename, cols)
    df = process_numeric(df)
    df = process_categorical(df, [
        'Bus_No', 'Route_Number', 'Reason', 'Bus_Company_Name',
        'Boro', 'School_Age_or_PreK', 'Breakdown_or_Running_Late',
        'Have_You_Alerted_OPT', 'Has_Contractor_Notified_Schools',
        'Has_Contractor_Notified_Parents', 'Informed_On'
    ])
    df = process_dates(df, ['Occurred_On', 'Created_On', 'Informed_On', 'Last_Updated_On'])
    df = normalize_numerics(df, ['Number_Of_Students_On_The_Bus', 'How_Long_Delayed'])
    return df


def augment_data(real_df, synthetic_df):
    synthetic_full = synthetic_df.reindex(columns=real_df.columns)
    return pd.concat([real_df, synthetic_full], ignore_index=True)


# ---------------------------
# Bayesian Network Functions
# ---------------------------
def build_bn(df, bn_nodes, potential_edges):
    existing_nodes = [node for node in bn_nodes if node in df.columns]
    G = nx.DiGraph()
    G.add_nodes_from(existing_nodes)
    for parent, child in potential_edges:
        if parent in existing_nodes and child in existing_nodes:
            G.add_edge(parent, child)
    return G


def convert_to_category(series):
    if series.dtype.name == 'category':
        return series.cat.codes
    return pd.cut(series, bins=3, labels=False)


def discretize_categorical(series):
    return convert_to_category(series)


def estimate_cpd(child, parent=None):
    if parent is None:
        counts = child.value_counts()
        # Перетворюємо ключі у int
        return {int(k): v for k, v in (counts / counts.sum()).to_dict().items()}
    cpd = {}
    temp = pd.DataFrame({'child': child, 'parent': parent}).dropna()
    for p_val in temp['parent'].unique():
        subset = temp[temp['parent'] == p_val]['child']
        counts = subset.value_counts()
        cpd[int(p_val)] = {int(k): v for k, v in (counts / counts.sum()).to_dict().items()} if not subset.empty else {}
    return cpd


def calculate_cpts(df, G, nodes):
    cpts = {}
    for node in nodes:
        parents = list(G.predecessors(node))
        node_data = discretize_categorical(df[node])
        if not parents:
            cpts[node] = estimate_cpd(node_data)
        elif len(parents) == 1:
            parent_data = discretize_categorical(df[parents[0]])
            cpts[node] = estimate_cpd(node_data, parent_data)
    return cpts


# ---------------------------
# Visualization Function
# ---------------------------
def plot_bn_structure(G):
    if len(G):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', 
                arrowsize=20, font_size=8, font_weight='bold')
        plt.title("Bayesian Network Structure")
        plt.tight_layout()
        plt.show()


# ---------------------------
# GAN for Synthetic Data Generation
# ---------------------------
def build_generator(latent_dim, output_dim):
    return models.Sequential([
        layers.Dense(16, activation='relu', input_dim=latent_dim),
        layers.Dense(32, activation='relu'),
        layers.Dense(output_dim, activation='linear')
    ])


def build_discriminator(input_dim):
    model = models.Sequential([
        layers.Dense(32, activation='relu', input_dim=input_dim),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def generate_synthetic_data(real_data, synthetic_frac=0.2, latent_dim=5, epochs=10, batch_size=10000):
    features = ['How_Long_Delayed', 'Number_Of_Students_On_The_Bus']
    data = real_data[features].dropna().values
    data_dim = data.shape[1]

    generator = build_generator(latent_dim, data_dim)
    discriminator = build_discriminator(data_dim)

    # Build GAN model
    gan_input = layers.Input(shape=(latent_dim,))
    fake_data = generator(gan_input)
    gan_output = discriminator(fake_data)
    gan = models.Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    # Pre-create labels for efficiency
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    
    # Training loop
    for epoch in range(epochs):
        for _ in range(batch_size):
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_batch = data[idx]
            
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_batch = generator.predict(noise, verbose=0)
            
            discriminator.train_on_batch(real_batch, real_labels)
            discriminator.train_on_batch(fake_batch, fake_labels)
            
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            gan.train_on_batch(noise, real_labels)

    n_synthetic = int(synthetic_frac * data.shape[0])
    noise = np.random.normal(0, 1, (n_synthetic, latent_dim))
    synthetic_samples = generator.predict(noise, verbose=0)
    synthetic_df = pd.DataFrame(synthetic_samples, columns=features).clip(0, 1)
    return synthetic_df

def export_bn_to_json(G, cpts):
    bn_dict = {
        "nodes": list(G.nodes()),
        "edges": []
    }
    for parent, child in G.edges():
        edge_info = {
            "parent": parent,
            "child": child
        }
        # Додаємо CPD, якщо він був обчислений для дитини
        if child in cpts:
            edge_info["cpd"] = cpts[child]
        bn_dict["edges"].append(edge_info)
    return json.dumps(bn_dict, indent=2, default=str)

# ---------------------------
# Main Function
# ---------------------------
def main():
    filename = 'data-full.csv'
    df = preprocess_data(filename)

    synthetic_df = generate_synthetic_data(df, synthetic_frac=0.2, epochs=10)
    df_augmented = augment_data(df, synthetic_df)

    bn_nodes = [
        'School_Age_or_PreK', 'Reason', 'Breakdown_or_Running_Late',
        'Bus_Company_Name', 'Bus_No', 'Route_Number',
        'Has_Contractor_Notified_Schools', 'Have_You_Alerted_OPT'
    ]
    potential_edges = [
        ('School_Age_or_PreK', 'Reason'),
        ('Reason', 'Breakdown_or_Running_Late'),
        ('Breakdown_or_Running_Late', 'Has_Contractor_Notified_Schools'),
        ('Bus_Company_Name', 'Bus_No'),
        ('Bus_No', 'Route_Number'),
        ('Breakdown_or_Running_Late', 'Have_You_Alerted_OPT')
    ]
    G = build_bn(df_augmented, bn_nodes, potential_edges)

    plot_bn_structure(G)

    existing_nodes = [node for node in bn_nodes if node in df_augmented.columns]
    cpts = calculate_cpts(df_augmented, G, existing_nodes)

    # Convert CPTs to JSON format before printing
    bn_json = export_bn_to_json(G, cpts)
    with open("bayesian_network.json", "w", encoding="utf-8") as f:
        f.write(bn_json)

    print("Файл bayesian_network.json успішно збережено.")
    print("Повна інформація про мережу у форматі JSON:")
    print(bn_json)

    print("\nNetwork Statistics:")
    print(f"Number of nodes: {len(G.nodes())}")
    print(f"Number of edges: {len(G.edges())}")
    print(f"Nodes: {list(G.nodes())}")
    print(f"Edges: {list(G.edges())}")

    print("\nData Summary:")
    print(f"Total records (including synthetic): {len(df_augmented)}")
    print(f"Columns used in BN: {existing_nodes}")
    print("Categorical missing values filled with 'Unknown'")

    print("\nData Quality Check:")
    for col in existing_nodes:
        missing = df_augmented[col].isnull().sum()
        unique = df_augmented[col].nunique()
        print(f"{col}: {missing} missing values, {unique} unique values")


if __name__ == "__main__":
    main()
