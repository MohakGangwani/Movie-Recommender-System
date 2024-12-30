from src.utils import *
import time

def main():
    a = time.time()
    install_spacy_model()

    params = load_params()

    # Load or preprocess data
    df_preprocessed = load_preprocess_data(params)

    # Load or generate vector
    vector = get_vector(df_preprocessed, params)

    # Load or create similarity model
    sim_model = get_sim_model(vector, params)
    
    print("Time Taken:")
    print(time.time()-a)
    return df_preprocessed, vector, sim_model

if __name__ == "__main__":
    main()